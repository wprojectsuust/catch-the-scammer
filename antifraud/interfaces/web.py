"""Local HTTP adapter. Sessions hold independent immutable analysis snapshots."""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from antifraud.application import AnalyzeOperations, Analysis
from antifraud.domain import Assessment, Client, Dataset, InvalidData
from antifraud.infrastructure.csv_data import load_dataset
from antifraud.infrastructure.feature_engine import CausalFeatureEngine
from antifraud.interfaces.presentation import assessment_dict, export_csv, operation_dict
from antifraud.sandbox import SCENARIOS, run_scenarios, scenario_dataset


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SESSIONS = 4


@dataclass
class Snapshot:
    analysis: Analysis
    chronological: tuple[Assessment, ...]
    ranked: tuple[Assessment, ...]
    by_id: dict[str, Assessment]
    clients: dict[str, Client]
    elapsed_seconds: float


class Sessions:
    def __init__(self):
        self.lock = Lock()
        self.items = OrderedDict()

    def add(self, analysis, dataset, elapsed):
        snapshot = Snapshot(
            analysis,
            tuple(sorted(analysis.assessments, key=lambda item: (item.operation.timestamp, item.operation.transaction_id))),
            tuple(sorted(analysis.assessments, key=lambda item: item.prediction.score, reverse=True)),
            {item.operation.transaction_id: item for item in analysis.assessments},
            {client.client_id: client for client in dataset.clients},
            elapsed,
        )
        identity = uuid4().hex
        with self.lock:
            self.items[identity] = snapshot
            while len(self.items) > MAX_SESSIONS:
                self.items.popitem(last=False)
        return identity, snapshot

    def get(self, identity):
        with self.lock:
            if identity not in self.items:
                raise HTTPException(404, "Анализ не найден или вытеснен из памяти. Загрузите данные снова.")
            self.items.move_to_end(identity)
            return self.items[identity]


class ScenarioInput(BaseModel):
    amount: float | None = Field(default=None, gt=0, le=1_000_000_000, allow_inf_nan=False)


def create_app(model, data_dir=None):
    app = FastAPI(title="Поймай мошенника", version="0.1.0")
    sessions = Sessions()
    service = AnalyzeOperations(CausalFeatureEngine(), model)

    @app.exception_handler(InvalidData)
    async def invalid_data(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(status_code=403, content={"detail": "Запрос с другого сайта отклонён."})
        length = request.headers.get("content-length", "0")
        if not length.isdigit() or int(length) > 3 * MAX_FILE_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Общий размер загрузки превышает 30 МБ."})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def analyze(dataset):
        started = perf_counter()
        analysis = service.execute(dataset)
        identity, snapshot = sessions.add(analysis, dataset, perf_counter() - started)
        return {"id": identity, "count": len(analysis.assessments), "threshold": model.default_threshold,
                "elapsed_seconds": snapshot.elapsed_seconds, "warnings": analysis.warnings}

    def read_upload(upload):
        data = upload.file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{upload.filename}: лимит файла 10 МБ.")
        return data

    @app.get("/api/health")
    def health():
        return {"status": "ok", "available_from": model.available_from.isoformat(),
                "threshold": model.default_threshold, "demo_available": data_dir is not None}

    @app.post("/api/analyses")
    def upload(clients: UploadFile = File(...), history: UploadFile = File(...), current: UploadFile = File(...)):
        return analyze(load_dataset(*(read_upload(item) for item in (clients, history, current))))

    @app.post("/api/demo")
    def demo():
        if data_dir is None:
            raise HTTPException(404, "Каталог демонстрационных данных не задан.")
        try:
            contents = [(Path(data_dir) / name).read_bytes() for name in
                        ("clients.csv", "transactions_history.csv", "transactions_test.csv")]
        except OSError:
            raise HTTPException(404, "Демонстрационные CSV недоступны.") from None
        return analyze(load_dataset(*contents))

    @app.get("/api/analyses/{identity}")
    def queue(identity: str, threshold: float = Query(ge=0, le=1), offset: int = Query(default=0, ge=0),
              limit: int = Query(default=30, ge=1, le=100), seen: int | None = Query(default=None, ge=0),
              review_only: bool = False):
        snapshot = sessions.get(identity)
        visible = snapshot.chronological[:seen]
        ranked = snapshot.ranked if len(visible) == len(snapshot.ranked) else sorted(
            visible, key=lambda item: item.prediction.score, reverse=True
        )
        alerts = [item for item in ranked if item.prediction.score >= threshold]
        ordered = alerts if review_only else ranked
        return {
            "total": len(snapshot.chronological), "observed": len(visible), "review_count": len(alerts),
            "limited_count": sum(bool(item.warnings) for item in visible), "filtered_count": len(ordered),
            "rows": [{key: value for key, value in assessment_dict(item, threshold).items()
                      if key not in ("contributions", "base_value")}
                     for item in ordered[offset:offset + limit]],
            "cutoff": visible[-1].operation.timestamp.isoformat(sep=" ") if visible else None,
        }

    @app.get("/api/analyses/{identity}/operations/{transaction_id}")
    def detail(identity: str, transaction_id: str, threshold: float = Query(ge=0, le=1)):
        snapshot = sessions.get(identity)
        item = snapshot.by_id.get(transaction_id)
        if item is None:
            raise HTTPException(404, "Операция не найдена.")
        prior = snapshot.analysis.prior_operations(item)
        client = snapshot.clients.get(item.operation.customer_id)
        profile = asdict(client) if client else None
        if profile and profile["registration_date"]:
            profile["registration_date"] = profile["registration_date"].isoformat()
        return {**assessment_dict(item, threshold), "client": profile,
                "history_count": len(prior), "history": [operation_dict(op) for op in prior[-50:]]}

    @app.get("/api/analyses/{identity}/export")
    def export(identity: str, threshold: float = Query(ge=0, le=1)):
        return Response(export_csv(sessions.get(identity).analysis, threshold), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="risk-report.csv"'})

    @app.get("/api/scenarios")
    def scenarios():
        return [{"id": name, "title": title, "description": description}
                for name, (title, description) in SCENARIOS.items()]

    @app.post("/api/scenarios/check")
    def scenario_checks():
        return run_scenarios(model)

    @app.post("/api/scenarios/{name}")
    def run_scenario(name: str, options: ScenarioInput):
        dataset = scenario_dataset(name, model.available_from, options.amount)
        return analyze(dataset)

    static = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    app.mount("/static", StaticFiles(directory=static), name="static")
    return app
