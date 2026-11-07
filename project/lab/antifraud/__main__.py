import argparse
from pathlib import Path

from antifraud.domain import InvalidData


def main():
    parser = argparse.ArgumentParser(description="Поймай мошенника — обучение, анализ и лаборатория.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data-dir", type=Path, required=True)
    training.add_argument("--model", type=Path, required=True)
    for name in ("analyze", "sandbox", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        if name == "analyze":
            command.add_argument("--data-dir", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--threshold", type=float)
        if name == "sandbox":
            command.add_argument("--output", type=Path, required=True)
        if name == "serve":
            command.add_argument("--port", type=int, default=8000)
            command.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "train":
            from antifraud.infrastructure.training import train
            result = train(args.data_dir, args.model)
            print(f"Модель сохранена. Порог по валидации: {result['default_threshold']:.6f}")
            return
        from antifraud.infrastructure.model import CatBoostRiskModel
        model = CatBoostRiskModel(args.model)
        if args.command == "sandbox":
            from antifraud.sandbox import run_and_save
            run_and_save(model, args.output)
        elif args.command == "serve":
            import uvicorn
            from antifraud.interfaces.web import create_app
            uvicorn.run(create_app(model, args.data_dir), host="127.0.0.1", port=args.port)
        else:
            from antifraud.application import AnalyzeOperations
            from antifraud.infrastructure.feature_engine import CausalFeatureEngine
            from antifraud.infrastructure.csv_data import load_dataset
            from antifraud.interfaces.presentation import export_csv
            dataset = load_dataset(*((args.data_dir / name).read_bytes() for name in
                                    ("clients.csv", "transactions_history.csv", "transactions_test.csv")))
            analysis = AnalyzeOperations(CausalFeatureEngine(), model).execute(dataset)
            threshold = model.default_threshold if args.threshold is None else args.threshold
            output = export_csv(analysis, threshold)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(output)
            print(f"Оценено операций: {len(analysis.assessments)}. Результат: {args.output}")
    except (InvalidData, ValueError, OSError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
