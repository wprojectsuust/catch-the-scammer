const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 });
const money = value => `${number.format(value)} ₽`;
const state = { session: null, offset: 0, limit: 25, epoch: 0, detailEpoch: 0, playback: null };

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function notify(message, error = false) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
  $("notice").classList.toggle("error", error);
}

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : "Проверьте параметры или содержимое файлов.";
    throw new Error(message);
  }
  return data;
}

async function action(button, task) {
  button.disabled = true;
  try { await task(); }
  catch (error) { notify(error.message, true); }
  finally { button.disabled = false; }
}

function tab(showLab) {
  $("lab-panel").hidden = !showLab;
  $("queue-panel").hidden = showLab;
  $("lab-tab").setAttribute("aria-pressed", String(showLab));
  $("queue-tab").setAttribute("aria-pressed", String(!showLab));
  if (showLab) stopPlayback();
}

function stopPlayback() {
  clearTimeout(state.playback);
  state.playback = null;
  $("play").textContent = "Воспроизвести";
}

function resetDetail() {
  state.detailEpoch += 1;
  $("detail").replaceChildren(element("p", "Выберите операцию для разбора."));
}

async function openSession(result, title) {
  stopPlayback();
  state.session = result;
  state.offset = 0;
  $("dataset-title").textContent = title;
  $("threshold").value = result.threshold;
  $("seen").max = result.count;
  $("seen").value = result.count;
  $("review-only").checked = false;
  $("results").hidden = false;
  $("empty").hidden = true;
  resetDetail();
  tab(false);
  notify([`Оценено операций: ${number.format(result.count)}.`, ...result.warnings].join(" "));
  await refresh();
}

async function refresh() {
  if (!state.session) return;
  const epoch = ++state.epoch;
  const base = `/api/analyses/${state.session.id}`;
  const threshold = $("threshold").value;
  const params = new URLSearchParams({
    threshold, offset: state.offset, limit: state.limit, seen: $("seen").value,
    review_only: $("review-only").checked,
  });
  const report = await request(`${base}?${params}`);
  if (epoch !== state.epoch) return;
  $("observed").textContent = `${number.format(report.observed)} / ${number.format(report.total)}`;
  $("alert-count").textContent = number.format(report.review_count);
  $("limited-count").textContent = number.format(report.limited_count);
  $("cutoff").textContent = report.cutoff || "До первой операции";
  $("export").href = `${base}/export?threshold=${encodeURIComponent(threshold)}`;
  const rows = report.rows.map(row => {
    const tr = element("tr");
    const link = element("button", row.transaction_id);
    link.addEventListener("click", () => action(link, () => showDetail(base, row.transaction_id, threshold)));
    const id = element("td");
    id.append(link);
    const decision = element("td");
    decision.append(element("span", row.review ? "На проверку" : "Ниже порога", row.review ? "chip alert" : "chip"));
    tr.append(id, element("td", row.timestamp), element("td", money(row.amount_rub)),
              element("td", row.score.toFixed(3)), decision);
    return tr;
  });
  if (!rows.length) {
    const empty = element("td", "В этом кадре нет операций для выбранного фильтра.");
    empty.colSpan = 5;
    const tr = element("tr");
    tr.append(empty);
    rows.push(tr);
  }
  $("rows").replaceChildren(...rows);
  $("page-status").textContent = report.filtered_count
    ? `${state.offset + 1}–${Math.min(state.offset + state.limit, report.filtered_count)} из ${report.filtered_count}`
    : "0 результатов";
  $("previous").disabled = state.offset === 0;
  $("next").disabled = state.offset + state.limit >= report.filtered_count;
}

function historyChart(history, currentAmount) {
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", "0 0 360 85");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Суммы прошлых операций и текущая операция");
  const amounts = [...history.map(row => row.amount_rub), currentAmount];
  const scale = Math.max(...amounts);
  const points = amounts.map((value, index) => [8 + 344 * index / (amounts.length - 1), 77 - 68 * value / scale]);
  const line = document.createElementNS(namespace, "polyline");
  line.setAttribute("points", points.map(point => point.join(",")).join(" "));
  line.setAttribute("stroke", "#3155aa");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("fill", "none");
  const dot = document.createElementNS(namespace, "circle");
  dot.setAttribute("cx", points.at(-1)[0]);
  dot.setAttribute("cy", points.at(-1)[1]);
  dot.setAttribute("r", "4");
  dot.setAttribute("fill", "#a5372d");
  svg.append(line, dot);
  return svg;
}

async function showDetail(base, transaction, threshold) {
  const epoch = ++state.detailEpoch;
  const row = await request(`${base}/operations/${encodeURIComponent(transaction)}?threshold=${encodeURIComponent(threshold)}`);
  if (epoch !== state.detailEpoch) return;
  const detail = $("detail");
  detail.replaceChildren(element("h2", row.transaction_id),
    element("p", `${money(row.amount_rub)} · ${row.transaction_type} · ${row.timestamp}`),
    element("p", `Оценка подозрительности ${row.score.toFixed(3)}. ${row.review ? "На проверку." : "Ниже порога."}`));
  if (row.warnings.length) {
    const notes = element("ul");
    notes.append(...row.warnings.map(text => element("li", text)));
    detail.append(element("h3", "Ограничения оценки"), notes);
  }
  detail.append(element("h3", "Вклад признаков CatBoost"), element("p", "Плюс повышает риск. Минус снижает. Шкала вкладов — логиты."));
  const top = row.contributions.slice(0, 6);
  const max = Math.max(0.001, ...top.map(item => Math.abs(item.impact)));
  for (const item of top) {
    const contribution = element("div", undefined, "contrib");
    let value = typeof item.value === "number" ? number.format(item.value) : item.value ?? "пропуск";
    if (item.feature === "amount_log") value = money(row.amount_rub);
    if (item.feature === "income_log" && row.client?.monthly_income_rub != null) value = money(row.client.monthly_income_rub);
    if (item.feature === "balance_fraction" && item.value != null) value = `${number.format(item.value * 100)}%`;
    const bar = element("meter");
    bar.max = max;
    bar.value = Math.abs(item.impact);
    bar.setAttribute("aria-label", item.label);
    contribution.append(element("span", `${item.label} · ${value}`),
      element("strong", `${item.impact > 0 ? "+" : ""}${item.impact.toFixed(2)}`), bar);
    detail.append(contribution);
  }
  detail.append(element("h3", `Клиент ${row.customer_id}`),
    element("p", row.client ? [row.client.account_type, row.client.client_category].filter(Boolean).join(" · ") : "Вне справочника."),
    element("p", `До операции в истории: ${row.history_count}.`));
  if (row.history.length) {
    detail.append(historyChart(row.history, row.amount_rub), element("p", "Красная точка — текущая операция."));
    const recent = element("ul");
    recent.append(...row.history.slice(-5).reverse().map(item =>
      element("li", `${item.timestamp} · ${item.transaction_type} · ${money(item.amount_rub)}`)));
    detail.append(recent);
  }
}

function queueChange(resetPage = true) {
  if (resetPage) state.offset = 0;
  resetDetail();
  refresh().catch(error => notify(error.message, true));
}

async function playbackTick() {
  const slider = $("seen");
  slider.value = Math.min(Number(slider.max), Number(slider.value) + Math.max(1, Math.ceil(Number(slider.max) / 40)));
  state.offset = 0;
  resetDetail();
  try { await refresh(); }
  catch (error) { notify(error.message, true); stopPlayback(); }
  if (state.playback === null) return;
  if (Number(slider.value) >= Number(slider.max)) stopPlayback();
  else state.playback = setTimeout(playbackTick, 350);
}

$("queue-tab").onclick = () => tab(false);
$("lab-tab").onclick = () => tab(true);
$("upload-open").onclick = () => $("upload-dialog").showModal();
$("upload-close").onclick = () => $("upload-dialog").close();
$("upload-form").onsubmit = event => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  $("upload-dialog").close();
  action($("upload-open"), async () => {
    notify("Анализируем операции…");
    await openSession(await request("/api/analyses", { method: "POST", body: data }), "Операции по риску");
  });
};
$("demo").onclick = () => action($("demo"), async () => {
  notify("Анализируем данные хакатона…");
  await openSession(await request("/api/demo", { method: "POST" }), "Операции хакатона по риску");
});
$("threshold").onchange = () => {
  if ($("threshold").reportValidity() && $("threshold").value !== "") queueChange();
};
$("review-only").onchange = () => queueChange();
$("seen").onchange = () => { stopPlayback(); queueChange(); };
$("previous").onclick = () => { state.offset = Math.max(0, state.offset - state.limit); queueChange(false); };
$("next").onclick = () => { state.offset += state.limit; queueChange(false); };
$("play").onclick = () => {
  if (state.playback !== null) return stopPlayback();
  if ($("seen").value === $("seen").max) $("seen").value = 0;
  $("play").textContent = "Пауза";
  state.playback = setTimeout(playbackTick, 0);
};
$("check-all").onclick = () => action($("check-all"), async () => {
  const report = await request("/api/scenarios/check", { method: "POST" });
  const list = element("ul");
  list.append(...report.scenarios.map(item =>
    element("li", `${item.title} · ${item.passed ? "прошёл" : "ошибка"} · риск ${item.assessment.score.toFixed(3)}`)));
  $("scenario-results").replaceChildren(element("h2", report.passed ? "Все проверки прошли" : "Есть ошибки"), list);
  $("scenario-results").hidden = false;
});

async function init() {
  const [health, scenarios] = await Promise.all([request("/api/health"), request("/api/scenarios")]);
  $("demo").hidden = !health.demo_available;
  $("model-status").textContent = "CatBoost · готов";
  for (const scenario of scenarios) {
    const card = element("article", undefined, "card");
    const button = element("button", "Запустить");
    button.onclick = () => action(button, async () => {
      const input = $("scenario-amount");
      if (!input.reportValidity()) return;
      const body = JSON.stringify({ amount: input.value ? Number(input.value) : null });
      const result = await request(`/api/scenarios/${scenario.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      });
      await openSession(result, scenario.title);
      await showDetail(`/api/analyses/${result.id}`, "DEMO-CURRENT", result.threshold);
    });
    card.append(element("h2", scenario.title), element("p", scenario.description), button);
    $("scenarios").append(card);
  }
}
init().catch(error => notify(error.message, true));
