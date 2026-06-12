// mindXtrain Coach — vanilla-JS state machine.
// No framework, no build step. Talk to /coach/api/* over fetch.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const state = {
  recipe: null,        // selected recipe name
  plan: null,          // AutotunePlan from /api/bench
  compileResult: null, // CompileResponse from /api/compile
  run: null,           // active Run from /api/runs/launch
  eventSource: null,   // active EventSource for the active run
  chart: null,         // Chart.js instance, if Chart is available
  logLines: 0,         // capped at MAX_LOG_LINES client-side
  preflightReady: false,
  corpusReady: false,
  hardware: null,      // most recent HardwareProfile from /coach/api/diagnostics/hardware
  mei: null,           // most recent MEIScoreView from /coach/api/mei/score
  meiChart: null,      // Chart.js radar instance for the MEI sub-indices
  chatBackendModel: "",  // detected ollama/vllm model name (drives /v1/chat/completions body)
  metrics: [],         // rolling buffer of MetricsEvent samples (capped METRICS_BUFFER_CAP)
  metricsTimer: null,  // setInterval handle for elapsed-time counter
  totalSteps: null,    // StepEvent.total_steps — drives the progress bar
  firstStepTs: null,   // wall-clock ms at the first observed step (for ETA)
  firstStep: 0,        // step number of that first observed step
  stepBuffer: [],      // {step,loss,mean_token_accuracy} for the result banner
};

const METRICS_BUFFER_CAP = 300;

const MAX_LOG_LINES = 2000;

// Per-step rows kept in the (collapsed) metrics table DOM. The full count is
// always reported in the accordion summary so the cap is never silent.
const MAX_TABLE_ROWS = 50;

// Loss-chart points retained. A real MI300X run logs thousands of steps; an
// unbounded canvas dataset makes the page janky. We keep a rolling window and
// surface "showing last X of N" so the compression is honest, not hidden.
const MAX_CHART_POINTS = 1500;

// When a log accumulates more than this many lines, auto-collapse the
// oldest ~80% into a folded sub-accordion so the operator's eye stays on
// the most recent activity. Tuned to keep the "live tail" UX legible on
// a 13-15" laptop screen.
const LOG_FOLD_THRESHOLD = 400;
const LOG_FOLD_KEEP_RECENT = 80;

function _foldLogElement(pre) {
  // Idempotent: collapses lines older than the most recent
  // LOG_FOLD_KEEP_RECENT once `pre` has > LOG_FOLD_THRESHOLD child text
  // nodes. Replays produce the same DOM so callers can call this on
  // every `appendLog` without rebuilding the world.
  const children = Array.from(pre.childNodes).filter(
    n => n.nodeType === Node.TEXT_NODE,
  );
  if (children.length <= LOG_FOLD_THRESHOLD) return;
  // Already folded earlier? Look for the sentinel <details> at the start.
  if (pre.firstChild && pre.firstChild.tagName === "DETAILS") {
    // The fold exists; just keep growing the visible tail. We re-fold
    // periodically by reading the visible-tail node count and migrating
    // overflow into the existing details summary.
    const det = pre.firstChild;
    const oldSummary = det.querySelector("summary");
    const folded = det.querySelector("pre");
    const recent = children.slice(LOG_FOLD_KEEP_RECENT * -1);
    const earlier = children.slice(0, -recent.length);
    for (const n of earlier) {
      // Move into the folded pre.
      if (n !== folded && (n.previousSibling !== det && n.parentNode === pre)) {
        folded.appendChild(n);
      }
    }
    const count = folded.childNodes.length;
    oldSummary.textContent = `▸ ${count} earlier lines (expand)`;
    return;
  }
  // First fold for this pre.
  const recent = children.slice(LOG_FOLD_KEEP_RECENT * -1);
  const earlier = children.slice(0, -recent.length);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `▸ ${earlier.length} earlier lines (expand)`;
  const foldedPre = document.createElement("pre");
  foldedPre.className = "log-tail folded";
  for (const n of earlier) {
    foldedPre.appendChild(n);
  }
  details.appendChild(summary);
  details.appendChild(foldedPre);
  // Insert the fold at the top.
  pre.insertBefore(details, pre.firstChild);
}

// Ordered step ids — used by progressTo to compute "next" if a caller
// doesn't pass one explicitly, and by syncPipelineHeader to pick a stage.
const STEP_ORDER = [
  "step-preflight",
  "step-dream-corpus",
  "step-recipes",
  "step-autotune",
  "step-compile",
  "step-deploy",
  "step-train",
  "step-chat",
];

// Map step ids to one of the three header pipeline stages.
const STAGE_FOR_STEP = {
  "step-preflight":    "automind",
  "step-hardware":     "automind",
  "step-dream-corpus": "automind",
  "step-create-dataset": "automind",
  "step-recipes":      "mind",
  "step-autotune":     "mind",
  "step-compile":      "mind",
  "step-deploy":       "mind",
  "step-train":        "mind",
  "step-receipt":      "cust",
  "step-boardroom":    "cust",
  "step-mei":          "cust",
  "step-chat":         "cust",
};

// --- auto-advance helpers ------------------------------------------------

function syncPipelineHeader(activeStepId) {
  const want = STAGE_FOR_STEP[activeStepId];
  if (!want) return;
  $$(".pipeline .stage").forEach((node) => {
    node.classList.toggle("active", node.dataset.stage === want);
  });
}

// Mark previous active card .done, mark `id` .active, scroll into view, sync
// the header pipeline. Safe to call repeatedly; idempotent for the same id.
function progressTo(id) {
  const next = document.getElementById(id);
  if (!next) return;
  $$("section.card.active").forEach((c) => {
    if (c.id !== id) {
      c.classList.remove("active");
      c.classList.add("done");
    }
  });
  next.classList.remove("done");
  next.classList.add("active");
  next.scrollIntoView({ behavior: "smooth", block: "start" });
  syncPipelineHeader(id);
}

function markCardDone(id) {
  const node = document.getElementById(id);
  if (!node) return;
  node.classList.remove("active");
  node.classList.add("done");
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

// --- step 1: preflight env --------------------------------------------------

async function runPreflight() {
  const btn = $("#run-preflight");
  const summary = $("#preflight-summary");
  const list = $("#preflight-list");
  const badge = $("#preflight-badge");
  if (btn) btn.disabled = true;
  summary.textContent = "checking…";
  try {
    const res = await getJSON("/coach/api/preflight");
    list.innerHTML = "";
    for (const name of [...res.required, ...res.optional]) {
      const present = !!res.vars[name];
      const li = document.createElement("li");
      const mark = present ? "✓" : "✗";
      const cls = present ? "fits-yes" : "fits-no";
      const tag = res.required.includes(name) ? "required" : "optional";
      li.innerHTML = `<span class="${cls}">${mark}</span> <code>${name}</code> <span class="muted">${tag}</span>`;
      list.appendChild(li);
    }
    state.preflightReady = res.ready;
    if (res.ready) {
      summary.textContent = `all required env vars set (${res.required.length})`;
      badge.textContent = "ready";
      badge.className = "badge-status succeeded";
      badge.hidden = false;
      markCardDone("step-preflight");
      // Auto-advance: env check → hardware probe → corpus check.
      progressTo("step-hardware");
      await runHardware();
      progressTo("step-dream-corpus");
      runDreamCorpus();
    } else {
      summary.textContent = `missing: ${res.required_missing.join(", ")}`;
      badge.textContent = "not ready";
      badge.className = "badge-status failed";
      badge.hidden = false;
    }
  } catch (e) {
    summary.textContent = `preflight probe failed: ${e}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

// --- advanced admin (always-on-top diagnostics) --------------------------

const ADMIN_POLL_MS = 2000;
const ADMIN_FIREHOSE_MAX = 800;  // hard cap on lines in admin firehose
const adminState = {
  pollTimer: null,
  paused: false,
  firehoseSources: new Map(),  // run_id → EventSource
  firehoseLines: 0,
};

function _adminMetricCell(label, value, statusClass) {
  return `<div class="admin-metric ${statusClass || ""}">` +
         `<div class="admin-metric-label">${label}</div>` +
         `<div class="admin-metric-value">${value}</div></div>`;
}

function _classifyPct(pct, warn = 70, bad = 90) {
  if (pct >= bad) return "bad";
  if (pct >= warn) return "warn";
  return "";
}

function _classifyLoad(load, cores) {
  if (load == null) return "";
  const ratio = load / Math.max(1, cores);
  if (ratio >= 1.5) return "bad";
  if (ratio >= 1.0) return "warn";
  return "";
}

async function refreshAdminMetrics() {
  if (adminState.paused) return;
  try {
    const m = await getJSON("/coach/api/diagnostics/live");
    const grid = $("#admin-metrics");
    if (!grid) return;
    const cores = m.cores || 1;
    const load1 = m.load_avg_1m;
    const cells = [
      _adminMetricCell("Load 1m", load1 == null ? "—" : `${load1.toFixed(2)} / ${cores}`,
                       _classifyLoad(load1, cores)),
      _adminMetricCell("Load 5m", m.load_avg_5m == null ? "—" : m.load_avg_5m.toFixed(2),
                       _classifyLoad(m.load_avg_5m, cores)),
      _adminMetricCell("Load 15m", m.load_avg_15m == null ? "—" : m.load_avg_15m.toFixed(2),
                       _classifyLoad(m.load_avg_15m, cores)),
      _adminMetricCell("RAM used",
                       `${m.ram_used_pct.toFixed(1)}% (${m.ram_available_gb.toFixed(1)} GB free)`,
                       _classifyPct(m.ram_used_pct)),
      _adminMetricCell("Disk used",
                       `${m.disk_used_pct.toFixed(1)}% of ${m.disk_total_gb.toFixed(0)} GB`,
                       _classifyPct(m.disk_used_pct, 80, 95)),
      _adminMetricCell("Operator RSS", `${m.operator_rss_mb.toFixed(0)} MB`, ""),
      _adminMetricCell("Operator threads", String(m.operator_threads), ""),
      _adminMetricCell("Cores", String(cores), ""),
    ];
    grid.innerHTML = cells.join("");
  } catch (e) {
    // Silent; the admin card is non-critical.
  }
}

function _fmtRunStart(iso) {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch (_) {
    return iso;
  }
}

async function refreshAdminRuns() {
  if (adminState.paused) return;
  try {
    const runs = await getJSON("/coach/api/diagnostics/runs");
    const tbody = $("#admin-runs-table tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    for (const r of runs.slice(0, 12)) {
      const tr = document.createElement("tr");
      const lossStr = r.last_loss == null ? "—" : r.last_loss.toFixed(4);
      tr.innerHTML =
        `<td class="mono">${r.id}</td>` +
        `<td>${r.recipe}</td>` +
        `<td class="status-${r.status}">${r.status}</td>` +
        `<td>${_fmtRunStart(r.created_at)}</td>` +
        `<td>${r.last_step ?? "—"}</td>` +
        `<td>${lossStr}</td>`;
      tbody.appendChild(tr);
      // Auto-subscribe firehose to any run we haven't seen yet.
      if (!adminState.firehoseSources.has(r.id) &&
          ["pending", "running"].includes(r.status)) {
        _adminSubscribeFirehose(r.id);
      }
    }
  } catch (e) { /* non-critical */ }
}

function _adminAppendFirehose(line) {
  const pre = $("#admin-firehose");
  if (!pre) return;
  pre.appendChild(document.createTextNode(line + "\n"));
  adminState.firehoseLines += 1;
  while (adminState.firehoseLines > ADMIN_FIREHOSE_MAX && pre.firstChild) {
    pre.removeChild(pre.firstChild);
    adminState.firehoseLines -= 1;
  }
  if (adminState.firehoseLines % 50 === 0) {
    _foldLogElement(pre);
  }
  pre.scrollTop = pre.scrollHeight;
  $("#admin-firehose-summary").textContent =
    `(${adminState.firehoseLines} lines · ${adminState.firehoseSources.size} runs)`;
}

function _adminSubscribeFirehose(runId) {
  if (adminState.firehoseSources.has(runId)) return;
  const es = new EventSource(`/coach/api/runs/${runId}/events`);
  adminState.firehoseSources.set(runId, es);
  const short = runId.slice(0, 8);
  es.addEventListener("status", (e) => {
    const ev = JSON.parse(e.data);
    _adminAppendFirehose(`[${short}] status → ${ev.status}: ${ev.message || ""}`);
    if (["succeeded", "failed", "cancelled"].includes(ev.status)) {
      es.close();
      adminState.firehoseSources.delete(runId);
    }
  });
  es.addEventListener("log", (e) => {
    const ev = JSON.parse(e.data);
    _adminAppendFirehose(`[${short}] ${ev.line}`);
  });
  es.addEventListener("step", (e) => {
    const ev = JSON.parse(e.data);
    _adminAppendFirehose(
      `[${short}] step=${ev.step} loss=${(ev.loss || 0).toFixed(4)} ` +
      `lr=${ev.lr || "—"} grad_norm=${ev.grad_norm || "—"}`,
    );
  });
  es.addEventListener("eval", (e) => {
    const ev = JSON.parse(e.data);
    _adminAppendFirehose(`[${short}] eval@${ev.step} ${JSON.stringify(ev.metrics)}`);
  });
  es.addEventListener("error", () => {
    _adminAppendFirehose(`[${short}] (event stream disconnected)`);
  });
}

function _adminStartPolling() {
  if (adminState.pollTimer != null) return;
  refreshAdminMetrics();
  refreshAdminRuns();
  adminState.pollTimer = setInterval(() => {
    refreshAdminMetrics();
    refreshAdminRuns();
  }, ADMIN_POLL_MS);
}

function _adminStopPolling() {
  if (adminState.pollTimer != null) {
    clearInterval(adminState.pollTimer);
    adminState.pollTimer = null;
  }
}

function _adminTogglePoll() {
  adminState.paused = !adminState.paused;
  const btn = $("#admin-toggle-poll");
  if (btn) btn.textContent = adminState.paused ? "Resume polling" : "Pause polling";
}

function _adminClearFirehose() {
  const pre = $("#admin-firehose");
  if (pre) pre.textContent = "";
  adminState.firehoseLines = 0;
  $("#admin-firehose-summary").textContent =
    `(0 lines · ${adminState.firehoseSources.size} runs)`;
}

// --- step 2: hardware diagnostics -----------------------------------------

function _hwPanelHTML(title, info, fields) {
  const cls = info.available && (fields.gpus ? fields.gpus.length > 0 : true)
    ? "hw-panel hw-ok"
    : "hw-panel hw-off";
  const fieldRows = (fields.entries || []).map(
    ([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`,
  ).join("");
  const noteHTML = info.note
    ? `<p class="hint hw-note">${info.note}</p>` : "";
  return `
    <div class="${cls}">
      <h3>${title}</h3>
      <dl>${fieldRows}</dl>
      ${noteHTML}
    </div>`;
}

function _fmtGB(v) {
  return typeof v === "number" ? `${v.toFixed(1)} GB` : "—";
}

async function runHardware() {
  const summary = $("#hardware-summary");
  const grid = $("#hardware-grid");
  const rec = $("#hardware-recommendation");
  const btn = $("#run-hardware");
  if (btn) btn.disabled = true;
  summary.textContent = "probing…";
  try {
    const p = await getJSON("/coach/api/diagnostics/hardware");
    state.hardware = p;
    // CPU panel — always available.
    const cpu = p.cpu || {};
    const cpuEntries = [
      ["model", cpu.model_name || "CPU"],
      ["vendor", cpu.vendor || "—"],
      ["cores", `${cpu.cores || 0} (Ryzen: ${cpu.is_ryzen ? "yes" : "no"})`],
      ["RAM", `${_fmtGB(cpu.ram_available_gb)} avail / ${_fmtGB(cpu.ram_total_gb)} total`],
      ["load 1m", cpu.load_avg_1m == null ? "—" : cpu.load_avg_1m.toFixed(2)],
    ];
    // AMD panel.
    const amd = p.amd || {};
    const amdGPUs = amd.gpus || [];
    const amdEntries = amd.available
      ? [
          ["ROCm", amd.rocm_version || "—"],
          ...amdGPUs.map((g, i) => [`gpu${i}`, `${g.name} · ${_fmtGB(g.vram_gb)}`]),
        ]
      : [["status", "not detected"]];
    // NVIDIA panel.
    const nv = p.nvidia || {};
    const nvGPUs = nv.gpus || [];
    const nvEntries = nv.available
      ? [
          ["driver", nv.driver_version || "—"],
          ["CUDA", nv.cuda_version || "—"],
          ...nvGPUs.map((g, i) => [`gpu${i}`, `${g.name} · ${_fmtGB(g.vram_gb)}`]),
        ]
      : [["status", "not detected"]];

    grid.innerHTML =
      _hwPanelHTML("CPU", cpu, { entries: cpuEntries }) +
      _hwPanelHTML("AMD GPU", amd, { entries: amdEntries, gpus: amdGPUs }) +
      _hwPanelHTML("NVIDIA GPU", nv, { entries: nvEntries, gpus: nvGPUs });

    const laneLabel = {
      "axolotl_amd": "AMD MI300X (axolotl)",
      "trl_local": "local GPU (trl_local)",
      "trl_cpu": "CPU (trl_cpu)",
    }[p.recommended_lane] || p.recommended_lane;
    rec.innerHTML = `Recommended lane: <strong>${laneLabel}</strong>`;
    rec.hidden = false;
    summary.textContent = `recommended: ${laneLabel}`;
    markCardDone("step-hardware");

    // Auto-suggest a matching recipe so the operator gets a one-click
    // training start. Recipe ↔ lane mapping:
    //   trl_cpu     → mindx_fallback_qwen3_1_5b_cpu_real  (full ~2 hr CPU fine-tune)
    //   trl_local   → mindx_fallback_qwen3_1_5b_local     (device-aware: consumer GPU else CPU)
    //   axolotl_amd → mindx_fallback_qwen3_1_5b_sft_lora  (1× MI300X)
    // The _smoke recipe stays available in the grid for tests + CI — UI
    // just doesn't recommend it because it won't actually adapt the model.
    // If the recipes haven't loaded yet, retry briefly — loadRecipes() is
    // racing with us on page bootstrap.
    const laneToRecipe = {
      "trl_cpu": "mindx_fallback_qwen3_1_5b_cpu_real",
      "trl_local": "mindx_fallback_qwen3_1_5b_local",
      "axolotl_amd": "mindx_fallback_qwen3_1_5b_sft_lora",
    };
    const recipeName = laneToRecipe[p.recommended_lane];
    if (recipeName) {
      _autoSelectRecipeWhenReady(recipeName, 0);
    }
  } catch (e) {
    summary.textContent = `probe failed: ${e}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

function _autoSelectRecipeWhenReady(name, attempt) {
  // Promote the hardware-recommended recipe to the prominent default slot.
  _recommendedRecipe = name;
  renderDefaultRecipe();
  const card = document.querySelector(`#recipe-list .recipe[data-name="${name}"]`);
  if (card) {
    card.classList.add("recommended");
    // Don't auto-click — the user picks deliberately (no surprise launch).
    return;
  }
  if (attempt < 20) {
    // loadRecipes() may still be in flight; retry up to ~4 seconds.
    setTimeout(() => _autoSelectRecipeWhenReady(name, attempt + 1), 200);
  }
}

// --- step 3: dream corpus ---------------------------------------------------

async function runDreamCorpus() {
  const summary = $("#corpus-summary");
  const stats = $("#corpus-stats");
  const note = $("#corpus-note");
  summary.textContent = "counting…";
  try {
    const res = await getJSON("/coach/api/dream-corpus");
    stats.innerHTML = "";
    const con = res.consolidation || { files: 0, raw_lines: 0, unique_rows: 0 };
    const evo = res.evolutions || { files: 0, raw_lines: 0, unique_rows: 0 };
    const fields = [
      ["root", res.root],
      ["consolidation", `${con.unique_rows} unique / ${con.files} files`],
      ["evolutions", `${evo.unique_rows} unique / ${evo.files} files`],
    ];
    for (const [k, v] of fields) {
      const li = document.createElement("li");
      li.innerHTML = `<code>${k}</code>=${v}`;
      stats.appendChild(li);
    }
    stats.hidden = false;
    state.corpusReady = res.ready;
    if (res.note) {
      note.textContent = res.note;
      note.hidden = false;
    } else {
      note.hidden = true;
    }
    if (res.ready) {
      const total = con.unique_rows + evo.unique_rows;
      const detail = evo.unique_rows > 0
        ? `${con.unique_rows} consolidation + ${evo.unique_rows} evolution`
        : `${con.unique_rows} consolidation`;
      summary.textContent = `${total} unique examples ready (${detail})`;
      markCardDone("step-dream-corpus");
      progressTo("step-recipes");
    } else {
      summary.textContent = "corpus not ready — see note";
    }
  } catch (e) {
    summary.textContent = `corpus probe failed: ${e}`;
  }
}

// --- step 3: recipes -----------------------------------------------------

// The default shown front-and-center until hardware recommends one. Device-aware
// `trl_local` runs on a laptop or a GPU unchanged, so it's the safe default.
const DEFAULT_RECIPE = "mindx_fallback_qwen3_1_5b_local";
let _recipeCache = [];
let _recommendedRecipe = null;

function _recipeCardEl(r, isDefault) {
  const div = document.createElement("div");
  div.className = "recipe" + (isDefault ? " recommended" : "");
  div.dataset.name = r.name;
  div.innerHTML = `
      <h3>${r.name}</h3>
      <div class="meta">
        <span class="badge">${r.method}</span>
        <span class="badge">${r.gpus}× GPU</span>
        ${r.base_model}
      </div>`;
  div.addEventListener("click", () => selectRecipe(r.name));
  return div;
}

function renderDefaultRecipe() {
  const host = $("#recipe-default");
  if (!host || !_recipeCache.length) return;
  const name = _recommendedRecipe || DEFAULT_RECIPE;
  const r = _recipeCache.find((x) => x.name === name) || _recipeCache[0];
  host.innerHTML = "";
  host.appendChild(_recipeCardEl(r, true));
}

async function loadRecipes() {
  const list = await getJSON("/coach/api/recipes");
  _recipeCache = list;
  $("#recipe-count").textContent = `(${list.length} total)`;
  // Full list lives in the collapsed "other recipes" accordion…
  const target = $("#recipe-list");
  target.innerHTML = "";
  for (const r of list) target.appendChild(_recipeCardEl(r, false));
  // …and the default/recommended one is shown prominently.
  renderDefaultRecipe();
}

async function selectRecipe(name) {
  state.recipe = name;
  state.compileResult = null;
  for (const node of $$(".recipe")) {
    node.classList.toggle("selected", node.dataset.name === name);
  }
  const detail = await getJSON(`/coach/api/recipes/${name}`);
  // Keep the detail around so the session headline can surface the
  // recipe's cpu_throttle percent during the run.
  state.recipeDetail = detail;
  $("#recipe-yaml").textContent = detail.yaml;
  const det = $("#recipe-detail");
  det.hidden = false;
  det.open = true;
  $("#run-compile").disabled = state.plan === null;
  // Auto-advance: recipe picked → autotune. Run bench automatically if not
  // already done; users who want to re-pick can click another recipe (this
  // function is re-entrant and resets compileResult).
  markCardDone("step-recipes");
  progressTo("step-autotune");
  if (state.plan === null) {
    runBench();
  }
}

// --- step 2: autotune ----------------------------------------------------

async function runBench() {
  $("#run-bench").disabled = true;
  $("#run-bench").textContent = "probing…";
  try {
    const plan = await postJSON("/coach/api/bench", {});
    state.plan = plan;
    $("#plan-json").textContent = JSON.stringify(plan, null, 2);
    $("#plan-json").hidden = false;
    const sum = $("#plan-summary");
    sum.innerHTML = "";
    const items = [
      ["attention", plan.attention_backend],
      ["gemm", plan.gemm_heuristic],
      ["rccl", plan.rccl_config],
      ["fsdp_shard", plan.fsdp_shard_width],
      ["arch", plan.gpu_arch],
      ["rocm", plan.rocm_version],
    ];
    for (const [k, v] of items) {
      const li = document.createElement("li");
      li.textContent = `${k}=${v}`;
      sum.appendChild(li);
    }
    sum.hidden = false;
    $("#run-compile").disabled = state.recipe === null;
    // Auto-advance: plan in hand → compile.
    if (state.recipe) {
      markCardDone("step-autotune");
      progressTo("step-compile");
      runCompile();
    }
  } finally {
    $("#run-bench").disabled = false;
    $("#run-bench").textContent = "Re-run autotune";
  }
}

// --- step 3: compile -----------------------------------------------------

async function runCompile() {
  if (!state.recipe || !state.plan) return;
  $("#run-compile").disabled = true;
  try {
    const res = await postJSON("/coach/api/compile", {
      recipe: state.recipe,
      plan: state.plan,
    });
    state.compileResult = res;
    const ov = $("#compile-overrides");
    ov.innerHTML = "";
    for (const o of res.overrides) {
      const li = document.createElement("li");
      li.textContent = o;
      ov.appendChild(li);
    }
    ov.hidden = false;
    $("#compile-yaml").textContent = JSON.stringify(res.axolotl_yaml, null, 2);
    $("#compile-yaml").hidden = false;
    $("#run-train").disabled = false;
    // Auto-advance: ready to deploy. Stop here for manual GitHub-push click —
    // we don't auto-push, that's an explicit spend authorization.
    markCardDone("step-compile");
    progressTo("step-deploy");
    const ghBtn = $("#run-github");
    if (ghBtn && !ghBtn.disabled) ghBtn.focus();
  } finally {
    $("#run-compile").disabled = false;
  }
}

// --- step 4: train (live) -----------------------------------------------

function setStatusBadge(label) {
  const el = $("#train-status");
  el.textContent = label;
  el.className = "badge-status " + (label.toLowerCase().replace(/\s+/g, "-"));
  el.hidden = false;
}

function ensureChart() {
  if (state.chart || typeof Chart === "undefined") {
    if (typeof Chart === "undefined") $("#chart-fallback").hidden = false;
    return state.chart;
  }
  const ctx = $("#loss-chart").getContext("2d");
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "loss", data: [], borderWidth: 2, tension: 0.2,
          yAxisID: "y", borderColor: "#f7921e",
        },
        {
          // mean_token_accuracy — "is it learning" signal. Non-trl
          // backends omit it; NaN gaps render cleanly.
          label: "accuracy", data: [], borderWidth: 2, tension: 0.2,
          yAxisID: "yAcc", borderColor: "#2ea043", spanGaps: false,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      scales: {
        x: { title: { display: true, text: "step" } },
        y: {
          position: "left",
          title: { display: true, text: "loss" },
          beginAtZero: false,
        },
        yAcc: {
          position: "right",
          min: 0, max: 1,
          title: { display: true, text: "accuracy" },
          grid: { drawOnChartArea: false },
        },
      },
      plugins: { legend: { display: true } },
    },
  });
  return state.chart;
}

function pushPoint(ev) {
  const acc = ev.mean_token_accuracy;
  const ent = ev.entropy;
  const tbody = $("#metrics-table tbody");
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td>${ev.step}</td><td>${ev.loss.toFixed(4)}</td>` +
    `<td>${acc != null ? acc.toFixed(3) : "&mdash;"}</td>` +
    `<td>${ent != null ? ent.toFixed(3) : "&mdash;"}</td>` +
    `<td>${ev.lr ?? "&mdash;"}</td><td>${ev.grad_norm ?? "&mdash;"}</td>`;
  tbody.appendChild(tr);
  // Cap the DOM table; report the true total in the accordion summary so the
  // cap is visible rather than silent.
  while (tbody.children.length > MAX_TABLE_ROWS) tbody.removeChild(tbody.firstChild);
  // Buffer for the terminal result banner (kept whole so loss A→B is exact).
  state.stepBuffer.push({ step: ev.step, loss: ev.loss, mean_token_accuracy: acc });
  _updateMetricsTableCount(state.stepBuffer.length);

  const chart = ensureChart();
  if (chart) {
    chart.data.labels.push(ev.step);
    chart.data.datasets[0].data.push(ev.loss);
    // accuracy on the 2nd axis — NaN where the backend didn't report it.
    chart.data.datasets[1].data.push(acc != null ? acc : NaN);
    // Roll the window so a thousands-of-steps run stays responsive.
    let trimmed = false;
    while (chart.data.labels.length > MAX_CHART_POINTS) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
      chart.data.datasets[1].data.shift();
      trimmed = true;
    }
    if (trimmed) {
      const note = $("#chart-window-note");
      if (note) {
        note.hidden = false;
        note.textContent =
          `showing last ${MAX_CHART_POINTS} of ${state.stepBuffer.length} steps`;
      }
    }
    chart.update("none");
  }
  _updateProgress(ev);
}

function _updateMetricsTableCount(total) {
  const el = $("#metrics-table-count");
  if (!el) return;
  const plural = total === 1 ? "" : "s";
  el.textContent = total > MAX_TABLE_ROWS
    ? `(${total} step${plural} · last ${MAX_TABLE_ROWS} shown)`
    : `(${total} step${plural})`;
}

function appendLog(ev) {
  const pre = $("#train-log");
  _narratePhase(ev.line);
  pre.appendChild(document.createTextNode(ev.line + "\n"));
  state.logLines += 1;
  if (state.logLines > MAX_LOG_LINES) {
    // Drop the first N text nodes to keep DOM bounded.
    while (state.logLines > MAX_LOG_LINES && pre.firstChild) {
      pre.removeChild(pre.firstChild);
      state.logLines -= 1;
    }
  }
  // Re-fold every 50 lines so the DOM stays compact on long runs.
  if (state.logLines % 50 === 0) {
    _foldLogElement(pre);
  }
  _updateLogCount();
  pre.scrollTop = pre.scrollHeight;
}

function _updateLogCount() {
  const el = $("#train-log-count");
  if (!el) return;
  const n = state.logLines;
  el.textContent = n >= MAX_LOG_LINES
    ? `(${n} lines · oldest dropped)`
    : `(${n} line${n === 1 ? "" : "s"})`;
}

function subscribeRun(runId) {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(`/coach/api/runs/${runId}/events`);
  state.eventSource = es;
  // Reveal the session-metrics tier and backfill the sparklines so
  // they don't sit blank waiting for the first 1 Hz tick.
  _enableSessionMetrics();
  _backfillSessionMetrics(runId);
  es.addEventListener("step", (e) => pushPoint(JSON.parse(e.data)));
  es.addEventListener("eval", (e) => _handleEvalEvent(JSON.parse(e.data)));
  es.addEventListener("log", (e) => appendLog(JSON.parse(e.data)));
  es.addEventListener("metrics", (e) => _handleMetricsEvent(JSON.parse(e.data)));
  es.addEventListener("status", (e) => {
    const ev = JSON.parse(e.data);
    setStatusBadge(ev.status);
    _setSessionStatus(ev.status);
    const terminal = ["succeeded", "failed", "cancelled"].includes(ev.status);
    if (terminal) {
      es.close();
      state.eventSource = null;
      _stopElapsedTimer();
      $("#cancel-train").hidden = true;
      $("#run-train").disabled = false;
      _renderResultBanner(ev.status, ev.message);
      _setPhase(ev.status === "succeeded" ? "Done" : "Failed");
      if (ev.status === "succeeded") {
        const fill = $("#progress-fill");
        if (fill) { fill.style.width = "100%"; fill.classList.add("done"); }
        markCardDone("step-train");
        // Reveal the push-to-ollama button — adapter is on disk and the
        // run registry knows about it. The user can pick a tag and merge
        // before (or in parallel with) MEI scoring.
        const pushWrap = $("#push-to-ollama-wrap");
        if (pushWrap) pushWrap.hidden = false;
        // Receipt was emitted at completion — populate the verification card
        // in place (no scroll; it sits between train and MEI).
        loadReceiptForRun(state.run && state.run.id);
        // Train succeeded — MEI scoring is the gate before promotion.
        progressTo("step-mei");
        loadMEIForRun(state.run && state.run.id);
      }
    }
  });
  es.addEventListener("error", () => {
    setStatusBadge("disconnected");
  });
}

async function runTrain() {
  if (!state.recipe || !state.plan) return;
  $("#run-train").disabled = true;
  $("#train-charts").hidden = false;
  $("#train-log-wrap").hidden = false;
  $("#train-log-wrap").open = true;
  $("#train-log").textContent = "";
  $("#metrics-table tbody").innerHTML = "";
  state.logLines = 0;
  _updateLogCount();
  _updateMetricsTableCount(0);
  const chartNote = $("#chart-window-note");
  if (chartNote) { chartNote.hidden = true; chartNote.textContent = ""; }
  if (state.chart) {
    state.chart.data.labels = [];
    state.chart.data.datasets[0].data = [];
    state.chart.data.datasets[1].data = [];
    state.chart.update("none");
  }
  // Reset the session-metrics tier for the new run.
  state.metrics = [];
  // Reset the realtime-feedback surfaces for the new run.
  state.totalSteps = null;
  state.firstStepTs = null;
  state.firstStep = 0;
  state.stepBuffer = [];
  _resetProgress();
  const resultEl = $("#train-result");
  if (resultEl) { resultEl.hidden = true; resultEl.textContent = ""; }
  _enableSessionMetrics();
  _renderSessionSparklines();
  setStatusBadge("launching");
  _setSessionStatus("launching");
  try {
    const r = await fetch("/coach/api/runs/launch", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ recipe: state.recipe, plan: state.plan }),
    });
    if (!r.ok) {
      const text = await r.text();
      setStatusBadge("failed");
      appendLog({ line: `launch failed (${r.status}): ${text}` });
      $("#run-train").disabled = false;
      return;
    }
    const run = await r.json();
    state.run = run;
    $("#train-id").textContent = `run ${run.id}`;
    $("#cancel-train").hidden = false;
    subscribeRun(run.id);
  } catch (e) {
    setStatusBadge("failed");
    appendLog({ line: `launch error: ${e}` });
    $("#run-train").disabled = false;
  }
}

// --- realtime training feedback: phase, progress, result ----------------
//
// Three surfaces that make a live CPU run legible: a plain-language
// phase line, a step X/N progress bar with an ETA, and a terminal
// outcome banner. Driven by the existing step/log/status SSE events.

// Maps trl_cpu's well-prefixed log lines to friendly phase labels.
const TRL_PHASES = [
  ["materializing dataset", "Preparing dataset…"],
  ["dataset size=", "Preparing dataset…"],
  ["split:", "Splitting train / eval…"],
  ["loading base model", "Loading base model…"],
  ["est_max_steps=", "Planning training run…"],
  ["starting trainer.train", "Training…"],
  ["training complete", "Saving checkpoint…"],
  ["checkpoint at", "Checkpoint written"],
];

function _setPhase(text) {
  const el = $("#train-phase");
  if (!el) return;
  el.hidden = false;
  el.textContent = text;
}

function _narratePhase(line) {
  // Surface a friendly phase from a raw trl_cpu log line, if it matches.
  if (typeof line !== "string" || !line.includes("[trl_cpu]")) return;
  for (const [needle, label] of TRL_PHASES) {
    if (line.includes(needle)) { _setPhase(label); return; }
  }
}

function _resetProgress() {
  const fill = $("#progress-fill");
  if (fill) { fill.style.width = "0%"; fill.classList.remove("done"); }
  const label = $("#progress-label");
  if (label) label.textContent = "step 0 / ? · 0% · ETA --:--";
  const wrap = $("#train-progress");
  if (wrap) wrap.hidden = false;
  _setPhase("Waiting to start…");
}

function _updateProgress(ev) {
  // Called per StepEvent. Fills the bar, computes an ETA from the
  // observed per-step cadence, and narrates the live step count.
  if (ev.total_steps) state.totalSteps = ev.total_steps;
  const wrap = $("#train-progress");
  if (wrap) wrap.hidden = false;
  if (state.firstStepTs == null) {
    state.firstStepTs = Date.now();
    state.firstStep = ev.step;
  }
  const total = state.totalSteps;
  const pct = total ? Math.min(100, Math.round((ev.step / total) * 100)) : 0;
  const fill = $("#progress-fill");
  if (fill && total) fill.style.width = pct + "%";
  let eta = "--:--";
  const stepsDone = ev.step - state.firstStep;
  if (total && stepsDone > 0) {
    const perStepMs = (Date.now() - state.firstStepTs) / stepsDone;
    const remainS = Math.max(0, Math.round((perStepMs * (total - ev.step)) / 1000));
    eta = _formatHMS(remainS).slice(3);  // HH:MM:SS → MM:SS
  }
  const label = $("#progress-label");
  if (label) {
    label.textContent = total
      ? `step ${ev.step} / ${total} · ${pct}% · ETA ${eta}`
      : `step ${ev.step} · ETA --:--`;
  }
  _setPhase(total
    ? `Training — step ${ev.step} of ${total}`
    : `Training — step ${ev.step}`);
}

function _renderResultBanner(status, message) {
  // Terminal outcome — assembled from the buffered step data so the
  // user reads the result without parsing the raw log.
  const el = $("#train-result");
  if (!el) return;
  const buf = state.stepBuffer || [];
  const first = buf[0];
  const last = buf[buf.length - 1];
  let elapsed = "—";
  if (state.run && state.run.created_at) {
    const ms = Date.now() - Date.parse(state.run.created_at);
    if (!Number.isNaN(ms)) elapsed = _formatHMS(Math.floor(ms / 1000));
  }
  const parts = [];
  if (status === "succeeded") {
    el.className = "train-result";
    parts.push(`✓ ${buf.length} step${buf.length === 1 ? "" : "s"}`, elapsed);
    if (first && last) {
      parts.push(`loss ${first.loss.toFixed(2)}→${last.loss.toFixed(2)}`);
    }
    if (last && last.mean_token_accuracy != null) {
      parts.push(`acc ${last.mean_token_accuracy.toFixed(2)}`);
    }
    if (/checkpoint/i.test(message || "")) parts.push("checkpoint written");
  } else {
    el.className = "train-result failed";
    parts.push(`✗ ${status}`);
    if (message) parts.push(message);
  }
  el.textContent = parts.filter(Boolean).join("  ·  ");
  el.hidden = false;
}

async function discoverActiveRun() {
  // Hands-free CPU lane: when the operator autostarts a run
  // (MINDXTRAIN_AUTOSTART) the UI must attach to it on page load with no
  // button push. Picks the most-recent pending/running run, opens the
  // Train card, and subscribes — the SSE ring buffer replays the steps
  // and metrics already emitted so nothing is missed.
  try {
    const runs = await getJSON("/coach/api/runs");
    const active = (runs || [])
      .filter((r) => r.status === "running" || r.status === "pending")
      .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0];
    if (!active) return;
    state.run = active;
    if (!state.recipe) state.recipe = active.recipe;
    // Pull the recipe detail so the headline shows the cpu_throttle %.
    try {
      state.recipeDetail = await getJSON(
        `/coach/api/recipes/${encodeURIComponent(active.recipe)}`,
      );
    } catch (_e) { /* throttle headline falls back to — */ }
    progressTo("step-train");
    $("#train-id").textContent = `run ${active.id} (autostarted · ${active.recipe})`;
    $("#train-charts").hidden = false;
    $("#train-log-wrap").hidden = false;
    $("#cancel-train").hidden = false;
    $("#run-train").disabled = true;
    setStatusBadge(active.status);
    appendLog({ line: `attached to autostarted run ${active.id}` });
    subscribeRun(active.id);
  } catch (_e) {
    /* no operator runs yet — the normal case on a fresh manual boot */
  }
}

async function refreshSEADecision() {
  // Surface the mindX SEA agent's go/no-go on autonomous training. The
  // operator only auto-launches a run when this gate is open; the user
  // can always start one by hand with the Run training button.
  const wrap = $("#sea-status");
  const pill = $("#sea-status-pill");
  const text = $("#sea-status-text");
  if (!wrap || !pill || !text) return;
  try {
    const d = await getJSON("/coach/api/sea-decision");
    wrap.hidden = false;
    if (!d.autostart_enabled) {
      pill.className = "badge-status tier-unknown";
      pill.textContent = "SEA";
      text.textContent =
        "Autonomous mode off — start a session with Run training below.";
      return;
    }
    if (d.open) {
      pill.className = "badge-status tier-correlated";
      pill.textContent = "SEA · go";
    } else {
      pill.className = "badge-status tier-drifted";
      pill.textContent = "SEA · hold";
    }
    text.textContent = d.reason || "";
  } catch (_e) {
    wrap.hidden = false;
    pill.className = "badge-status tier-unknown";
    pill.textContent = "SEA";
    text.textContent = "SEA decision unavailable.";
  }
}

async function cancelTrain() {
  if (!state.run) return;
  $("#cancel-train").disabled = true;
  try {
    await fetch(`/coach/api/runs/${state.run.id}/cancel`, { method: "POST" });
  } finally {
    $("#cancel-train").disabled = false;
  }
}

function openModelfileBuilder() {
  // Open the standalone Modelfile builder in a separate window, pre-filled for
  // the current run when available (adapter = its checkpoint, tag = recipe).
  const params = new URLSearchParams();
  const run = state.run;
  if (run) {
    if (run.out_dir) params.set("adapter", `${run.out_dir}/checkpoint`);
    const tag = ($("#ollama-push-tag") && $("#ollama-push-tag").value.trim()) || run.recipe || "";
    if (tag) params.set("tag", tag);
  }
  if (state.chatBackendModel) params.set("from", state.chatBackendModel);
  const qs = params.toString();
  window.open(`/coach/modelfile${qs ? "?" + qs : ""}`, "mindxtrain-modelfile",
    "width=960,height=900,scrollbars=yes,resizable=yes");
}

async function pushTrainedRunToOllama() {
  if (!state.run) return;
  const btn = $("#push-to-ollama-btn");
  const status = $("#push-to-ollama-status");
  const tag = ($("#ollama-push-tag").value || "").trim();
  btn.disabled = true;
  status.textContent = "merging + creating…";
  status.className = "hint";
  // Push log lines stream into the train-log SSE channel — surface them
  // so the user isn't watching a frozen button for ~30-60s of merge time.
  const logWrap = $("#train-log-wrap");
  if (logWrap) {
    logWrap.hidden = false;
    logWrap.open = true;
  }
  try {
    // The log lines fire through the train SSE channel — the user is
    // already watching #train-log, so the merge progress shows up there.
    const registerFallback = ($("#register-fallback") || {}).checked === true;
    const body = {};
    if (tag) body.tag = tag;
    if (registerFallback) body.register_with_mindx = true;
    const r = await fetch(
      `/coach/api/runs/${encodeURIComponent(state.run.id)}/push-to-ollama`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      status.textContent = `failed (${r.status}): ${data.detail || "unknown error"}`;
      status.className = "hint bad";
      return;
    }
    let msg = `pushed: ${data.tag} (${data.merged_dir})`;
    if (data.mindx_fallback_swapped && data.mindx_fallback_swap) {
      const prev = data.mindx_fallback_swap.previous || "?";
      const cur = data.mindx_fallback_swap.current || "?";
      msg += ` · mindX fallback: ${prev} → ${cur}`;
    } else if (registerFallback) {
      msg += " · mindX swap failed (see log)";
    }
    status.textContent = msg;
    status.className = "hint ready";
    // Re-probe so the chat card flips to the freshly pushed model on its
    // next status read.
    probeChat();
  } catch (e) {
    status.textContent = `error: ${e}`;
    status.className = "hint bad";
  } finally {
    btn.disabled = false;
  }
}

// --- step 4: cost --------------------------------------------------------

async function runCost() {
  const gpus = parseInt($("#cost-gpus").value, 10);
  const hours = parseFloat($("#cost-hours").value);
  const res = await postJSON("/coach/api/cost", { gpus, hours, safety_margin: 1.15 });
  const tbody = $("#cost-table tbody");
  tbody.innerHTML = "";
  for (const row of [res.mi300x, res.h100, res.h200]) {
    const tr = document.createElement("tr");
    const fits = row.fits_qwen3_8b_bf16_bs8_seq4096;
    tr.innerHTML = `
      <td>${row.name}</td>
      <td>$${row.rate_usdc_per_hour.toFixed(2)}</td>
      <td>${row.gpus}</td>
      <td>${res.hours}</td>
      <td><strong>$${row.cost_usdc.toFixed(2)}</strong></td>
      <td class="fits-${fits ? "yes" : "no"}">${fits ? "✓ " : "✗ "}${row.note}</td>
    `;
    tbody.appendChild(tr);
  }
  $("#cost-table").hidden = false;
  $("#cost-headline").hidden = false;
  $("#cost-headline").innerHTML =
    `MI300X is <strong>${res.speedup_vs_h100_x.toFixed(2)}×</strong> cheaper than the H100 baseline for this workload.`;
}

// --- step 6: deploy (github push + droplet provision/sync) -------------

// Generic SSE attachment used by all three deploy cards. Returns the EventSource
// so the caller can keep a reference for cancellation.
function attachDeployStream(runId, opts) {
  const { logEl, badgeEl, cancelBtn, runBtn, onTerminal } = opts;
  const es = new EventSource(`/coach/api/runs/${runId}/events`);
  const setBadge = (label) => {
    badgeEl.textContent = label;
    badgeEl.className = "badge-status " + label.toLowerCase().replace(/\s+/g, "-");
    badgeEl.hidden = false;
  };
  setBadge("running");
  logEl.hidden = false;
  logEl.textContent = "";
  let lineCount = 0;
  const append = (line) => {
    logEl.appendChild(document.createTextNode(line + "\n"));
    lineCount += 1;
    if (lineCount > MAX_LOG_LINES) {
      while (lineCount > MAX_LOG_LINES && logEl.firstChild) {
        logEl.removeChild(logEl.firstChild);
        lineCount -= 1;
      }
    }
    if (lineCount % 50 === 0) {
      _foldLogElement(logEl);
    }
    logEl.scrollTop = logEl.scrollHeight;
  };
  es.addEventListener("log", (e) => append(JSON.parse(e.data).line));
  es.addEventListener("status", (e) => {
    const ev = JSON.parse(e.data);
    setBadge(ev.status);
    const terminal = ["succeeded", "failed", "cancelled"].includes(ev.status);
    if (terminal) {
      es.close();
      if (cancelBtn) cancelBtn.hidden = true;
      if (runBtn) runBtn.disabled = false;
      if (onTerminal) onTerminal(ev);
    }
  });
  es.addEventListener("error", () => setBadge("disconnected"));
  return es;
}

const deploy = {
  github: { es: null, runId: null },
  provision: { es: null, runId: null },
  sync: { es: null, runId: null },
};

function fmtMissing(missing) {
  if (!missing || !missing.length) return "";
  return `set ${missing.join(", ")} in .env`;
}

async function refreshGithubStatus() {
  try {
    const s = await getJSON("/coach/api/github/status");
    const target = $("#github-target");
    const status = $("#github-status");
    const button = $("#run-github");
    if (s.configured) {
      status.textContent = "ready";
      status.className = "hint deploy-status ready";
      target.textContent = `→ github.com/${s.target}`;
      button.disabled = false;
    } else {
      status.textContent = fmtMissing(s.missing) || "not configured";
      status.className = "hint deploy-status notready";
      target.textContent = "";
      button.disabled = true;
    }
  } catch (e) {
    $("#github-status").textContent = `status probe failed: ${e}`;
  }
}

async function refreshDropletStatus() {
  try {
    const s = await getJSON("/coach/api/droplet/status");
    // Provision card.
    const pStatus = $("#provision-status"), pTarget = $("#provision-target"), pBtn = $("#run-provision");
    if (s.provision.configured) {
      pStatus.textContent = "ready";
      pStatus.className = "hint deploy-status ready";
      pTarget.textContent = `→ ${s.provision.target}`;
      pBtn.disabled = false;
    } else {
      pStatus.textContent = fmtMissing(s.provision.missing) || "not configured";
      pStatus.className = "hint deploy-status notready";
      pTarget.textContent = "";
      pBtn.disabled = true;
    }
    // Sync card.
    const sStatus = $("#sync-status"), sTarget = $("#sync-target"), sBtn = $("#run-sync");
    if (s.sync.configured) {
      sStatus.textContent = "ready";
      sStatus.className = "hint deploy-status ready";
      sTarget.textContent = `→ ${s.sync.target}`;
      sBtn.disabled = false;
    } else {
      sStatus.textContent = fmtMissing(s.sync.missing) || "not configured";
      sStatus.className = "hint deploy-status notready";
      sTarget.textContent = "";
      sBtn.disabled = true;
    }
  } catch (e) {
    $("#provision-status").textContent = `status probe failed: ${e}`;
    $("#sync-status").textContent = `status probe failed: ${e}`;
  }
}

async function runDeploy({ url, body, slot, runBtn, cancelBtn, logEl, badgeEl, onSuccess, onStart }) {
  runBtn.disabled = true;
  cancelBtn.hidden = false;
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) {
      const text = await r.text();
      badgeEl.textContent = "failed";
      badgeEl.className = "badge-status failed";
      badgeEl.hidden = false;
      logEl.hidden = false;
      logEl.textContent = `${r.status}: ${text}`;
      runBtn.disabled = false;
      cancelBtn.hidden = true;
      return;
    }
    const run = await r.json();
    deploy[slot].runId = run.id;
    deploy[slot].es = attachDeployStream(run.id, {
      logEl, badgeEl, cancelBtn, runBtn,
      onTerminal: (ev) => {
        deploy[slot].es = null;
        if (ev.status === "succeeded" && typeof onSuccess === "function") {
          onSuccess();
        }
      },
    });
    // onStart fires after the run-id is known. Used by provision-with-recipe
    // to bind the Train card's SSE to the same run before training begins
    // streaming events, so the loss chart populates in real time.
    if (typeof onStart === "function") {
      try { onStart(); } catch (_) { /* non-fatal */ }
    }
  } catch (e) {
    badgeEl.textContent = "failed";
    badgeEl.className = "badge-status failed";
    badgeEl.hidden = false;
    logEl.hidden = false;
    logEl.textContent = String(e);
    runBtn.disabled = false;
    cancelBtn.hidden = true;
  }
}

async function cancelDeploy(slot, cancelBtn) {
  const runId = deploy[slot].runId;
  if (!runId) return;
  cancelBtn.disabled = true;
  try {
    await fetch(`/coach/api/runs/${runId}/cancel`, { method: "POST" });
  } finally {
    cancelBtn.disabled = false;
  }
}

function runGithubPush() {
  return runDeploy({
    url: "/coach/api/github/push",
    body: { force: $("#github-force").checked },
    slot: "github",
    runBtn: $("#run-github"),
    cancelBtn: $("#cancel-github"),
    logEl: $("#github-log"),
    badgeEl: $("#github-badge"),
    onSuccess: () => {
      // Focus the next deploy action — provision is the production path on
      // a fresh MI300X; sync is the alternative if the user already has one.
      const pBtn = $("#run-provision");
      if (pBtn && !pBtn.disabled) pBtn.focus();
    },
  });
}

function runDropletProvision() {
  // Pass the picked recipe through so cloud-init runs `mindxtrain train`
  // and the orchestrator bridges its log into this run's SSE stream. If
  // the user somehow reaches Provision without picking a recipe, we still
  // provision (bench-only) — the API treats recipe as optional.
  const body = state.recipe ? { recipe: state.recipe } : {};
  return runDeploy({
    url: "/coach/api/droplet/provision",
    body,
    slot: "provision",
    runBtn: $("#run-provision"),
    cancelBtn: $("#cancel-provision"),
    logEl: $("#provision-log"),
    badgeEl: $("#provision-badge"),
    onStart: () => {
      // The provision run-id is also where training events will land, so
      // open the Train card immediately and bind its SSE to this run. The
      // loss chart will populate as soon as the droplet starts training.
      const runId = deploy.provision.runId;
      if (!runId) return;
      markCardDone("step-deploy");
      progressTo("step-train");
      $("#train-id").textContent = `run ${runId} (remote MI300X)`;
      $("#train-charts").hidden = false;
      $("#train-log-wrap").hidden = false;
      subscribeRun(runId);
    },
  });
}

function runDropletSync() {
  const body = state.recipe ? { recipe: state.recipe } : {};
  return runDeploy({
    url: "/coach/api/droplet/sync",
    body,
    slot: "sync",
    runBtn: $("#run-sync"),
    cancelBtn: $("#cancel-sync"),
    logEl: $("#sync-log"),
    badgeEl: $("#sync-badge"),
    onSuccess: () => {
      markCardDone("step-deploy");
      progressTo("step-train");
    },
  });
}

// --- step 7: chat (gated on backend health) ------------------------------

async function probeChat() {
  // Backend badge + ollama status + the model list that drives the chat.
  try {
    const h = await getJSON("/coach/api/health");
    state.chatBackendModel = h.chat_backend_model || "";
    _updateBackendBadge(h);
  } catch (e) {
    _updateBackendBadge({ chat_backend_ready: false, chat_backend_name: "" });
  }
  await refreshOllamaStatus();
  await loadChatModels();
}

async function refreshOllamaStatus() {
  const el = $("#ollama-status");
  if (!el) return;
  try {
    const s = await getJSON("/coach/api/ollama/status");
    state.ollamaReachable = s.reachable;
    el.textContent = s.reachable
      ? `ollama: running (${(s.serve_pids || []).length || 1} proc)`
      : (s.has_ollama_bin ? "ollama: stopped" : "ollama: not installed");
    el.className = "hint " + (s.reachable ? "ready" : "notready");
    const startBtn = $("#ollama-start");
    const stopBtn = $("#ollama-stop");
    if (startBtn) startBtn.hidden = s.reachable;
    if (stopBtn) stopBtn.hidden = !s.reachable;
  } catch (e) {
    el.textContent = "ollama: ?";
  }
}

async function loadChatModels() {
  const sel = $("#chat-model");
  const status = $("#chat-status");
  if (!sel) return;
  let models = [];
  try { models = (await getJSON("/coach/api/models")).models || []; } catch (e) { /* offline */ }
  // Preserve the user's current choice across the 30 s re-probe.
  const previous = sel.value;
  sel.innerHTML = "";
  if (!models.length) {
    if ($("#chat-disabled-msg")) $("#chat-disabled-msg").hidden = false;
    if ($("#chat-send")) $("#chat-send").disabled = true;
    if (status) { status.textContent = "no model"; status.className = "hint notready"; }
    return;
  }
  for (const id of models) {
    const o = document.createElement("option");
    o.value = id; o.textContent = id;
    sel.appendChild(o);
  }
  // Keep the user's selection if still present; else the detected backend model;
  // else the first (local-first, sorted server-side).
  if (previous && models.includes(previous)) {
    sel.value = previous;
  } else if (state.chatBackendModel && models.includes(state.chatBackendModel)) {
    sel.value = state.chatBackendModel;
  }
  if ($("#chat-disabled-msg")) $("#chat-disabled-msg").hidden = true;
  if ($("#chat-send")) $("#chat-send").disabled = false;
  if (status) { status.textContent = `${models.length} model(s) ready`; status.className = "hint ready"; }
}

async function startOllama() {
  const el = $("#ollama-status");
  if (el) el.textContent = "ollama: starting…";
  try { await postJSON("/coach/api/ollama/start", {}); } catch (e) { /* report via status */ }
  setTimeout(probeChat, 1500);
}

async function stopOllama() {
  const el = $("#ollama-status");
  if (el) el.textContent = "ollama: stopping…";
  try { await postJSON("/coach/api/ollama/stop", {}); } catch (e) { /* report via status */ }
  setTimeout(probeChat, 800);
}

function _updateBackendBadge(h) {
  const badge = $("#backend-badge");
  if (!badge) return;
  if (h && h.chat_backend_ready) {
    const tail = h.chat_backend_model ? ` (${h.chat_backend_model})` : "";
    badge.textContent = `${h.chat_backend_name}${tail} live`;
    badge.className = "backend-badge ready";
  } else {
    const name = (h && h.chat_backend_name) || "no backend";
    badge.textContent = `${name} cold`;
    badge.className = "backend-badge notready";
  }
}

function _startChatBackendPolling() {
  // Re-probe every 30s so the chat card flips live if the user starts
  // ollama or vLLM after page-load. Also re-probe on tab focus — the
  // user often switches to a terminal, starts the daemon, then flips
  // back expecting the UI to know.
  setInterval(() => { probeChat(); }, 30000);
  window.addEventListener("focus", () => { probeChat(); });
}

let _chatHistory = [];
let _chatAbort = null;

function _appendChatMsg(role, text) {
  const t = $("#chat-transcript");
  t.hidden = false;
  const div = document.createElement("div");
  div.className = "chat-msg chat-" + role;
  div.innerHTML = `<span class="chat-role">${role}</span><span class="chat-text"></span>`;
  div.querySelector(".chat-text").textContent = text;
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
  return div.querySelector(".chat-text");
}

async function sendChat() {
  const inputEl = $("#chat-input");
  const input = inputEl.value.trim();
  if (!input) return;
  const model = $("#chat-model") ? $("#chat-model").value : "";
  if (!model) { $("#chat-status").textContent = "pick a model first"; return; }

  inputEl.value = "";
  _appendChatMsg("user", input);
  _chatHistory.push({ role: "user", content: input });
  const out = _appendChatMsg("assistant", "thinking…");
  $("#chat-send").disabled = true;
  $("#chat-stop").hidden = false;
  _chatAbort = new AbortController();
  let acc = "";
  try {
    const resp = await fetch("/coach/api/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model, messages: _chatHistory, max_tokens: 384 }),
      signal: _chatAbort.signal,
    });
    // Consume the SSE text stream (AI-SDK textStream pattern) token-by-token.
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let stop = false;
    while (!stop) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, nl);
        buf = buf.slice(nl + 2);
        let isErr = false;
        for (const line of frame.split("\n")) {
          if (line.startsWith("event: error")) isErr = true;
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") { stop = true; break; }
          try {
            const piece = JSON.parse(data);
            if (isErr) { out.textContent = "⚠ " + piece; }
            else { acc += piece; out.textContent = acc; }
          } catch (e) { /* skip unparseable frame */ }
        }
        $("#chat-transcript").scrollTop = $("#chat-transcript").scrollHeight;
      }
    }
    if (acc) _chatHistory.push({ role: "assistant", content: acc });
    else if (out.textContent === "…") out.textContent = "(no content — try a larger model)";
  } catch (e) {
    if (e.name !== "AbortError") out.textContent = `⚠ ${e}`;
  } finally {
    $("#chat-send").disabled = false;
    $("#chat-stop").hidden = true;
    _chatAbort = null;
  }
}

function stopChat() {
  if (_chatAbort) _chatAbort.abort();
}

// --- step 7: MEI card -----------------------------------------------------

const MEI_AXIS_LABELS = ["Q", "Dt", "Pp", "M", "E"];
const MEI_AXIS_FULL = [
  "Quality", "Decode throughput", "Prefill throughput", "Memory", "Energy",
];

function _ensureMEIRadar(axes) {
  if (state.meiChart || typeof Chart === "undefined") return state.meiChart;
  const ctx = $("#mei-radar").getContext("2d");
  state.meiChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: MEI_AXIS_FULL,
      datasets: [{
        label: "MEI sub-indices",
        data: axes,
        borderColor: "#f7921e",
        backgroundColor: "rgba(247, 146, 30, 0.18)",
        borderWidth: 2,
        pointBackgroundColor: "#f7921e",
      }],
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: { stepSize: 0.2, color: "#8b949e", backdropColor: "transparent" },
          grid: { color: "#30363d" },
          angleLines: { color: "#30363d" },
          pointLabels: { color: "#e6edf3", font: { size: 11 } },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
  return state.meiChart;
}

function _renderMEISubindices(view) {
  const ul = $("#mei-subindex-list");
  ul.innerHTML = "";
  const fields = [
    ["quality", view.quality],
    ["decode_throughput", view.decode_throughput],
    ["prefill_throughput", view.prefill_throughput],
    ["memory", view.memory],
    ["energy", view.energy],
  ];
  for (const [k, v] of fields) {
    const li = document.createElement("li");
    li.innerHTML = `<code>${k}</code>${v.toFixed(3)}`;
    ul.appendChild(li);
  }
}

function _renderMEIPromotion(view) {
  const promoteBtn = $("#mei-promote");
  const reasonsList = $("#mei-reasons");
  promoteBtn.disabled = !view.promotable;
  if (view.promotable) {
    reasonsList.hidden = true;
    reasonsList.innerHTML = "";
  } else {
    reasonsList.hidden = false;
    reasonsList.innerHTML = "";
    for (const r of view.promotion_reasons) {
      const li = document.createElement("li");
      li.textContent = r;
      reasonsList.appendChild(li);
    }
  }
}

function _renderMEINotes(view) {
  const ul = $("#mei-notes");
  ul.innerHTML = "";
  if (!view.notes || !view.notes.length) {
    ul.hidden = true;
    return;
  }
  ul.hidden = false;
  for (const n of view.notes) {
    const li = document.createElement("li");
    li.textContent = n;
    ul.appendChild(li);
  }
}

async function loadMEIForRun(runId) {
  const summary = $("#mei-summary");
  const body = $("#mei-card-body");
  if (!runId) {
    summary.textContent = "no active run";
    return;
  }
  summary.textContent = `loading score for run ${runId}…`;
  try {
    const view = await getJSON(`/coach/api/mei/score/${encodeURIComponent(runId)}`);
    state.mei = view;
    body.hidden = false;
    $("#mei-composite").textContent = view.composite.toFixed(3);
    $("#mei-provisional-flag").hidden = !view.mab_provisional;
    summary.textContent = view.promotable
      ? "promotable to AgenticPlace"
      : "below promotion gate — see reasons";
    const axes = [
      view.quality, view.decode_throughput, view.prefill_throughput,
      view.memory, view.energy,
    ];
    const chart = _ensureMEIRadar(axes);
    if (chart) {
      chart.data.datasets[0].data = axes;
      chart.update("none");
    }
    _renderMEISubindices(view);
    _renderMEINotes(view);
    _renderMEIPromotion(view);
    refreshMEIHistory();
  } catch (e) {
    summary.textContent = `no MEI score for ${runId} yet — score the run via \`mindxtrain mei score\``;
    body.hidden = true;
  }
}

// Short BLAKE3 view: first 8 + last 8 hex chars, or a dash when absent.
function _shortHash(h) {
  if (!h) return "—";
  return h.length > 20 ? `${h.slice(0, 8)}…${h.slice(-8)}` : h;
}

const RECEIPT_HASH_LABELS = {
  config_yaml: "config",
  checkpoint: "checkpoint",
  autotune_plan: "autotune plan",
  dataset: "dataset",
  eval_json: "eval",
};

async function loadReceiptForRun(runId) {
  const empty = $("#receipt-empty");
  const body = $("#receipt-card-body");
  const badge = $("#receipt-badge");
  const list = $("#receipt-hashes");
  if (!runId) {
    if (empty) empty.textContent = "no active run";
    return;
  }
  try {
    const view = await getJSON(`/coach/api/receipt/${encodeURIComponent(runId)}`);
    state.receipt = view;
    if (empty) empty.hidden = true;
    if (body) body.hidden = false;
    if (badge) {
      badge.hidden = false;
      badge.textContent = view.verified ? "verified ✓" : "unverified";
      badge.className = "badge-status " + (view.verified ? "succeeded" : "failed");
    }
    if (list) {
      list.innerHTML = "";
      for (const [field, value] of Object.entries(view.hashes)) {
        const ok = view.checks[field];
        const li = document.createElement("li");
        const label = RECEIPT_HASH_LABELS[field] || field;
        const mark = value ? (ok ? "✓" : "✗") : "·";
        li.innerHTML =
          `<code>${mark} ${label}</code>` +
          `<span class="mono">${_shortHash(value)}</span>`;
        list.appendChild(li);
      }
    }
  } catch (e) {
    // 409 = run finished but no manifest (rare best-effort emit failure);
    // 404 = unknown run. Either way, leave the card in its waiting state.
    if (empty) {
      empty.hidden = false;
      empty.textContent = "no receipt for this run yet";
    }
    if (body) body.hidden = true;
  }
}

// --- create script (dataset authoring) -----------------------------------

function _parseExchanges(text) {
  const out = [];
  for (const raw of (text || "").split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const i = line.indexOf(":::");
    if (i < 0) continue;
    const user = line.slice(0, i).trim();
    const assistant = line.slice(i + 3).trim();
    if (user && assistant) out.push({ user, assistant });
  }
  return out;
}

function _linesToList(text) {
  return (text || "").split("\n").map((s) => s.trim()).filter(Boolean);
}

async function seedFromPersona() {
  try {
    const p = await getJSON("/coach/api/persona");
    if (p.name) $("#ds-persona-name").value = p.name;
    if (p.system_prompt) $("#ds-system").value = p.system_prompt;
    if (Array.isArray(p.voice_examples)) $("#ds-voice").value = p.voice_examples.join("\n");
    $("#ds-status").textContent = `seeded from persona "${p.name}"`;
  } catch (e) {
    $("#ds-status").textContent = "no persona available (set MINDXTRAIN_PERSONA_PATH)";
  }
}

async function loadPersonasAndSkills() {
  const sel = $("#ds-persona");
  const skillsHost = $("#ds-skills");
  let body = { personas: [], skills: [] };
  try { body = await getJSON("/coach/api/personas"); } catch (e) { /* offline */ }
  if (sel) {
    sel.innerHTML = '<option value="">(custom — use fields below)</option>';
    for (const p of body.personas) {
      const o = document.createElement("option");
      o.value = p.name; o.textContent = p.label;
      sel.appendChild(o);
    }
  }
  if (skillsHost) {
    skillsHost.innerHTML = "";
    for (const s of body.skills) {
      const lab = document.createElement("label");
      lab.innerHTML = `<input type="checkbox" class="ds-skill" value="${s.name}"> ${s.label}`;
      lab.title = s.addendum;
      skillsHost.appendChild(lab);
    }
  }
}

function _selectedSkills() {
  return Array.from(document.querySelectorAll(".ds-skill:checked")).map((c) => c.value);
}

async function saveScript() {
  const status = $("#ds-status");
  const skills = _selectedSkills();
  const persona = ($("#ds-persona") && $("#ds-persona").value) || "";
  const body = {
    name: $("#ds-name").value.trim() || "script",
    persona,
    persona_name: $("#ds-persona-name").value.trim() || "actor",
    system_prompt: $("#ds-system").value.trim(),
    voice_examples: _linesToList($("#ds-voice").value),
    exchanges: _parseExchanges($("#ds-exchanges").value),
    skills,
    seed_voice: $("#ds-seed-voice").checked,
  };
  if (!persona && !skills.length && !body.exchanges.length &&
      !(body.seed_voice && body.voice_examples.length)) {
    status.textContent = "pick a persona/skill, add an exchange, or a voice example";
    return;
  }
  status.textContent = "saving…";
  try {
    const info = await postJSON("/coach/api/datasets", body);
    const tp = info.train_params || {};
    status.textContent = `✓ ${info.rows} rows → ${info.path}` +
      (tp.epochs ? ` · suggested: ${tp.epochs} epochs, grad_accum ${tp.grad_accum}` : "");
    refreshDatasets();
  } catch (e) {
    status.textContent = `save failed: ${e}`;
  }
}

async function refreshDatasets() {
  const list = $("#ds-list");
  if (!list) return;
  try {
    const rows = await getJSON("/coach/api/datasets");
    list.innerHTML = "";
    if (!rows.length) { list.hidden = true; return; }
    for (const s of rows) {
      const li = document.createElement("li");
      li.innerHTML = `<code>${s.name}</code><span class="mono">${s.rows} rows · ${s.path}</span>`;
      list.appendChild(li);
    }
    list.hidden = false;
  } catch (e) {
    list.hidden = true;
  }
}

function wireCreateDataset() {
  const save = $("#ds-save");
  const seed = $("#ds-seed-persona");
  if (save) save.addEventListener("click", saveScript);
  if (seed) seed.addEventListener("click", seedFromPersona);
  loadPersonasAndSkills();
  refreshDatasets();
}

// --- governance: boardroom (any-N) + dojo (prime-N) ----------------------

let _lastBoardSize = 0;

async function loadBoardroomPresets() {
  const sel = $("#br-preset");
  if (!sel) return;
  try {
    const presets = await getJSON("/coach/api/boardroom/presets");
    sel.innerHTML = "";
    for (const name of Object.keys(presets)) {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = `${name} (${presets[name].length})`;
      o.dataset.roles = JSON.stringify(presets[name]);
      sel.appendChild(o);
    }
  } catch (e) { /* presets unavailable; leave empty */ }
}

async function loadBoardroomModels() {
  const sel = $("#br-model");
  if (!sel) return;
  let models = [];
  try {
    const body = await getJSON("/coach/api/models");
    models = body.models || [];
  } catch (e) { /* backend unreachable */ }
  sel.innerHTML = "";
  if (!models.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "(no backend models — start ollama)";
    sel.appendChild(o);
    return;
  }
  // Prefer a small local (non-cloud) model first.
  models.sort((a, b) => (a.includes(":cloud") ? 1 : 0) - (b.includes(":cloud") ? 1 : 0));
  for (const id of models) {
    const o = document.createElement("option");
    o.value = id; o.textContent = id;
    sel.appendChild(o);
  }
}

function _boardMembers() {
  const sel = $("#br-preset");
  const opt = sel && sel.selectedOptions[0];
  const roles = opt ? JSON.parse(opt.dataset.roles || "[]") : ["advocate", "critic", "analyst"];
  const model = $("#br-model").value.trim() || "llama3.2";
  return roles.map((r, i) => ({ id: `${r}-${i}`, role: r, model }));
}

function _defaultMotion() {
  return ($("#br-motion").value.trim())
    || (state.run ? `promote actor ${state.run.id}` : "promote the actor");
}

async function conveneBoardroom() {
  const status = $("#br-status");
  const members = _boardMembers();
  _lastBoardSize = members.length;
  status.textContent = "convening — members deliberating…";
  $("#br-dojo").hidden = true;
  try {
    const body = await postJSON("/coach/api/boardroom/convene", {
      motion: _defaultMotion(), members,
      model: $("#br-model").value.trim() || "llama3.2", use_models: true,
    });
    const d = body.decision;
    const outcome = $("#br-outcome");
    outcome.hidden = false;
    outcome.textContent = `${d.outcome.toUpperCase()} — ${d.rationale}`;
    const list = $("#br-votes");
    list.hidden = false;
    list.innerHTML = "";
    for (const del of body.deliberations) {
      const li = document.createElement("li");
      li.innerHTML = `<code>${del.vote} · ${del.role}</code>` +
        `<span class="mono">${(del.rationale || del.error || "").slice(0, 140)}</span>`;
      list.appendChild(li);
    }
    status.textContent = `decided: ${d.outcome}`;
    if (d.disputed) {
      $("#br-dojo").hidden = false;
      $("#br-verdict").textContent = "disputed — settle in a prime dojo";
    }
  } catch (e) {
    status.textContent = `convene failed: ${e} (is a chat backend running?)`;
  }
}

async function settleDojo() {
  const v = $("#br-verdict");
  v.textContent = "dojo deliberating…";
  try {
    const verdict = await postJSON("/coach/api/dojo/settle", {
      motion: _defaultMotion(), size: _lastBoardSize || 3,
      model: $("#br-model").value.trim() || "llama3.2", use_models: true,
    });
    v.textContent = `dojo (${verdict.judges.length} judges) → ` +
      `${verdict.winner.toUpperCase()} (${verdict.approvals}-${verdict.rejections})`;
  } catch (e) {
    v.textContent = `settle failed: ${e}`;
  }
}

function wireBoardroom() {
  const convene = $("#br-convene");
  const settle = $("#br-settle");
  if (convene) convene.addEventListener("click", conveneBoardroom);
  if (settle) settle.addEventListener("click", settleDojo);
  loadBoardroomPresets();
  loadBoardroomModels();
}

async function refreshMEIHistory() {
  const wrap = $("#mei-history-wrap");
  const tbody = $("#mei-history-table tbody");
  try {
    const rows = await getJSON("/coach/api/mei/history?last=10");
    tbody.innerHTML = "";
    if (!rows.length) {
      wrap.hidden = true;
      return;
    }
    wrap.hidden = false;
    for (const r of rows) {
      const tr = document.createElement("tr");
      const mark = r.promoted
        ? '<span class="promoted-mark">★</span>'
        : "·";
      tr.innerHTML = `
        <td>${new Date(r.timestamp).toLocaleString()}</td>
        <td class="mono">${r.run_id}</td>
        <td>${r.model_id}</td>
        <td><strong>${r.composite.toFixed(3)}</strong></td>
        <td>${mark}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    wrap.hidden = true;
  }
}

async function promoteCurrentMEI() {
  if (!state.mei) return;
  const promoteBtn = $("#mei-promote");
  const badge = $("#mei-promote-badge");
  promoteBtn.disabled = true;
  badge.hidden = false;
  badge.textContent = "promoting…";
  badge.className = "badge-status running";
  try {
    const r = await fetch(
      `/coach/api/mei/promote/${encodeURIComponent(state.mei.run_id)}`,
      { method: "POST" },
    );
    if (!r.ok) {
      const text = await r.text();
      badge.textContent = `failed (${r.status})`;
      badge.className = "badge-status failed";
      console.error("promote failed:", text);
      promoteBtn.disabled = false;
      return;
    }
    const body = await r.json();
    if (body.promoted) {
      badge.textContent = "promoted";
      badge.className = "badge-status succeeded";
      promoteBtn.disabled = true;  // already promoted; refresh below
      refreshMEIHistory();
    } else {
      badge.textContent = "blocked";
      badge.className = "badge-status failed";
      const reasonsList = $("#mei-reasons");
      reasonsList.hidden = false;
      reasonsList.innerHTML = "";
      for (const reason of body.reasons || []) {
        const li = document.createElement("li");
        li.textContent = reason;
        reasonsList.appendChild(li);
      }
      promoteBtn.disabled = false;
    }
  } catch (e) {
    badge.textContent = String(e);
    badge.className = "badge-status failed";
    promoteBtn.disabled = false;
  }
}

// --- bootstrap -----------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  $("#run-preflight").addEventListener("click", runPreflight);
  $("#run-hardware").addEventListener("click", runHardware);

  // Admin card: poll only when expanded; pause when collapsed. This
  // keeps the operator process idle when nobody's looking at the panel.
  const adminCard = $("#step-admin");
  if (adminCard) {
    adminCard.addEventListener("toggle", () => {
      if (adminCard.open) _adminStartPolling();
      else _adminStopPolling();
    });
    $("#admin-toggle-poll").addEventListener("click", _adminTogglePoll);
    $("#admin-clear-firehose").addEventListener("click", _adminClearFirehose);
  }
  $("#run-bench").addEventListener("click", runBench);
  $("#run-compile").addEventListener("click", runCompile);
  $("#run-train").addEventListener("click", runTrain);
  $("#cancel-train").addEventListener("click", cancelTrain);
  const pushBtn = $("#push-to-ollama-btn");
  if (pushBtn) pushBtn.addEventListener("click", pushTrainedRunToOllama);
  const mfBtn = $("#open-modelfile-btn");
  if (mfBtn) mfBtn.addEventListener("click", openModelfileBuilder);
  // Cost card is hidden for now; keep the handler wired if present.
  const costBtn = $("#run-cost");
  if (costBtn) costBtn.addEventListener("click", runCost);
  $("#chat-send").addEventListener("click", sendChat);
  const chatRecheck = $("#chat-recheck");
  if (chatRecheck) {
    chatRecheck.addEventListener("click", () => {
      $("#chat-status").textContent = "re-probing…";
      probeChat();
    });
  }
  // Streaming chat + ollama controls.
  const chatStop = $("#chat-stop");
  if (chatStop) chatStop.addEventListener("click", stopChat);
  const chatInput = $("#chat-input");
  if (chatInput) chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendChat(); }
  });
  const ollamaStart = $("#ollama-start");
  if (ollamaStart) ollamaStart.addEventListener("click", startOllama);
  const ollamaStop = $("#ollama-stop");
  if (ollamaStop) ollamaStop.addEventListener("click", stopOllama);
  const ollamaModels = $("#ollama-refresh-models");
  if (ollamaModels) ollamaModels.addEventListener("click", loadChatModels);
  $("#mei-refresh").addEventListener("click", () => {
    loadMEIForRun(state.run && state.run.id);
  });
  $("#mei-promote").addEventListener("click", promoteCurrentMEI);

  // Create script (dataset authoring).
  wireCreateDataset();

  // Governance: boardroom + dojo.
  wireBoardroom();

  // Deploy section.
  $("#run-github").addEventListener("click", runGithubPush);
  $("#cancel-github").addEventListener("click", () => cancelDeploy("github", $("#cancel-github")));
  $("#run-provision").addEventListener("click", runDropletProvision);
  $("#cancel-provision").addEventListener("click", () => cancelDeploy("provision", $("#cancel-provision")));
  $("#run-sync").addEventListener("click", runDropletSync);
  $("#cancel-sync").addEventListener("click", () => cancelDeploy("sync", $("#cancel-sync")));

  // Start the auto-advance chain at the top. Each step's success handler
  // calls progressTo(...) for the next step. loadRecipes runs eagerly so
  // the recipe list is rendered even if preflight/corpus fail — the user
  // can still see what's available.
  syncPipelineHeader("step-preflight");
  loadRecipes();
  probeChat();
  _startChatBackendPolling();
  refreshGithubStatus();
  refreshDropletStatus();
  refreshMEIHistory();
  runHardware();
  runPreflight();
  refreshChronos();
  _startChronosPolling();
  // Hands-free: attach to any run the operator autostarted at boot so
  // the Train card is live without a button push.
  discoverActiveRun();
  // SEA gate — show the mindX agent's autonomous-training verdict and
  // re-poll every 30s so a fresh decision file flips the banner live.
  refreshSEADecision();
  setInterval(refreshSEADecision, 30000);
});


// --- session metrics (per-training-run system load) ---------------------
//
// Five d3 sparklines + a five-cell mono headline live inside #step-train.
// Data shape: array of MetricsEvent dicts (see runs.py:MetricsEvent), each
// carrying ts/cpu_pct/ram_pct/load_1m/proc_rss_mb/proc_cpu_seconds.

function _enableSessionMetrics() {
  const headline = $("#session-headline");
  const metrics = $("#session-metrics");
  if (headline) headline.hidden = false;
  if (metrics) metrics.hidden = false;
  _renderSessionThrottle();
  _startElapsedTimer();
}

function _setSessionStatus(status) {
  const pill = $("#session-status");
  if (!pill) return;
  pill.textContent = status;
  // Map run status → tier class so the pill picks up the same colour
  // tokens defined for the chronos card (correlated/degraded/drifted).
  const tier = {
    running: "tier-correlated",
    succeeded: "tier-correlated",
    pending: "tier-unknown",
    launching: "tier-unknown",
    failed: "tier-drifted",
    cancelled: "tier-degraded",
  }[status] || "tier-unknown";
  pill.className = `badge-status ${tier}`;
}

function _renderSessionThrottle() {
  // Surface the recipe's cpu_throttle % and, when the host core count is
  // known, the resolved "N of M cores" with a row of core pips — the
  // same floor((cores*pct)/100) math the trl_cpu backend applies.
  const el = $("#session-throttle");
  if (!el) return;
  const yaml = (state.recipeDetail || {}).yaml || "";
  const m = yaml.match(/cpu_throttle:[\s\S]{0,200}?percent:\s*(\d+)/);
  if (!m) { el.textContent = "—"; return; }
  const pct = parseInt(m[1], 10);
  const cores = (state.hardware && state.hardware.cpu && state.hardware.cpu.cores) || 0;
  el.innerHTML = "";
  const label = document.createElement("span");
  if (cores > 0) {
    const threads = Math.max(1, Math.floor((cores * pct) / 100));
    label.textContent = `${pct}% · ${threads} of ${cores} cores `;
    el.appendChild(label);
    for (let i = 0; i < cores; i += 1) {
      const pip = document.createElement("span");
      pip.className = "core-pip" + (i < threads ? " on" : "");
      el.appendChild(pip);
    }
  } else {
    label.textContent = `${pct}%`;
    el.appendChild(label);
  }
}

function _startElapsedTimer() {
  if (state.metricsTimer) return;
  // Wall + cpu-time counter ticks at 1 Hz even before the first sample.
  state.metricsTimer = setInterval(() => {
    if (!state.run) return;
    const startedMs = Date.parse(state.run.created_at || "");
    if (!Number.isNaN(startedMs)) {
      const elapsedS = Math.max(0, Math.floor((Date.now() - startedMs) / 1000));
      const wall = $("#session-wall");
      if (wall) wall.textContent = _formatHMS(elapsedS);
    }
    const last = state.metrics[state.metrics.length - 1];
    if (last) {
      const cpu = $("#session-cputime");
      if (cpu) cpu.textContent = _formatHMS(Math.floor(last.proc_cpu_seconds));
    }
  }, 1000);
}

function _stopElapsedTimer() {
  if (state.metricsTimer) {
    clearInterval(state.metricsTimer);
    state.metricsTimer = null;
  }
}

function _formatHMS(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(r)}`;
}

async function _backfillSessionMetrics(runId) {
  try {
    const body = await getJSON(
      `/coach/api/runs/${encodeURIComponent(runId)}/metrics`,
    );
    const samples = body.samples || [];
    if (samples.length) {
      state.metrics = samples.slice(-METRICS_BUFFER_CAP);
      _renderSessionSparklines();
      // Update last-loss + cpu-time headline with the freshest sample.
      _updateHeadlineFromSample(samples[samples.length - 1]);
    }
  } catch (_e) { /* graceful — first launch has no samples */ }
}

function _handleMetricsEvent(ev) {
  state.metrics.push(ev);
  if (state.metrics.length > METRICS_BUFFER_CAP) {
    state.metrics.splice(0, state.metrics.length - METRICS_BUFFER_CAP);
  }
  _updateHeadlineFromSample(ev);
  _renderSessionSparklines();
}

function _updateHeadlineFromSample(ev) {
  const cpu = $("#session-cputime");
  if (cpu) cpu.textContent = _formatHMS(Math.floor(ev.proc_cpu_seconds));
  // Last-loss comes from the existing step chart (Chart.js), not the
  // metrics stream — sync it here so the headline always reflects the
  // most recent loss value.
  const loss = state.chart && state.chart.data && state.chart.data.datasets[0];
  if (loss && loss.data && loss.data.length) {
    const v = loss.data[loss.data.length - 1];
    const lossEl = $("#session-last-loss");
    if (lossEl) lossEl.textContent = (typeof v === "number") ? v.toFixed(4) : "—";
  }
}

function _handleEvalEvent(ev) {
  // Every eval checkpoint streams to the raw log...
  const metrics = ev.metrics || {};
  appendLog({ line: `[eval@${ev.step}] ${JSON.stringify(metrics)}` });
  // ...and the headline surfaces the freshest eval signal so model
  // quality is visible without scrolling the log.
  const el = $("#session-eval");
  if (!el) return;
  let label = "loss";
  let val = metrics.eval_loss;
  if (typeof val !== "number") {
    const entry = Object.entries(metrics).find(([, v]) => typeof v === "number");
    if (entry) { [label, val] = entry; }
  }
  if (typeof val === "number") {
    el.textContent = `${val.toFixed(4)} ${label} @${ev.step}`;
  }
}

function _renderSessionSparklines() {
  if (typeof d3 === "undefined") return;
  const samples = state.metrics;
  _renderSparkline("#spark-cpu", "#spark-cpu-val", samples,
    s => s.cpu_pct, v => `${v.toFixed(1)}%`, "#f7921e");
  _renderSparkline("#spark-ram", "#spark-ram-val", samples,
    s => s.ram_pct, v => `${v.toFixed(1)}%`, "#2f81f7");
  _renderSparkline("#spark-load", "#spark-load-val", samples,
    s => s.load_1m, v => v.toFixed(2), "#2ea043");
  _renderSparkline("#spark-rss", "#spark-rss-val", samples,
    s => s.proc_rss_mb, v => `${v.toFixed(0)} MB`, "#f7921e");
  // cpu-s/s: derive a delta-per-second from consecutive samples.
  const rates = [];
  for (let i = 1; i < samples.length; i += 1) {
    const dt = samples[i].ts - samples[i - 1].ts;
    const dCpu = samples[i].proc_cpu_seconds - samples[i - 1].proc_cpu_seconds;
    rates.push({
      ts: samples[i].ts,
      value: dt > 0 ? dCpu / dt : 0,
    });
  }
  _renderSparkline("#spark-cpurate", "#spark-cpurate-val", rates,
    s => s.value, v => v.toFixed(2), "#d29922");
}

function _renderSparkline(svgSel, valSel, series, getY, formatVal, color) {
  const svg = d3.select(svgSel);
  if (svg.empty()) return;
  svg.selectAll("*").remove();
  const w = +svg.attr("width") || 240;
  const h = +svg.attr("height") || 40;
  const pad = 3;
  const valEl = document.querySelector(valSel);
  if (!series.length) {
    if (valEl) valEl.textContent = "—";
    return;
  }
  const ys = series.map(getY);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const yScale = d3.scaleLinear()
    .domain([yMin, Math.max(yMax, yMin + 0.0001)])
    .range([h - pad, pad]);
  const xScale = d3.scaleLinear()
    .domain([0, Math.max(1, series.length - 1)])
    .range([pad, w - pad]);
  const line = d3.line()
    .x((_, i) => xScale(i))
    .y((_, i) => yScale(ys[i]))
    .curve(d3.curveMonotoneX);
  svg.append("path")
    .datum(series)
    .attr("d", line)
    .attr("fill", "none")
    .attr("stroke", color)
    .attr("stroke-width", 1.5);
  if (valEl) valEl.textContent = formatVal(ys[ys.length - 1]);
}

// --- chronos card --------------------------------------------------------

async function refreshChronos() {
  const summary = document.getElementById("chronos-summary");
  let body;
  try {
    body = await getJSON("/coach/api/diagnostics/chronos");
  } catch (e) {
    if (summary) summary.textContent = `probe failed: ${e}`;
    _renderChronosHeadline({ consensus: "unavailable", utc: "", confidence_ms: 0 });
    return;
  }
  const pt = body.promised_time || {};
  _renderChronosHeadline(pt);
  if (summary) {
    const anchors = body.anchor_count != null ? `${body.anchor_count} anchors` : "no anchors";
    summary.textContent = `${pt.consensus || "?"} · ${anchors}`;
  }
  // Drift sparkline + density bars (d3) — degrades to no-op when d3 absent.
  if (typeof d3 !== "undefined") {
    _renderChronosDrift(body.drift_history || { buckets: [] });
    _renderChronosDensity(body.anchors || []);
  }
  // Measurement-confidence cross-check.
  try {
    const mc = await getJSON("/coach/api/diagnostics/measurement-confidence");
    _renderMeasurementConfidence(mc);
  } catch (e) {
    /* graceful: chip stays "unknown" */
  }
}

function _renderChronosHeadline(pt) {
  const utcEl = document.getElementById("chronos-utc");
  const consEl = document.getElementById("chronos-consensus");
  const confEl = document.getElementById("chronos-confidence");
  if (utcEl) utcEl.textContent = pt.utc || "—";
  if (consEl) {
    consEl.textContent = pt.consensus || "unknown";
    consEl.className = `badge-status tier-${pt.consensus || "unknown"}`;
  }
  if (confEl) {
    const ms = typeof pt.confidence_ms === "number" ? pt.confidence_ms : null;
    confEl.textContent = ms !== null ? `± ${ms.toFixed(1)} ms` : "± ? ms";
  }
}

function _renderChronosDrift(hist) {
  const svg = d3.select("#chronos-drift");
  svg.selectAll("*").remove();
  const buckets = hist.buckets || [];
  const w = +svg.attr("width") || 320;
  const h = +svg.attr("height") || 60;
  const pad = 4;
  if (buckets.length === 0) {
    svg.append("text").attr("x", w / 2).attr("y", h / 2)
      .attr("text-anchor", "middle").attr("fill", "#8b949e")
      .style("font", "11px ui-monospace, monospace")
      .text("no anchors in last 24h");
    return;
  }
  const xs = d3.scaleLinear()
    .domain([0, Math.max(1, buckets.length - 1)])
    .range([pad, w - pad]);
  const drifts = buckets.map(b => b.drift_mean_ms);
  const yMin = Math.min(0, ...drifts);
  const yMax = Math.max(0, ...drifts);
  const ys = d3.scaleLinear()
    .domain([yMin, yMax])
    .range([h - pad, pad]);
  // Zero line.
  svg.append("line")
    .attr("x1", pad).attr("x2", w - pad)
    .attr("y1", ys(0)).attr("y2", ys(0))
    .attr("stroke", "#30363d").attr("stroke-dasharray", "2 3");
  // Drift line.
  const line = d3.line()
    .x((_, i) => xs(i)).y(d => ys(d.drift_mean_ms))
    .curve(d3.curveMonotoneX);
  svg.append("path")
    .datum(buckets)
    .attr("d", line)
    .attr("fill", "none")
    .attr("stroke", "#f7921e")
    .attr("stroke-width", 1.5);
  // Hover dots.
  svg.selectAll("circle").data(buckets).enter()
    .append("circle")
    .attr("cx", (_, i) => xs(i))
    .attr("cy", d => ys(d.drift_mean_ms))
    .attr("r", 1.8)
    .attr("fill", "#f7921e");
}

function _renderChronosDensity(anchors) {
  const svg = d3.select("#chronos-density");
  svg.selectAll("*").remove();
  const w = +svg.attr("width") || 320;
  const h = +svg.attr("height") || 60;
  const pad = 4;
  // Bucket anchors by hour (epoch-hour). Last 24 buckets.
  const nowH = Math.floor(Date.now() / 3600000);
  const counts = new Array(24).fill(0);
  for (const a of anchors) {
    const tsMs = (a.captured_at_ns || 0) / 1e6;
    if (!tsMs) continue;
    const hr = Math.floor(tsMs / 3600000);
    const offset = nowH - hr;
    if (offset >= 0 && offset < 24) counts[23 - offset] += 1;
  }
  const maxC = Math.max(1, ...counts);
  const barW = (w - pad * 2) / 24;
  svg.selectAll("rect").data(counts).enter()
    .append("rect")
    .attr("x", (_, i) => pad + i * barW)
    .attr("y", c => h - pad - (h - 2 * pad) * (c / maxC))
    .attr("width", Math.max(1, barW - 1))
    .attr("height", c => (h - 2 * pad) * (c / maxC))
    .attr("fill", "#2f81f7");
}

function _renderMeasurementConfidence(mc) {
  const chip = document.getElementById("chronos-mc");
  const detail = document.getElementById("chronos-mc-detail");
  if (!chip) return;
  const band = mc.confidence_band || "unknown";
  chip.textContent = band;
  chip.className = `badge-status tier-${band}`;
  if (detail && mc.ok) {
    detail.textContent =
      `Δcpu=${mc.cpu_delta_pp}pp Δrss=${mc.rss_delta_mb}MB ` +
      `(psutil cpu=${mc.psutil_cpu_pct}% ps cpu=${mc.ps_cpu_pct}%)`;
  } else if (detail) {
    detail.textContent = "";
  }
}

function _startChronosPolling() {
  // 5s refresh — UI feels live without thrashing the mindX backend.
  setInterval(() => { refreshChronos(); }, 5000);
}
