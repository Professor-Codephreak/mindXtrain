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
};

const MAX_LOG_LINES = 2000;

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
  "step-cost",
  "step-chat",
];

// Map step ids to one of the three header pipeline stages.
const STAGE_FOR_STEP = {
  "step-preflight":    "automind",
  "step-dream-corpus": "automind",
  "step-recipes":      "mind",
  "step-autotune":     "mind",
  "step-compile":      "mind",
  "step-deploy":       "mind",
  "step-train":        "mind",
  "step-cost":         "cust",
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
      // Auto-advance to corpus check.
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

// --- step 2: dream corpus ---------------------------------------------------

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

async function loadRecipes() {
  const list = await getJSON("/coach/api/recipes");
  $("#recipe-count").textContent = `${list.length} built-in`;
  const target = $("#recipe-list");
  target.innerHTML = "";
  for (const r of list) {
    const div = document.createElement("div");
    div.className = "recipe";
    div.dataset.name = r.name;
    div.innerHTML = `
      <h3>${r.name}</h3>
      <div class="meta">
        <span class="badge">${r.method}</span>
        <span class="badge">${r.gpus}× GPU</span>
        ${r.base_model}
      </div>
    `;
    div.addEventListener("click", () => selectRecipe(r.name));
    target.appendChild(div);
  }
}

async function selectRecipe(name) {
  state.recipe = name;
  state.compileResult = null;
  for (const node of $$(".recipe")) {
    node.classList.toggle("selected", node.dataset.name === name);
  }
  const detail = await getJSON(`/coach/api/recipes/${name}`);
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
    data: { labels: [], datasets: [{ label: "loss", data: [], borderWidth: 2, tension: 0.2 }] },
    options: {
      animation: false,
      responsive: true,
      scales: {
        x: { title: { display: true, text: "step" } },
        y: { title: { display: true, text: "loss" }, beginAtZero: false },
      },
      plugins: { legend: { display: false } },
    },
  });
  return state.chart;
}

function pushPoint(ev) {
  const tbody = $("#metrics-table tbody");
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${ev.step}</td><td>${ev.loss.toFixed(4)}</td><td>${ev.lr ?? "&mdash;"}</td><td>${ev.grad_norm ?? "&mdash;"}</td>`;
  tbody.appendChild(tr);
  // Cap table to last 50 rows.
  while (tbody.children.length > 50) tbody.removeChild(tbody.firstChild);
  const chart = ensureChart();
  if (chart) {
    chart.data.labels.push(ev.step);
    chart.data.datasets[0].data.push(ev.loss);
    chart.update("none");
  }
}

function appendLog(ev) {
  const pre = $("#train-log");
  pre.appendChild(document.createTextNode(ev.line + "\n"));
  state.logLines += 1;
  if (state.logLines > MAX_LOG_LINES) {
    // Drop the first N text nodes to keep DOM bounded.
    while (state.logLines > MAX_LOG_LINES && pre.firstChild) {
      pre.removeChild(pre.firstChild);
      state.logLines -= 1;
    }
  }
  pre.scrollTop = pre.scrollHeight;
}

function subscribeRun(runId) {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(`/coach/api/runs/${runId}/events`);
  state.eventSource = es;
  es.addEventListener("step", (e) => pushPoint(JSON.parse(e.data)));
  es.addEventListener("eval", (e) => {
    const ev = JSON.parse(e.data);
    appendLog({ line: `[eval@${ev.step}] ${JSON.stringify(ev.metrics)}` });
  });
  es.addEventListener("log", (e) => appendLog(JSON.parse(e.data)));
  es.addEventListener("status", (e) => {
    const ev = JSON.parse(e.data);
    setStatusBadge(ev.status);
    const terminal = ["succeeded", "failed", "cancelled"].includes(ev.status);
    if (terminal) {
      es.close();
      state.eventSource = null;
      $("#cancel-train").hidden = true;
      $("#run-train").disabled = false;
      if (ev.status === "succeeded") {
        markCardDone("step-train");
        // Train succeeded — let the user reach for Cost / Chat next.
        progressTo("step-cost");
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
  if (state.chart) {
    state.chart.data.labels = [];
    state.chart.data.datasets[0].data = [];
    state.chart.update("none");
  }
  setStatusBadge("launching");
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

async function cancelTrain() {
  if (!state.run) return;
  $("#cancel-train").disabled = true;
  try {
    await fetch(`/coach/api/runs/${state.run.id}/cancel`, { method: "POST" });
  } finally {
    $("#cancel-train").disabled = false;
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
  try {
    const h = await getJSON("/coach/api/health");
    const status = $("#chat-status");
    if (h.chat_backend_ready) {
      status.textContent = `${h.chat_backend_name} ready`;
      status.className = "hint ready";
      $("#chat-disabled-msg").hidden = true;
      $("#chat-form").hidden = false;
    } else {
      const name = h.chat_backend_name || "(no backend configured)";
      status.textContent = `${name} not ready`;
      status.className = "hint notready";
    }
  } catch (e) {
    $("#chat-status").textContent = "health probe failed";
  }
}

async function sendChat() {
  const input = $("#chat-input").value.trim();
  if (!input) return;
  $("#chat-send").disabled = true;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "mindxtrain-demo",
        messages: [
          { role: "system", content: "You are mindXtrain's demo agent." },
          { role: "user", content: input },
        ],
        max_tokens: 256,
      }),
    });
    const text = await r.text();
    $("#chat-response").textContent = text;
    $("#chat-response").hidden = false;
  } finally {
    $("#chat-send").disabled = false;
  }
}

// --- bootstrap -----------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  $("#run-preflight").addEventListener("click", runPreflight);
  $("#run-bench").addEventListener("click", runBench);
  $("#run-compile").addEventListener("click", runCompile);
  $("#run-train").addEventListener("click", runTrain);
  $("#cancel-train").addEventListener("click", cancelTrain);
  $("#run-cost").addEventListener("click", runCost);
  $("#chat-send").addEventListener("click", sendChat);

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
  refreshGithubStatus();
  refreshDropletStatus();
  runPreflight();
});
