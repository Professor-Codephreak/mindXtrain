// prompt-tools — cheap non-permanent test → evaluate → promote to a Modelfile.
"use strict";

const $ = (id) => document.getElementById(id);
let lastResponse = "";

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function loadModels() {
  try {
    const r = await fetch("/coach/api/models");
    if (!r.ok) return;
    const data = await r.json();
    const sel = $("pt-model");
    const models = data.models || data || [];
    if (!models.length) return;
    sel.innerHTML = "";
    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    });
  } catch (_e) { /* keep default */ }
}

function addShot(user = "", assistant = "") {
  const row = document.createElement("div");
  row.className = "pt-shot";
  row.innerHTML =
    `<input type="text" class="pt-shot-u" placeholder="user" />` +
    `<input type="text" class="pt-shot-a" placeholder="assistant" />` +
    `<button class="ghost pt-shot-x" type="button">✕</button>`;
  row.querySelector(".pt-shot-u").value = user;
  row.querySelector(".pt-shot-a").value = assistant;
  row.querySelector(".pt-shot-x").addEventListener("click", () => row.remove());
  $("pt-shots").appendChild(row);
}

function collectMessages() {
  const msgs = [];
  const system = $("pt-system").value.trim();
  if (system) msgs.push({ role: "system", content: system });
  document.querySelectorAll(".pt-shot").forEach((row) => {
    const u = row.querySelector(".pt-shot-u").value.trim();
    const a = row.querySelector(".pt-shot-a").value.trim();
    if (u) msgs.push({ role: "user", content: u });
    if (a) msgs.push({ role: "assistant", content: a });
  });
  return msgs;
}

async function runTest() {
  const btn = $("pt-run");
  btn.disabled = true;
  $("pt-status").textContent = "running…";
  $("pt-response").textContent = "";
  lastResponse = "";

  const messages = collectMessages();
  messages.push({ role: "user", content: $("pt-user").value.trim() });
  const body = { model: $("pt-model").value, messages, max_tokens: 384, temperature: 0.7 };

  try {
    const resp = await fetch("/coach/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      $("pt-response").textContent = "error: HTTP " + resp.status;
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let tok;
        try { tok = JSON.parse(payload); } catch (_e) { continue; }
        if (typeof tok === "string") {
          lastResponse += tok;
          $("pt-response").textContent = lastResponse;
        }
      }
    }
    $("pt-status").textContent = "done.";
  } catch (e) {
    $("pt-response").textContent = "error: " + String(e);
    $("pt-status").textContent = "failed.";
  } finally {
    btn.disabled = false;
  }
}

function metric(key, val) {
  const cls = val >= 0.6 ? "pass" : "fail";
  return `<div class="pt-metric"><div class="v ${cls}">${val.toFixed(3)}</div>` +
         `<div class="k">${esc(key)}</div></div>`;
}

async function evaluate() {
  if (!lastResponse.trim()) { $("pt-verdict").textContent = "Run a test first."; return; }
  const btn = $("pt-eval");
  btn.disabled = true;
  $("pt-verdict").textContent = "scoring…";
  const body = {
    query: $("pt-user").value.trim(),
    response: lastResponse,
    reference: $("pt-reference").value.trim(),
    use_judge: $("pt-judge").checked,
    model: $("pt-model").value,
    guidelines: $("pt-guidelines").value.trim(),
  };
  try {
    const r = await fetch("/coach/api/eval/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) { $("pt-verdict").textContent = "eval error: HTTP " + r.status; return; }
    const data = await r.json();
    const box = $("pt-scores");
    box.innerHTML = "";
    Object.entries(data.scores || {}).forEach(([k, s]) => {
      box.insertAdjacentHTML("beforeend", metric(k, s.score));
    });
    box.insertAdjacentHTML("beforeend", metric("overall", data.overall));
    box.classList.remove("hidden");
    $("pt-verdict").innerHTML = data.advantageous
      ? `<span class="pass">Advantageous (${data.overall}).</span> Worth making permanent.`
      : `<span class="fail">Not advantageous yet (${data.overall}).</span> Tune the prompt and retest.`;
  } catch (e) {
    $("pt-verdict").textContent = "eval error: " + String(e);
  } finally {
    btn.disabled = false;
  }
}

function buildSpec() {
  const messages = collectMessages().filter((m) => m.role !== "system");
  return {
    from_model: $("pt-model").value,
    system: $("pt-system").value.trim(),
    messages,
  };
}

async function previewModelfile() {
  try {
    const r = await fetch("/coach/api/modelfile/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildSpec()),
    });
    if (r.ok) { $("pt-modelfile").textContent = (await r.json()).modelfile || ""; }
  } catch (_e) { /* preview optional */ }
}

async function promote() {
  const tag = $("pt-tag").value.trim();
  if (!tag) { $("pt-promote-status").textContent = "a tag is required."; return; }
  const btn = $("pt-promote");
  btn.disabled = true;
  $("pt-promote-status").textContent = "ollama create…";
  try {
    const r = await fetch("/coach/api/modelfile/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag, spec: buildSpec() }),
    });
    const data = await r.json();
    if (!r.ok) {
      $("pt-promote-status").textContent = "error: " + (data.detail || r.status);
    } else {
      $("pt-promote-status").innerHTML = data.status === "ok"
        ? `<span class="pass">created ${esc(tag)}.</span>`
        : `<span class="fail">${esc(data.status || "error")}: ${esc(data.output || "")}</span>`;
    }
  } catch (e) {
    $("pt-promote-status").textContent = "error: " + String(e);
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadModels();
  addShot();
  $("pt-add-shot").addEventListener("click", () => addShot());
  $("pt-run").addEventListener("click", runTest);
  $("pt-eval").addEventListener("click", evaluate);
  $("pt-promote").addEventListener("click", () => { previewModelfile(); promote(); });
  ["pt-system", "pt-model"].forEach((id) =>
    $(id).addEventListener("change", previewModelfile));
});
