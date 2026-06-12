"use strict";
// Standalone Ollama Modelfile builder. Fetches the PARAMETER catalogue, renders a
// toggle + input per parameter, and POSTs the assembled spec to /coach/api/modelfile/build.

const $ = (s) => document.querySelector(s);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

// Toggleable text/area instructions: SYSTEM, TEMPLATE, ADAPTER, LICENSE, REQUIRES.
const INSTRUCTIONS = [
  { key: "system", label: "SYSTEM", area: true, ph: "You are Codephreak, augmentic intelligence orchestrator." },
  { key: "template", label: "TEMPLATE", area: true, ph: "{{ .System }}\n{{ .Prompt }}" },
  { key: "adapter", label: "ADAPTER", area: false, ph: "./out/runs/<run>/checkpoint" },
  { key: "license", label: "LICENSE", area: true, ph: "Apache-2.0" },
  { key: "requires", label: "REQUIRES", area: false, ph: "0.5.0" },
];

function renderInstructions(prefill) {
  const host = $("#mf-instructions");
  for (const ins of INSTRUCTIONS) {
    const wrap = document.createElement("div");
    wrap.className = "mf-row";
    const pre = (prefill[ins.key] || "");
    const field = ins.area
      ? `<textarea id="mf-${ins.key}" rows="2" placeholder="${ins.ph}">${pre}</textarea>`
      : `<input id="mf-${ins.key}" type="text" placeholder="${ins.ph}" value="${pre}">`;
    wrap.innerHTML =
      `<label><input type="checkbox" class="mf-ins-toggle" data-key="${ins.key}" ${pre ? "checked" : ""}> ${ins.label}</label>` +
      `<div class="mf-toggle-field" id="mf-field-${ins.key}" ${pre ? "" : "hidden"}>${field}</div>`;
    host.appendChild(wrap);
  }
  host.querySelectorAll(".mf-ins-toggle").forEach((cb) => {
    cb.addEventListener("change", () => {
      $(`#mf-field-${cb.dataset.key}`).hidden = !cb.checked;
    });
  });
}

async function renderParams() {
  const host = $("#mf-params");
  let params = [];
  try { params = (await getJSON("/coach/api/modelfile/params")).parameters; }
  catch (e) { host.textContent = "could not load parameters"; return; }
  for (const p of params) {
    const div = document.createElement("div");
    div.className = "mf-param";
    const inputType = (p.type === "int" || p.type === "float") ? "number" : "text";
    const step = p.type === "float" ? "0.05" : "1";
    div.innerHTML =
      `<input type="checkbox" class="mf-param-toggle" data-name="${p.name}" data-type="${p.type}">` +
      `<code>${p.name}</code>` +
      `<input class="mf-param-val" data-name="${p.name}" type="${inputType}" step="${step}" ` +
      `value="${p.default ?? ""}" title="${p.description}">` +
      `<span class="mf-desc">${p.description}</span>`;
    host.appendChild(div);
  }
}

function buildSpec() {
  const spec = { from_model: $("#mf-from").value.trim(), parameters: {} };
  // Toggled instructions.
  for (const ins of INSTRUCTIONS) {
    const cb = document.querySelector(`.mf-ins-toggle[data-key="${ins.key}"]`);
    if (cb && cb.checked) {
      const v = $(`#mf-${ins.key}`).value.trim();
      if (v) spec[ins.key] = v;
    }
  }
  // Toggled parameters.
  document.querySelectorAll(".mf-param-toggle").forEach((cb) => {
    if (!cb.checked) return;
    const name = cb.dataset.name;
    const raw = document.querySelector(`.mf-param-val[data-name="${name}"]`).value.trim();
    if (raw === "") return;
    spec.parameters[name] = (cb.dataset.type === "int") ? parseInt(raw, 10)
      : (cb.dataset.type === "float") ? parseFloat(raw) : raw;
  });
  // Stop sequences.
  const stop = $("#mf-stop").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (stop.length) spec.stop = stop;
  // Messages: "role: content".
  const messages = [];
  for (const line of $("#mf-messages").value.split("\n")) {
    const i = line.indexOf(":");
    if (i < 0) continue;
    const role = line.slice(0, i).trim().toLowerCase();
    const content = line.slice(i + 1).trim();
    if (["system", "user", "assistant"].includes(role) && content) messages.push({ role, content });
  }
  if (messages.length) spec.messages = messages;
  return spec;
}

async function buildModelfile() {
  const status = $("#mf-status");
  const spec = buildSpec();
  if (!spec.from_model) { status.textContent = "FROM is required"; return; }
  status.textContent = "building…";
  try {
    const body = await postJSON("/coach/api/modelfile/build", spec);
    $("#mf-output").hidden = false;
    $("#mf-output").textContent = body.modelfile;
    status.textContent = "built ✓";
  } catch (e) { status.textContent = `build failed: ${e}`; }
}

async function createModel() {
  const s = $("#mf-create-status");
  const tag = $("#mf-tag").value.trim();
  if (!tag) { s.textContent = "tag required"; return; }
  const spec = buildSpec();
  if (!spec.from_model) { s.textContent = "FROM is required"; return; }
  s.textContent = `running ollama create ${tag}…`;
  try {
    const res = await postJSON("/coach/api/modelfile/create", { tag, spec });
    s.textContent = `${res.status}: ${(res.output || "").slice(-200)}`;
  } catch (e) { s.textContent = `create failed: ${e}`; }
}

function prefillFromQuery() {
  const q = new URLSearchParams(location.search);
  const out = {};
  for (const k of ["system", "template", "adapter", "license", "requires"]) {
    if (q.get(k)) out[k] = q.get(k);
  }
  if (q.get("from")) $("#mf-from").value = q.get("from");
  if (q.get("tag")) $("#mf-tag").value = q.get("tag");
  return out;
}

window.addEventListener("DOMContentLoaded", async () => {
  const prefill = prefillFromQuery();
  renderInstructions(prefill);
  await renderParams();
  $("#mf-build").addEventListener("click", buildModelfile);
  $("#mf-create").addEventListener("click", createModel);
  $("#mf-copy").addEventListener("click", () => {
    const t = $("#mf-output").textContent;
    if (t) navigator.clipboard && navigator.clipboard.writeText(t);
  });
});
