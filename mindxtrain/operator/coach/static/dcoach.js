// dcoach — drive the proof loop (imprint → classroom → boardroom → feedback) and
// render the read-only decentralized-network panel. Plain ES, no build step.
"use strict";

const $ = (id) => document.getElementById(id);

async function loadPersonas() {
  try {
    const r = await fetch("/coach/api/personas");
    if (!r.ok) return;
    const data = await r.json();
    const sel = $("dc-persona");
    sel.innerHTML = "";
    (data.personas || []).forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.label || p.name;
      if (p.name === "codephreak") opt.selected = true;
      sel.appendChild(opt);
    });
    const skills = $("dc-skills");
    skills.innerHTML = "";
    (data.skills || []).forEach((s) => {
      const lbl = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = s.name;
      cb.className = "dc-skill";
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(" " + (s.label || s.name)));
      skills.appendChild(lbl);
    });
  } catch (_e) { /* personas optional */ }
}

async function loadDecentralized() {
  try {
    const r = await fetch("/coach/api/decentralized");
    if (!r.ok) return;
    const data = await r.json();
    $("dc-thesis").textContent = data.thesis || "";
    const fit = $("dc-fit-body");
    fit.innerHTML = "";
    (data.fit || []).forEach((f) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td><strong>${esc(f.primitive)}</strong></td>` +
        `<td>${esc(f.mindxtrain)}</td><td>${esc(f.maps_to)}</td>`;
      fit.appendChild(tr);
    });
    const net = $("dc-net");
    net.innerHTML = "";
    (data.networks || []).forEach((n) => {
      const card = document.createElement("div");
      card.className = "dc-net-card";
      card.innerHTML =
        `<h4>${esc(n.name)}</h4>` +
        `<p>${esc(n.what)}</p>` +
        `<p class="meta"><strong>Hardware:</strong> ${esc(n.hardware)}</p>` +
        `<p class="meta"><strong>Token:</strong> ${esc(n.token)}</p>` +
        `<p class="fit">${esc(n.fit)}</p>`;
      net.appendChild(card);
    });
  } catch (_e) { /* panel optional */ }
}

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function addPhase(tag, msg) {
  const row = document.createElement("div");
  row.className = "dc-phase";
  row.innerHTML = `<span class="tag">${esc(tag)}</span><span>${esc(msg)}</span>`;
  $("dc-phases").appendChild(row);
  $("dc-phases").scrollTop = $("dc-phases").scrollHeight;
}

function fmt(x) { return (typeof x === "number") ? x.toFixed(4) : String(x); }

function renderResult(res) {
  const c = res.classroom || {};
  $("r-before").textContent = fmt(c.before_recall);
  $("r-after").textContent = fmt(c.recall);
  const delta = c.imprint_delta;
  const de = $("r-delta");
  de.textContent = (delta >= 0 ? "+" : "") + fmt(delta);
  de.className = "v " + (delta > 0 ? "pass" : "fail");
  const pm = $("r-persona");
  pm.textContent = c.persona_maintained ? "yes" : "no";
  pm.className = "v " + (c.persona_maintained ? "pass" : "fail");
  $("r-board").textContent = res.boardroom_outcome || "–";
  const passed = !!res.passed;
  const pe = $("r-passed");
  pe.textContent = passed ? "PASS" : "FAIL";
  pe.className = "v " + (passed ? "pass" : "fail");
  $("r-rationale").textContent = res.boardroom_rationale || "";
  const np = res.next_params || {};
  $("r-next").textContent =
    `Autotune feedback → next run: epochs=${np.epochs}, grad_accum=${np.grad_accum}` +
    (passed ? " (held — the imprint took)." : " (trains harder — the imprint was weak).");
  $("dc-result").classList.remove("hidden");
}

async function run() {
  const btn = $("dc-run");
  btn.disabled = true;
  $("dc-status").textContent = "training on CPU — this takes a few minutes…";
  $("dc-phases").innerHTML = "";
  $("dc-live").classList.remove("hidden");
  $("dc-result").classList.add("hidden");

  const skills = Array.from(document.querySelectorAll(".dc-skill"))
    .filter((c) => c.checked).map((c) => c.value);
  const body = {
    persona: $("dc-persona").value,
    skills,
    base_model: $("dc-base").value.trim() || "HuggingFaceTB/SmolLM2-135M",
    board_preset: $("dc-board").value,
    board_model: $("dc-board-model").value.trim() || null,
    max_new_tokens: parseInt($("dc-tokens").value, 10) || 48,
  };

  try {
    const resp = await fetch("/coach/api/dcoach/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok || !resp.body) {
      addPhase("error", "could not start: HTTP " + resp.status);
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
        let evt;
        try { evt = JSON.parse(payload); } catch (_e) { continue; }
        if (evt.phase === "result") {
          renderResult(evt.result);
          $("dc-status").textContent = "done.";
        } else if (evt.phase === "error") {
          addPhase("error", evt.msg);
          $("dc-status").textContent = "failed.";
        } else if (evt.phase === "start") {
          addPhase("start", "run " + evt.run_id);
        } else {
          addPhase(evt.phase, evt.msg);
        }
      }
    }
  } catch (e) {
    addPhase("error", String(e));
    $("dc-status").textContent = "failed.";
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadPersonas();
  loadDecentralized();
  $("dc-run").addEventListener("click", run);
});
