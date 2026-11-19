"use strict";
(function () {
  const app = document.getElementById("app");
  let sid = null;
  const given = {};            // step id -> last answer, so Back can prefill

  // ---- tiny safe DOM helper: strings become text nodes, never HTML --------------------------------------------------
  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v === true) n.setAttribute(k, "");
      else if (v !== false && v != null) n.setAttribute(k, String(v));
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      n.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    }
    return n;
  }

  async function api(path, body) {
    const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    let res;
    try { res = await fetch(path, opt); } catch (e) { throw new Error("Can't reach the local server. Is it still running?"); }
    let data = {};
    try { data = await res.json(); } catch (e) { /* not JSON */ }
    if (!res.ok) throw new Error(data.error || "Request failed (" + res.status + ")");
    return data;
  }

  function render(...nodes) {
    app.replaceChildren(...nodes);
    const h = app.querySelector("h1, h2");
    if (h) { h.setAttribute("tabindex", "-1"); h.focus({ preventScroll: false }); }
  }
  function busy(msg) { render(el("p", { class: "hint", role: "status", text: msg || "One moment…" })); }
  function fail(msg, again) {
    render(el("h2", { text: "Something went wrong" }), el("p", { class: "error", role: "alert", text: msg }),
           el("button", { class: "primary", onclick: again, text: "Start over" }));
  }

  // ---- screens -----------------------------------------------------------------------------------------------------
  function disclaimer() {
    for (const k of Object.keys(given)) delete given[k];
    render(
      el("h1", { text: "Health Coach" }),
      el("p", { text: "This tool gives general health information and helps you decide how quickly to get care. It does not diagnose conditions, prescribe treatment, or replace a doctor." }),
      el("p", { class: "strong", text: "If you think you may be having an emergency, call 911 (or your local emergency number) now." }),
      el("p", { text: "Your answers stay on this computer. They are kept in memory only, never saved to disk or sent anywhere, and are discarded when you start over or after 30 minutes of inactivity." }),
      el("button", { class: "primary", onclick: start, text: "I understand, continue" }));
  }

  async function start() {
    busy("Starting…");
    try { const r = await api("/api/start", {}); sid = r.sid; step(r.step); } catch (e) { fail(e.message, disclaimer); }
  }

  async function submit(s, value) {
    given[s.id] = value;
    busy();
    try {
      const r = await api("/api/answer", { sid: sid, value: value });
      if (r.done) return result();
      step(r.step);
    } catch (e) { step(s, e.message); }
  }

  async function back() {
    busy();
    try { const r = await api("/api/back", { sid: sid }); step(r.step); } catch (e) { fail(e.message, disclaimer); }
  }

  async function restart() {
    if (sid) { try { await api("/api/end", { sid: sid }); } catch (e) { /* ignore */ } }
    sid = null; disclaimer();
  }

  // ---- input widgets: each returns { node, value() } --------------------------------------------------------------
  function multiSelect(s, prev) {
    const chosen = new Set(prev || []);
    const boxes = s.options.map((o, i) => {
      const cb = el("input", { type: "checkbox", id: "o" + i, value: o.value });
      cb.checked = chosen.has(o.value);
      return [o, cb];
    });
    boxes.forEach(([o, cb]) => cb.addEventListener("change", () => {
      if (!cb.checked) return;
      boxes.forEach(([o2, cb2]) => { if (cb2 !== cb && (o.value === "none" || o2.value === "none")) cb2.checked = false; });
    }));
    return {
      node: el("div", {}, boxes.map(([o, cb], i) => el("label", { class: "opt", for: "o" + i }, cb, el("span", { text: o.label })))),
      value: () => boxes.filter(([o, cb]) => cb.checked).map(([o]) => o.value),
    };
  }

  function symptomPicker(s, prev) {
    const selected = new Set(prev || []);
    let none = selected.has("none");
    selected.delete("none");
    const search = el("input", { type: "search", placeholder: "Search, for example: headache, cough, rash", "aria-label": "Search symptoms" });
    const list = el("div", { class: "symlist" });
    const note = el("p", { class: "hint", role: "status" });
    const noneBox = el("input", { type: "checkbox", id: "none-box" });
    noneBox.checked = none;
    noneBox.addEventListener("change", () => { none = noneBox.checked; if (none) { selected.clear(); draw(); } count(); });
    function count() { note.textContent = selected.size + " of 5 selected"; }
    function draw() {
      const q = search.value.trim().toLowerCase();
      list.replaceChildren();
      let last = null, shown = 0;
      s.options.forEach((o, i) => {
        if (o.value === "none") return;
        if (q && !((o.search || o.label.toLowerCase()).includes(q) || (o.category || "").toLowerCase().includes(q))) return;
        if (o.category !== last) { list.appendChild(el("h3", { text: o.category || "" })); last = o.category; }
        const cb = el("input", { type: "checkbox", id: "sym" + i });
        cb.checked = selected.has(o.value);
        cb.addEventListener("change", () => {
          if (cb.checked) {
            if (selected.size >= 5) { cb.checked = false; note.textContent = "You can choose up to 5."; return; }
            selected.add(o.value); none = false; noneBox.checked = false;
          } else selected.delete(o.value);
          count();
        });
        list.appendChild(el("label", { class: "opt", for: "sym" + i }, cb, el("span", { text: o.label })));
        shown++;
      });
      if (!shown) list.appendChild(el("p", { class: "hint", text: "No match. Try other words, or choose “None of these match” below." }));
    }
    search.addEventListener("input", draw);
    draw(); count();
    return {
      node: el("div", {}, search, note, list, el("label", { class: "opt", for: "none-box" }, noneBox, el("span", { text: "None of these match" }))),
      value: () => (none ? ["none"] : Array.from(selected)),
    };
  }

  function singleSelect(s, prev) {
    const name = "r-" + s.id;
    const radios = s.options.map((o, i) => {
      const r = el("input", { type: "radio", name: name, id: "r" + i, value: o.value });
      r.checked = prev === o.value;
      return [o, r];
    });
    return {
      node: el("div", {}, radios.map(([o, r], i) => el("label", { class: "opt", for: "r" + i }, r, el("span", { text: o.label })))),
      value: () => { const c = radios.find(([o, r]) => r.checked); if (!c) throw new Error("Please choose one."); return c[0].value; },
    };
  }

  function numberInput(s, prev) {
    const input = el("input", { type: "number", step: "any", inputmode: "decimal", "aria-label": s.text });
    if (prev != null) input.value = typeof prev === "object" ? prev.value : prev;
    const unit = s.units ? el("select", { "aria-label": "Unit" }, s.units.map((u) => el("option", { value: u, text: "°" + u }))) : null;
    if (unit && prev && typeof prev === "object") unit.value = prev.unit;
    const hint = s.optional ? "Optional: leave blank to skip." : (s.min != null ? "Between " + s.min + " and " + s.max + "." : "");
    return {
      node: el("div", {}, input, unit, hint ? el("p", { class: "hint", text: hint }) : null),
      value: () => {
        const raw = input.value.trim();
        if (raw === "") { if (s.optional) return null; throw new Error("Please enter a number."); }
        const n = Number(raw);
        if (!Number.isFinite(n)) throw new Error("Please enter a number.");
        return unit ? { value: n, unit: unit.value } : n;
      },
    };
  }

  function textInput(s, prev) {
    const t = el("textarea", { rows: "4", maxlength: "2000", "aria-label": s.text });
    t.value = prev || "";
    return { node: el("div", {}, t, s.optional ? el("p", { class: "hint", text: "Optional." }) : null), value: () => t.value };
  }

  // ---- one question -------------------------------------------------------------------------------------------------
  function step(s, err) {
    const prev = given[s.id];
    const parts = [el("h2", { text: s.text })];
    if (s.why) parts.push(el("details", {}, el("summary", { text: "Why do we ask?" }), el("p", { text: s.why })));
    const errBox = el("p", { class: "error", role: "alert", text: err || "" });
    const nav = el("div", { class: "nav" });
    if (s.index > 1) nav.appendChild(el("button", { onclick: back, text: "Back" }));

    if (s.kind === "yes_no_unsure") {
      parts.push(el("div", { class: "inputs" }, [["yes", "Yes"], ["no", "No"], ["unsure", "Not sure"]].map(([v, l]) =>
        el("button", { class: prev === v ? "picked" : "", onclick: () => submit(s, v), text: l }))));
    } else if (s.kind === "scale") {
      parts.push(el("div", { class: "inputs" }, Array.from({ length: 11 }, (_, i) =>
        el("button", { class: prev === i ? "picked" : "", "aria-label": String(i), onclick: () => submit(s, i), text: String(i) }))));
    } else {
      const w = s.kind === "multi_select" ? (s.id === "symptoms" ? symptomPicker(s, prev) : multiSelect(s, prev))
              : s.kind === "single_select" ? singleSelect(s, prev)
              : s.kind === "number" ? numberInput(s, prev) : textInput(s, prev);
      parts.push(el("div", { class: "inputs" }, w.node));
      nav.appendChild(el("button", { class: "primary", text: "Next", onclick: () => {
        let v;
        try { v = w.value(); } catch (e) { errBox.textContent = e.message; return; }
        submit(s, v);
      } }));
    }
    parts.push(errBox, nav);
    render(...parts);
  }

  // ---- result -------------------------------------------------------------------------------------------------------
  function safeLink(src) {
    const label = src.title + " (" + src.source + ")";
    return /^https?:\/\//.test(src.url || "") ? el("a", { href: src.url, target: "_blank", rel: "noopener noreferrer", text: label }) : el("span", { text: label });
  }

  async function result() {
    busy("Preparing your result…");
    let r;
    try { r = await api("/api/result", { sid: sid }); } catch (e) { return fail(e.message, restart); }
    const parts = [el("div", { class: "banner " + r.tier, role: "status" }, el("h2", { text: r.tier_label }))];
    if (r.message) parts.push(el("p", { class: "big", text: r.message }));
    (r.extra_resources || []).forEach((t) => parts.push(el("p", { class: "big", text: t })));
    if (r.explanation) {
      const ex = r.explanation;
      if (ex.when_to_seek_care) parts.push(el("h3", { text: "When to get care" }), el("p", { text: ex.when_to_seek_care }));
      if ((ex.considerations || []).length) parts.push(el("h3", { text: "What the reference material says" }), el("ul", {}, ex.considerations.map((c) => el("li", { text: c }))));
    }
    if ((r.sources || []).length) {
      parts.push(el("h3", { text: "Sources" }), el("ul", {}, r.sources.map((s) => el("li", {}, safeLink(s)))));
      (r.attribution || []).forEach((a) => parts.push(el("p", { class: "hint", text: a })));
    }
    if ((r.uncovered || []).length) parts.push(el("p", { class: "hint", text: "No reference material was found for: " + r.uncovered.join(", ") + "." }));
    if (r.used_fallback) parts.push(el("p", { class: "hint", text: "A detailed explanation is not shown because it could not be verified against the reference material." }));
    if ((r.assumptions || []).length) parts.push(el("details", {}, el("summary", { text: "Assumptions made" }), el("ul", {}, r.assumptions.map((a) => el("li", { text: a })))));
    if ((r.reasons || []).length) parts.push(el("details", {}, el("summary", { text: "Why this result (technical)" }), el("ul", {}, r.reasons.map((a) => el("li", { text: a })))));
    parts.push(el("div", { class: "nav" }, el("button", { class: "primary", onclick: restart, text: "Start over" })));
    render(...parts);
  }

  // ---- boot ---------------------------------------------------------------------------------------------------------
  async function boot() {
    try {
      const h = await api("/api/health");
      const kb = h.knowledge_base;
      document.getElementById("status").textContent =
        "Rules: on · Reference material: " + (kb ? Object.entries(kb.sources).map(([k, v]) => k + " " + (v || "")).join(", ") + (kb.stale ? " (old)" : "") : "off") +
        " · Language model: " + (h.language_model ? "on" : "off") + " · Risk model: " + (h.risk_model ? "on" : "off");
    } catch (e) { /* status line is optional */ }
    disclaimer();
  }
  boot();
})();
