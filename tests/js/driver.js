// Drives static/app.js against a REAL running server with a minimal fake DOM. Usage: node driver.js BASE_URL path/to/app.js
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const [BASE, JS] = process.argv.slice(2);

class Text { constructor(t) { this.textContent = t; } }
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this._text = ""; this.checked = false; this.value = ""; this.className = ""; }
  setAttribute(k, v) { this.attrs[k] = v; if (k === "value") this.value = v; }
  appendChild(c) { this.children.push(c); return c; }
  replaceChildren(...n) { this.children = n; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  focus() {}
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  querySelector(sel) { const tags = sel.split(",").map((s) => s.trim()); const f = (n) => { for (const c of n.children || []) { if (tags.includes(c.tag)) return c; const r = f(c); if (r) return r; } return null; }; return f(this); }
}
const app = new Node("main"), status = new Node("p");
globalThis.document = { createElement: (t) => new Node(t), createTextNode: (t) => new Text(t), getElementById: (id) => (id === "app" ? app : status) };
const realFetch = globalThis.fetch;
globalThis.fetch = (p, o) => realFetch(BASE + p, o);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, what, ms = 4000) {
  const t0 = Date.now();
  for (;;) {
    try { const v = fn(); if (v) return v; } catch (e) { /* retry */ }
    if (Date.now() - t0 > ms) throw new Error("timed out waiting for " + what + "\nDOM text: " + app.textContent.slice(0, 400));
    await sleep(15);
  }
}
function all(node, pred, out = []) { for (const c of node.children || []) { if (pred(c)) out.push(c); all(c, pred, out); } return out; }
const h2 = () => (all(app, (n) => n.tag === "h2")[0] || { textContent: "" }).textContent;
const button = (t) => all(app, (n) => n.tag === "button" && n.textContent === t)[0];
const click = (n) => { assert(n, "button not found"); for (const f of n.listeners.click || []) f({}); };
function check(text, exact) {
  const lab = all(app, (n) => n.tag === "label" && (exact ? n.textContent === text : n.textContent.includes(text)))[0];
  assert(lab, "option not found: " + text);
  const cb = lab.children.find((c) => c.tag === "input");
  cb.checked = true;
  for (const f of cb.listeners.change || []) f({});
}
const numberBox = () => all(app, (n) => n.tag === "input" && n.attrs.type === "number")[0];

(async () => {
  vm.runInThisContext(fs.readFileSync(JS, "utf8"));
  await waitFor(() => button("I understand, continue"), "disclaimer");
  await waitFor(() => status.textContent.includes("Rules: on"), "status line");

  // ---- path 1: emergency screener item -> immediate emergency result, no further questions
  click(button("I understand, continue"));
  await waitFor(() => h2().includes("is any of this happening right now"), "screener");
  check("Struggling to breathe");
  click(button("Next"));
  await waitFor(() => app.textContent.includes("Call emergency services now"), "emergency banner");
  assert(app.textContent.includes("911"), "emergency number shown");
  assert(!app.textContent.includes("How old are you"), "no further questions after an emergency");

  // ---- path 2: ordinary symptom, with validation error and Back
  click(button("Start over"));
  await waitFor(() => button("I understand, continue"), "disclaimer again");
  click(button("I understand, continue"));
  await waitFor(() => h2().includes("is any of this happening right now"), "screener 2");
  check("None of these", true); click(button("Next"));
  await waitFor(() => h2().includes("How old are you"), "age");
  numberBox().value = "200"; click(button("Next"));
  await waitFor(() => app.textContent.includes("between 0 and 120"), "age validation message");
  numberBox().value = "30"; click(button("Next"));
  await waitFor(() => h2().includes("What is your sex"), "sex");
  check("Male", true); click(button("Next"));
  await waitFor(() => h2().includes("What are you experiencing"), "symptoms");
  click(button("Back"));
  await waitFor(() => h2().includes("What is your sex"), "back to sex");
  const male = all(app, (n) => n.tag === "input" && n.attrs.type === "radio" && n.value === "male")[0];
  assert(male && male.checked, "Back prefilled the earlier answer");
  click(button("Next"));
  await waitFor(() => h2().includes("What are you experiencing"), "symptoms again");
  check("Cough", true); click(button("Next"));
  await waitFor(() => h2().includes("How bad is it"), "severity"); click(button("2"));
  await waitFor(() => h2().includes("How long"), "duration"); check("About a day", true); click(button("Next"));
  await waitFor(() => h2().includes("readings handy"), "optional readings"); click(button("Next"));
  await waitFor(() => h2().includes("Anything else"), "free text"); click(button("Next"));
  await waitFor(() => app.textContent.includes("May be manageable at home"), "self-care banner");
  assert(button("Start over"), "start over offered");

  // ---- model text must never be interpreted as HTML (the page only ever uses textContent)
  assert(!/innerHTML|insertAdjacentHTML|document\.write|eval\(/.test(fs.readFileSync(JS, "utf8")), "unsafe DOM API used");
  console.log("OK");
})().catch((e) => { console.error("FAIL:", e.message); process.exit(1); });
