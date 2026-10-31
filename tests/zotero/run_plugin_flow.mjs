// Runs the plugin's extraction against a real local service with a fake Zotero (tests/test_zotero_plugin.py starts
// the service). Usage: node run_plugin_flow.mjs <service base url> <pdf path>. Prints the result as JSON.
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const [base, pdfPath] = process.argv.slice(2);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../zotero-plugin");

class FakeItem {
  constructor(kind, props = {}) {
    Object.assign(this, { kind, id: FakeItem.next++, libraryID: 1, key: `K${FakeItem.next}`, notes: [], attachments: [] }, props);
  }
  isRegularItem() { return this.kind === "regular"; }
  isPDFAttachment() { return this.kind === "pdf"; }
  getDisplayTitle() { return this.title || ""; }
  async getBestAttachment() { return this.attachments[0] || null; }
  getAttachments() { return this.attachments.map((a) => a.id); }
  getNotes() { return this.notes.map((n) => n.id); }
  async getFilePathAsync() { return this.path; }
  getNote() { return this.html || ""; }
  setNote(html) { this.html = html; }
  async saveTx() {
    store.set(this.id, this);
    const parent = store.get(this.parentID);
    if (parent && !parent.notes.includes(this)) parent.notes.push(this);
    saves.push(this.id);
  }
}
FakeItem.next = 1;
const store = new Map();
const saves = [];
const lines = [];

const Zotero = {
  Prefs: { get: (k) => (k === "paperstructure.server" ? base : undefined) },
  HTTP: {
    async request(method, url, opts) {
      const r = await fetch(url, { method, body: opts.body, headers: opts.headers });
      let response = null;
      try { response = await r.json(); } catch { response = null; }
      return { status: r.status, response };
    },
  },
  Promise: { delay: (ms) => new Promise((res) => setTimeout(res, Math.min(ms, 100))) },
  Items: { get: (id) => store.get(id) },
  Item: function (kind) { return new FakeItem(kind); },
  Libraries: { userLibraryID: 1 },
  Groups: { getGroupIDFromLibraryID: (id) => id * 10 },
  ProgressWindow: class {
    changeHeadline() {}
    show() {}
    addDescription(t) { lines.push({ description: t }); }
    startCloseTimer() {}
    get ItemProgress() {
      return class {
        constructor(icon, text) { this.rec = { text, progress: 0, error: false }; lines.push(this.rec); }
        setProgress(p) { this.rec.progress = p; }
        setText(t) { this.rec.text = t; }
        setError() { this.rec.error = true; }
      };
    }
  },
  logError: (e) => lines.push({ logged: String(e) }),
};

const ctx = vm.createContext({ Zotero, console, setTimeout });
for (const f of ["content/outline-note.js", "content/paper-structure.js"]) {
  vm.runInContext(fs.readFileSync(path.join(root, f), "utf8"), ctx, { filename: f });
}
const PS = vm.runInContext("PaperStructure", ctx);
PS.init({ id: "paper-structure@wku.edu.cn", version: "0.1.0", rootURI: "file:///x/" });

const pdf = new FakeItem("pdf", { path: pdfPath, libraryID: 1 });
const paper = new FakeItem("regular", { title: "A paper", attachments: [pdf] });
pdf.parentItem = paper;
const noPdf = new FakeItem("regular", { title: "No PDF" });
for (const it of [pdf, paper, noPdf]) store.set(it.id, it);
const win = { ZoteroPane: { getSelectedItems: () => [pdf, paper, noPdf] } };   // the PDF and its parent: one item

await PS.extractSelected(win);
const first = paper.notes.map((n) => n.html);
await PS.extractSelected(win);   // again: the outline note is replaced, not added
console.log(JSON.stringify({ lines, notes_after_first: first.length, notes_after_second: paper.notes.length,
  html: paper.notes[0] && paper.notes[0].html, pdfKey: pdf.key, saves: saves.length }));
