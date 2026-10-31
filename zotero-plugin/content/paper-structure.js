/* Paper Structure for Zotero 7: item menu "提取章节结构（本地）".
 * For each selected item: its PDF attachment's local path -> the local service (POST /api/open_path; the service runs
 * the model on this computer, nothing leaves it) -> poll GET /api/jobs/<id> -> a child note with the outline, each
 * section linked to its PDF page (zotero://open-pdf/...?page=N). An existing outline note is replaced, not duplicated.
 * The service: `paper-structure app --no-browser` (address in the preference pane). */

var PaperStructure = {
  id: null,
  version: null,
  rootURI: null,
  MENU_IDS: ["paper-structure-sep", "paper-structure-extract"],

  init({ id, version, rootURI }) {
    this.id = id;
    this.version = version;
    this.rootURI = rootURI;
    Zotero.PaperStructure = this;   // the preference pane script reaches the plugin through Zotero
  },

  uninit() {
    if (Zotero.PaperStructure === this) delete Zotero.PaperStructure;
  },

  server() {
    return String(Zotero.Prefs.get("paperstructure.server") || "http://127.0.0.1:8765").trim().replace(/\/+$/, "");
  },

  async http(method, path, body) {
    const xhr = await Zotero.HTTP.request(method, this.server() + path, {
      body: body === undefined ? undefined : JSON.stringify(body),
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      responseType: "json",
      successCodes: false,
      timeout: 30000,
    });
    return { status: xhr.status, data: xhr.response };
  },

  async info() {
    const r = await this.http("GET", "/api/info");
    if (r.status !== 200 || !r.data) throw new Error(`本地服务返回 ${r.status}`);
    return r.data;
  },

  // ---- main window menu
  addToWindow(win) {
    const doc = win.document;
    const menu = doc.getElementById("zotero-itemmenu");
    if (!menu || doc.getElementById("paper-structure-extract")) return;
    const sep = doc.createXULElement("menuseparator");
    sep.id = "paper-structure-sep";
    const mi = doc.createXULElement("menuitem");
    mi.id = "paper-structure-extract";
    mi.setAttribute("label", "提取章节结构（本地）");
    mi.addEventListener("command", () => this.extractSelected(win));
    menu.appendChild(sep);
    menu.appendChild(mi);
  },

  removeFromWindow(win) {
    for (const id of this.MENU_IDS) win.document.getElementById(id)?.remove();
  },

  // ---- extraction
  topItems(items) {
    const out = new Map();
    for (const it of items) {
      const top = it.isRegularItem() ? it : it.parentItem;
      if (top && top.isRegularItem()) out.set(top.id, top);
    }
    return [...out.values()];
  },

  async pdfAttachment(item) {
    const best = await item.getBestAttachment();
    if (best && best.isPDFAttachment()) return best;
    for (const id of item.getAttachments()) {
      const a = Zotero.Items.get(id);
      if (a && a.isPDFAttachment()) return a;
    }
    return null;
  },

  async waitJob(jobId, onProgress) {
    for (let i = 0; i < 7200; i++) {   // up to two hours: a 200-page document on a CPU
      const r = await this.http("GET", `/api/jobs/${jobId}`);
      const v = r.data || {};
      if (r.status !== 200) throw new Error(v.error || `本地服务返回 ${r.status}`);
      if (v.status === "done") return v;
      if (v.status === "error") throw new Error(v.error || "处理失败");
      onProgress(v.done || 0, v.total || 0);
      await Zotero.Promise.delay(1000);
    }
    throw new Error("处理超时");
  },

  libraryPath(att) {
    return att.libraryID === Zotero.Libraries.userLibraryID
      ? "library" : `groups/${Zotero.Groups.getGroupIDFromLibraryID(att.libraryID)}`;
  },

  async writeNote(item, att, view) {
    const when = new Date().toLocaleString("zh-CN", { hour12: false });
    const html = PaperStructureNote.buildNoteHTML(view, { library: this.libraryPath(att), key: att.key }, when);
    let note = null;
    for (const id of item.getNotes()) {
      const n = Zotero.Items.get(id);
      if (n && PaperStructureNote.isOutlineNote(n.getNote())) { note = n; break; }
    }
    if (!note) {
      note = new Zotero.Item("note");
      note.libraryID = item.libraryID;
      note.parentID = item.id;
    }
    note.setNote(html);
    await note.saveTx();
    return note;
  },

  async extractSelected(win) {
    const items = this.topItems(win.ZoteroPane.getSelectedItems());
    const pw = new Zotero.ProgressWindow({ closeOnClick: true });
    pw.changeHeadline("章节结构（本地模型）");
    pw.show();
    if (!items.length) {
      pw.addDescription("请选中带 PDF 附件的文献条目（独立的 PDF 需要先创建父条目）。");
      pw.startCloseTimer(6000);
      return;
    }
    try {
      await this.info();
    } catch (e) {
      pw.addDescription(`没连上本地服务 ${this.server()}。请先在命令行运行：paper-structure app --no-browser`);
      pw.startCloseTimer(10000);
      return;
    }
    for (const item of items) {
      const title = item.getDisplayTitle() || item.key;
      const line = new pw.ItemProgress("", title);
      try {
        const att = await this.pdfAttachment(item);
        if (!att) throw new Error("没有 PDF 附件");
        const path = await att.getFilePathAsync();
        if (!path) throw new Error("PDF 文件不在本机（可能还没同步下来）");
        const r = await this.http("POST", "/api/open_path", { path });
        if (r.status !== 200) throw new Error((r.data && r.data.error) || `本地服务返回 ${r.status}`);
        line.setProgress(2);
        const view = await this.waitJob(r.data.job_id, (done, total) => {
          if (total) line.setProgress(Math.max(2, Math.round((98 * done) / total)));
        });
        await this.writeNote(item, att, view);
        line.setProgress(100);
        line.setText(`${title}：${(view.sections || []).length} 节${view.cached ? "（缓存）" : ""}`);
      } catch (e) {
        line.setError();
        line.setText(`${title}：${e.message}`);
        Zotero.logError(e);
      }
    }
    pw.startCloseTimer(8000);
  },

  async testConnection(doc) {
    const out = doc.getElementById("paperstructure-status");
    out.textContent = "连接中…";
    try {
      const i = await this.info();
      out.textContent = `已连接：模型 ${i.model || "未设置"}，设备 ${i.device}，版本 ${i.version}`;
    } catch (e) {
      out.textContent = `没连上 ${this.server()}：请先运行 paper-structure app --no-browser（${e.message}）`;
    }
  },
};
