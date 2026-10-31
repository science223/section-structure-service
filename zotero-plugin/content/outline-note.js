/* Builds the Zotero note for a paper's section outline. Pure functions (no Zotero objects), so they are tested with
 * Node (tests/zotero/test_outline_note.mjs) and loaded into the plugin as is.
 *
 * view: the job view of the local service (GET /api/jobs/<id>): sections [{id, title, depth, page, tokens}],
 *       stats {tokens_full, ...}, pages, unlocated [{predicted_title}].
 * link: {library: "library" | "groups/<id>", key: "<attachment key>"} -> zotero://open-pdf links per section page.
 */
var PaperStructureNote = (function () {
  const MARKER = "章节结构（paper-structure）";

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function pageLink(link, page) {
    if (!link || !link.key || !page) return null;
    return `zotero://open-pdf/${link.library}/items/${link.key}?page=${page}`;
  }

  /** Nested <ul> from the flat, ordered section list (depth 1..n). */
  function outlineList(sections, link) {
    // A deeper section opens a list inside the open <li> of its parent; a shallower one closes lists back to its
    // depth. The title already carries its number ("2.1 Data"), so the id is not repeated.
    // Skipping a level (a document that starts at depth 2, or 1 -> 3) puts an empty <li> around the inner list.
    let html = "";
    let level = 0;
    let liOpen = false;
    for (const s of sections) {
      const d = Math.max(1, s.depth | 0);
      if (d > level) {
        while (level < d) {
          if (level > 0 && !liOpen) html += "<li>";
          html += "<ul>";
          level++;
          liOpen = false;
        }
      } else {
        html += "</li>";
        while (level > d) { html += "</ul></li>"; level--; }
      }
      const title = esc(String(s.title).replace(/\s+/g, " ").trim());
      const href = pageLink(link, s.page);
      const head = href ? `<a href="${esc(href)}">${title}</a>` : title;
      const meta = [s.page ? `p. ${s.page}` : null, s.tokens ? `~${s.tokens} tokens` : null].filter(Boolean).join(" · ");
      html += `<li>${head}${meta ? ` <span style="color:#888">${esc(meta)}</span>` : ""}`;
      liOpen = true;
    }
    if (level > 0) {
      html += "</li>";
      while (level > 1) { html += "</ul></li>"; level--; }
      html += "</ul>";
    }
    return html;
  }

  function buildNoteHTML(view, link, when) {
    const secs = view.sections || [];
    const stats = view.stats || {};
    const facts = [
      `${secs.length} 节`,
      view.pages ? `${view.pages} 页` : null,
      stats.tokens_full ? `全文约 ${stats.tokens_full} tokens` : null,
      when ? `生成于 ${when}` : null,
    ].filter(Boolean).join(" · ");
    let html = `<h1>${esc(MARKER)}</h1><p>${esc(facts)}</p>`;
    html += secs.length ? outlineList(secs, link) : "<p>没有识别出章节标题。</p>";
    const un = (view.unlocated || []).map((u) => u.predicted_title).filter(Boolean);
    if (un.length) html += `<p>模型识别出、但在原文里没定位到的标题：${esc(un.join("；"))}</p>`;
    html += `<p style="color:#888">本地模型识别，不联网；点标题跳到 PDF 对应页。</p>`;
    return html;
  }

  function isOutlineNote(noteHTML) {
    return typeof noteHTML === "string" && noteHTML.includes(MARKER);
  }

  return { MARKER, esc, pageLink, outlineList, buildNoteHTML, isOutlineNote };
})();

if (typeof module !== "undefined") module.exports = PaperStructureNote;
