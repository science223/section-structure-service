// Node tests of zotero-plugin/content/outline-note.js:  node tests/zotero/test_outline_note.mjs
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const N = require("../../zotero-plugin/content/outline-note.js");

function balanced(html) {
  for (const tag of ["ul", "li"]) {
    const open = (html.match(new RegExp(`<${tag}>|<${tag} `, "g")) || []).length;
    const close = (html.match(new RegExp(`</${tag}>`, "g")) || []).length;
    assert.equal(open, close, `${tag}: ${open} open, ${close} closed in ${html}`);
  }
}

const view = {
  sections: [
    { id: "abstract", title: "Abstract", depth: 1, page: 1, tokens: 50 },
    { id: "1", title: "1 Introduction", depth: 1, page: 1, tokens: 300 },
    { id: "2", title: "2 Method", depth: 1, page: 2, tokens: 120 },
    { id: "2.1", title: "2.1  Data <raw>", depth: 2, page: 2, tokens: 80 },
    { id: "2.1.1", title: "2.1.1 Cleaning", depth: 3, page: 3, tokens: 40 },
    { id: "3", title: "3 Results & Discussion", depth: 1, page: 4, tokens: 200 },
  ],
  stats: { tokens_full: 900 }, pages: 5, unlocated: [{ predicted_title: "Appendix Z" }],
};
const link = { library: "library", key: "ABCD1234" };

// nesting: 2.1 inside 2, 2.1.1 inside 2.1, 3 back at the top
const list = N.outlineList(view.sections, link);
balanced(list);
assert.match(list, /<li><a href="zotero:\/\/open-pdf\/library\/items\/ABCD1234\?page=2">2 Method<\/a>[^]*?<ul><li><a [^>]*>2\.1 Data &lt;raw&gt;<\/a>[^]*?<ul><li>/);
assert.match(list, /<\/ul><\/li><\/ul><\/li><li><a [^>]*page=4">3 Results &amp; Discussion<\/a>/);
assert.ok(!list.includes("2.1 2.1"), "the id is not repeated before a numbered title");

// whole note: marker, facts, unlocated, escaped
const html = N.buildNoteHTML(view, link, "2026-10-21 15:00");
balanced(html);
assert.ok(N.isOutlineNote(html));
assert.ok(html.includes("6 节") && html.includes("5 页") && html.includes("全文约 900 tokens"));
assert.ok(html.includes("Appendix Z"));
assert.ok(!html.includes("<raw>"));

// group library links, no page -> no link
assert.equal(N.pageLink({ library: "groups/42", key: "K" }, 7), "zotero://open-pdf/groups/42/items/K?page=7");
assert.equal(N.pageLink(link, null), null);
const noPages = N.outlineList([{ id: "1", title: "1 Intro", depth: 1 }], link);
assert.ok(!noPages.includes("<a "));
balanced(noPages);

// a document that starts deep, and an empty one
balanced(N.outlineList([{ id: "a", title: "A.1 x", depth: 2 }, { id: "b", title: "B y", depth: 1 }], link));
assert.ok(N.buildNoteHTML({ sections: [] }, link).includes("没有识别出章节标题"));
assert.equal(N.isOutlineNote("<p>my own note</p>"), false);
console.log("outline-note: all tests passed");
