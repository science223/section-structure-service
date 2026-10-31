/* Zotero 7 bootstrap (https://www.zotero.org/support/dev/zotero_7_for_developers): load the two scripts, register
 * the preference pane, add the item menu entry to every main window, and remove it again on shutdown. */

var PaperStructure;
var PaperStructureNote;

function install() {}

async function startup({ id, version, rootURI }) {
  Services.scriptloader.loadSubScript(rootURI + "content/outline-note.js");
  Services.scriptloader.loadSubScript(rootURI + "content/paper-structure.js");
  PaperStructure.init({ id, version, rootURI });
  Zotero.PreferencePanes.register({
    pluginID: id,
    src: rootURI + "content/preferences.xhtml",
    scripts: [rootURI + "content/prefs-pane.js"],
    label: "Paper Structure",
    image: rootURI + "content/icon.svg",
  });
  for (const win of Zotero.getMainWindows()) PaperStructure.addToWindow(win);
}

function onMainWindowLoad({ window }) {
  PaperStructure?.addToWindow(window);
}

function onMainWindowUnload({ window }) {
  PaperStructure?.removeFromWindow(window);
}

function shutdown() {
  if (PaperStructure) {
    for (const win of Zotero.getMainWindows()) PaperStructure.removeFromWindow(win);
    PaperStructure.uninit();
  }
  PaperStructure = undefined;
  PaperStructureNote = undefined;
}

function uninstall() {}
