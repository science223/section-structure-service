/* Preference pane: the "测试连接" button asks the local service for its model, device and version. */
(function () {
  const button = document.getElementById("paperstructure-test");
  if (button && Zotero.PaperStructure) {
    button.addEventListener("command", () => Zotero.PaperStructure.testConnection(document));
  }
})();
