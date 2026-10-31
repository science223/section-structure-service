"""Minimal MCP stdio client: no SDK, just JSON-RPC lines, the way any agent host talks to the server.

Usage: python mcp_probe.py <paper-structure.exe> <out.json> <file> [<file> ...]
For each file: get_outline, then read_section on the first section id and search_document for "the".
"""
import json
import subprocess
import sys
import time

exe, out_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
proc = subprocess.Popen([exe, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        env=None)
log = []
next_id = 0


def send(method, params=None, notify=False):
    global next_id
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if not notify:
        next_id += 1
        msg["id"] = next_id
    proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
    proc.stdin.flush()
    if notify:
        return None
    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("server closed stdout; stderr: " + proc.stderr.read().decode("utf-8", "replace")[-2000:])
        resp = json.loads(line)
        if resp.get("id") == msg["id"]:
            log.append({"method": method, "params": params, "seconds": round(time.time() - t0, 3), "response": resp})
            return resp


def tool_text(resp):
    res = resp.get("result", {})
    return "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"), res.get("isError")


init = send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                           "clientInfo": {"name": "bare-probe", "version": "0"}})
print("initialize:", init.get("result", {}).get("serverInfo"), init.get("result", {}).get("protocolVersion"))
send("notifications/initialized", notify=True)
tools = send("tools/list", {})
names = [t["name"] for t in tools.get("result", {}).get("tools", [])]
print("tools:", names)

for f in files:
    resp = send("tools/call", {"name": "get_outline", "arguments": {"file_path": f}})
    text, is_error = tool_text(resp)
    print("\n==", f)
    print("get_outline isError=%s seconds=%s chars=%d" % (is_error, log[-1]["seconds"], len(text)))
    print(text[:600])
    try:
        outline = json.loads(text)
    except ValueError:
        continue
    if "error" in outline:
        continue
    secs = outline.get("sections") or []
    if secs:
        sid = secs[min(1, len(secs) - 1)].get("id")
        r = send("tools/call", {"name": "read_section", "arguments": {"file_path": f, "section_id": sid}})
        t, e = tool_text(r)
        print("read_section %s isError=%s chars=%d" % (sid, e, len(t)))
    r = send("tools/call", {"name": "search_document", "arguments": {"file_path": f, "query": "the"}})
    t, e = tool_text(r)
    print("search_document isError=%s chars=%d" % (e, len(t)))

proc.stdin.close()
try:
    proc.wait(timeout=20)
except subprocess.TimeoutExpired:
    proc.kill()
err = proc.stderr.read().decode("utf-8", "replace")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump({"tools": names, "calls": log, "stderr_tail": err[-4000:], "exit": proc.returncode}, fh,
              ensure_ascii=False, indent=1)
print("\nserver exit", proc.returncode, "stderr chars", len(err))
