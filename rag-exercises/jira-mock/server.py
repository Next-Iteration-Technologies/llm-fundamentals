"""Tiny Jira mock for the Contoso RAG lab (fictional data). Standard library only.
GET  /rest/api/2/search?jql=project = ONB AND status != Done AND text ~ "BlueRiver"
GET  /rest/api/2/issue/ONB-15
PUT  /rest/api/2/issue/ONB-15   body {"fields": {"status": {"name": "Open"}}}   (or {"status": "Open"})
Reads ISSUES_FILE once; changes live in memory until restart."""
import json, os, re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ISSUES = json.load(open(os.environ.get("ISSUES_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "contoso-pack", "jira", "issues.json"))))["issues"]
BY_KEY = {i["key"]: i for i in ISSUES}


def match(issue, jql):
    f = issue["fields"]
    for cond in re.split(r"\s+AND\s+", jql.strip(), flags=re.I):
        m = re.match(r'(\w+)\s*(!=|=|~)\s*"?([^"]*?)"?\s*$', cond.strip(), re.I)
        if not m:
            continue
        field, op, val = m.group(1).lower(), m.group(2), m.group(3).lower()
        have = {"project": f["project"]["key"], "status": f["status"]["name"], "key": issue["key"],
                "assignee": (f.get("assignee") or {}).get("displayName", ""),
                "text": f"{f['summary']} {f.get('description') or ''}"}.get(field, "").lower()
        if (op == "=" and have != val) or (op == "!=" and have == val) or (op == "~" and not all(w in have for w in val.split())):
            return False
    return True


class H(BaseHTTPRequestHandler):
    def send(self, code, obj):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/rest/api/2/search":
            jql = parse_qs(u.query).get("jql", [""])[0]
            hits = [i for i in ISSUES if match(i, jql)]
            return self.send(200, {"jql": jql, "total": len(hits), "issues": hits})
        if u.path.startswith("/rest/api/2/issue/") and u.path.rsplit("/", 1)[1] in BY_KEY:
            return self.send(200, BY_KEY[u.path.rsplit("/", 1)[1]])
        self.send(404, {"error": "not found", "try": "/rest/api/2/search?jql=project=ONB"})

    def do_PUT(self):
        key = urlparse(self.path).path.rsplit("/", 1)[1]
        if key not in BY_KEY:
            return self.send(404, {"error": f"unknown issue {key}"})
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        status = (data.get("fields", {}).get("status") or {}).get("name") or data.get("status")
        if status:
            BY_KEY[key]["fields"]["status"]["name"] = status
            BY_KEY[key]["fields"]["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.send(200, BY_KEY[key])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"jira-mock on :{port} with {len(ISSUES)} issues")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
