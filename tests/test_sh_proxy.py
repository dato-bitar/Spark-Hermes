"""B1 contract: no token → 401; pinned sampling overrides the request; usage is recorded per episode; revoke works."""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sh.validator.proxy import Tokens, serve


class Fake(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        out = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_proxy_enforces_tokens_overrides_sampling_and_records_usage(tmp_path: Path):
    up = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    tokens = tmp_path / "tokens"
    usage = tmp_path / "usage"
    px = serve(
        "127.0.0.1:0", f"http://127.0.0.1:{up.server_port}", tokens, usage, {"temperature": 0.0, "max_tokens": 4096}
    )
    threading.Thread(target=px.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{px.server_port}/v1/chat/completions"
    body = json.dumps({"model": "m", "messages": [], "temperature": 1.5, "max_tokens": 10}).encode()

    def call(tok):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {tok}"} if tok else {})},
        )
        try:
            return urllib.request.urlopen(req, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code

    assert call(None) == 401
    t = Tokens(tokens)
    tok = t.issue("ep-1", ttl=60)
    assert call(tok) == 200
    assert Fake.seen[-1]["temperature"] == 0.0 and Fake.seen[-1]["max_tokens"] == 4096  # pinned sampling won
    rec = json.loads((usage / "ep-1.jsonl").read_text().splitlines()[0])
    assert rec["usage"]["prompt_tokens"] == 7
    t.revoke(tok)
    assert call(tok) == 401


def test_tokens_issued_concurrently_all_survive(tmp_path: Path):
    """The c=4 failure: a shared JSON blob rewritten per issue lost three of every four tokens, and those
    episodes died on their first call with `401 no valid episode token`."""
    import concurrent.futures

    t = Tokens(tmp_path / "tokens")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        toks = list(pool.map(lambda i: t.issue(f"ep-{i}", 60), range(64)))
    assert len(set(toks)) == 64
    assert [t.lookup(tok) for tok in toks] == [f"ep-{i}" for i in range(64)]


def test_revoking_one_token_leaves_the_others(tmp_path: Path):
    t = Tokens(tmp_path / "tokens")
    a, b = t.issue("ep-a", 60), t.issue("ep-b", 60)
    t.revoke(a)
    assert t.lookup(a) is None and t.lookup(b) == "ep-b"
    t.revoke(a)  # revoking twice is not an error


def test_an_expired_token_stops_working(tmp_path: Path):
    t = Tokens(tmp_path / "tokens")
    tok = t.issue("ep-a", -1)
    assert t.lookup(tok) is None
    assert t.sweep() == 1


def test_a_token_never_appears_in_the_store_listing(tmp_path: Path):
    t = Tokens(tmp_path / "tokens")
    tok = t.issue("ep-a", 60)
    assert all(tok not in p.name for p in (tmp_path / "tokens").iterdir())
