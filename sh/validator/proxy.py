"""Inference proxy (spec §12.2): the only thing an episode can talk to.

  * single-use bearer tokens per episode — issued by the host, revoked when the boundary is destroyed
  * the era's pinned sampling parameters OVERRIDE whatever the request carries (a bundle cannot change them)
  * per-call `usage` recorded by episode, independent of the agent's self-report
  * SSE streaming passed through, with `stream_options.include_usage` injected so usage is still captured

    python -m sh.validator.proxy --listen 0.0.0.0:8080 --upstream http://127.0.0.1:8080 --tokens /run/sh/tokens.json --usage-dir /var/sh/usage
The token store is a directory holding one file per live token; the host creates and removes them, and the proxy
reads only the one file a request names, so issuing many tokens at once cannot lose any of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "reasoning_effort",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
)


class Tokens:
    """A token store as a **directory with one file per token**.

    It began as a single JSON blob rewritten on every issue and revoke, which is a read-modify-write race: four
    episodes starting at once read the same blob and the last writer dropped the other three, so their agents met
    `401 no valid episode token` on their first call and the episodes died one turn in. Issue and revoke are now a
    single atomic filesystem operation each and there is nothing shared to lose.
    """

    def __init__(self, path: Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, token: str) -> Path:
        # the filename is a digest, so a token never lands in a directory listing, a log line or an `ls`
        return self.dir / f"{hashlib.sha256(token.encode()).hexdigest()}.json"

    def lookup(self, token: str) -> str | None:
        if not token:
            return None
        try:
            rec = json.loads(self._path(token).read_text())
        except (OSError, ValueError):
            return None
        if rec.get("expires", 0) < time.time():
            return None
        return rec.get("episode")

    def issue(self, episode: str, ttl: int) -> str:
        tok = "sh_" + secrets.token_urlsafe(32)
        path = self._path(tok)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"episode": episode, "expires": time.time() + ttl}))
        tmp.replace(path)  # atomic: a reader sees the whole record or no file at all
        return tok

    def revoke(self, token: str) -> None:
        self._path(token).unlink(missing_ok=True)

    def sweep(self) -> int:
        """Drop expired records. Not required for correctness — `lookup` checks the expiry itself."""
        n = 0
        for f in self.dir.glob("*.json"):
            try:
                if json.loads(f.read_text()).get("expires", 0) < time.time():
                    f.unlink(missing_ok=True)
                    n += 1
            except (OSError, ValueError):
                pass
        return n


def make_handler(upstream: str, tokens: Tokens, usage_dir: Path, sampling: dict):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _deny(self, code: int, msg: str):
            body = json.dumps({"error": msg}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            episode = tokens.lookup((self.headers.get("Authorization") or "").removeprefix("Bearer ").strip())
            if not episode:
                return self._deny(401, "no valid episode token")
            self._forward(b"", episode)

        def do_POST(self):
            episode = tokens.lookup((self.headers.get("Authorization") or "").removeprefix("Bearer ").strip())
            if not episode:
                return self._deny(401, "no valid episode token")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return self._deny(400, "bad json")
            if isinstance(body, dict):
                body.update({k: v for k, v in sampling.items() if v is not None})  # pinned sampling wins
                if body.get("stream"):
                    so = body.get("stream_options") or {}
                    so["include_usage"] = True
                    body["stream_options"] = so
                raw = json.dumps(body).encode()
            self._forward(raw, episode)

        def _forward(self, raw: bytes, episode: str):
            req = urllib.request.Request(
                upstream + self.path,
                data=raw if self.command == "POST" else None,
                method=self.command,
                headers={"Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")},
            )
            try:
                resp = urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as e:
                data = e.read()
                self.send_response(e.code)
                self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                return self._deny(502, f"upstream: {e!r}"[:200])
            ctype = resp.headers.get("Content-Type", "application/json")

            def record(u):  # written BEFORE the client sees the end of the response, so a reader never races it
                if self.command != "POST" or not u:
                    return
                usage_dir.mkdir(parents=True, exist_ok=True)
                with open(usage_dir / f"{episode}.jsonl", "a") as f:
                    f.write(json.dumps({"t": time.time(), "usage": u}) + "\n")

            if "text/event-stream" in ctype:
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for line in resp:
                    if line.startswith(b"data:") and b'"usage"' in line:
                        try:
                            record(json.loads(line[5:].strip()).get("usage"))
                        except ValueError:
                            pass
                    self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                data = resp.read()
                try:
                    record(json.loads(data).get("usage"))
                except ValueError:
                    pass
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def handle_error(self, request, client_address):
        """A client hanging up mid-response is ordinary; a traceback per occurrence buried the real errors."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(listen: str, upstream: str, tokens_path: Path, usage_dir: Path, sampling: dict):
    host, port = listen.rsplit(":", 1)
    return Server((host, int(port)), make_handler(upstream.rstrip("/"), Tokens(tokens_path), usage_dir, sampling))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:8080")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--tokens", required=True, help="token store DIRECTORY (one file per live token)")
    ap.add_argument("--usage-dir", required=True)
    ap.add_argument("--sampling", default='{"temperature": 0.2, "top_p": 0.95, "max_tokens": 8192}')
    a = ap.parse_args(argv)
    srv = serve(a.listen, a.upstream, Path(a.tokens), Path(a.usage_dir), json.loads(a.sampling))
    print(f"proxy on {a.listen} -> {a.upstream}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
