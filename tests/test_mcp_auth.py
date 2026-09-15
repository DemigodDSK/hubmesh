"""Batch-2 regression tests: MCP network authentication and write policy.

External review findings: the SSE transport exposed read/write tools
with no authentication, and the documented tunnel recipe disabled
DNS-rebinding protection on top. Policy now: network exposure requires
a bearer key; tunnels default to read-only; stdio is untouched.
"""
import asyncio

import pytest

mcp_sdk = pytest.importorskip("mcp", reason="mcp extra not installed")

from hubmesh.mcp_server import (  # noqa: E402
    BearerAuthASGI, SecurityPolicy, build_sse_app, resolve_security,
)
import hubmesh.mcp_server as srv  # noqa: E402


# ---- policy matrix -------------------------------------------------------

class TestResolveSecurity:
    def test_loopback_sse_needs_no_key(self):
        p = resolve_security("sse", "127.0.0.1", False, None, False, False)
        assert p.api_key is None and p.read_only is False

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::"])
    def test_nonloopback_bind_without_key_refuses(self, host):
        with pytest.raises(SystemExit, match="without authentication"):
            resolve_security("sse", host, False, None, False, False)

    def test_tunnel_without_key_refuses(self):
        with pytest.raises(SystemExit, match="without authentication"):
            resolve_security("sse", "127.0.0.1", True, None, False, False)

    def test_tunnel_with_key_defaults_read_only(self):
        p = resolve_security("sse", "127.0.0.1", True, "s3cret", False, False)
        assert p.api_key == "s3cret" and p.read_only is True

    def test_tunnel_allow_writes_is_explicit(self):
        p = resolve_security("sse", "127.0.0.1", True, "s3cret", False, True)
        assert p.read_only is False

    def test_read_only_flag_sticks_everywhere(self):
        p = resolve_security("sse", "127.0.0.1", False, None, True, False)
        assert p.read_only is True

    def test_stdio_never_requires_key(self):
        p = resolve_security("stdio", "0.0.0.0", True, None, False, False)
        assert p.api_key is None


# ---- bearer middleware over real ASGI semantics --------------------------

async def _call(app, headers):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "method": "GET", "path": "/sse",
             "headers": headers}
    await app(scope, receive, send)
    return sent


async def _inner_ok(scope, receive, send):
    await send({"type": "http.response.start", "status": 200,
                "headers": []})
    await send({"type": "http.response.body", "body": b"reached"})


class TestBearerAuthASGI:
    def test_missing_header_is_401_with_challenge(self):
        app = BearerAuthASGI(_inner_ok, "k3y")
        sent = asyncio.run(_call(app, []))
        assert sent[0]["status"] == 401
        assert (b"www-authenticate", b"Bearer") in sent[0]["headers"]

    def test_wrong_key_is_401(self):
        app = BearerAuthASGI(_inner_ok, "k3y")
        sent = asyncio.run(_call(app, [(b"authorization", b"Bearer nope")]))
        assert sent[0]["status"] == 401

    def test_correct_key_reaches_app(self):
        app = BearerAuthASGI(_inner_ok, "k3y")
        sent = asyncio.run(_call(app, [(b"authorization", b"Bearer k3y")]))
        assert sent[0]["status"] == 200
        assert sent[1]["body"] == b"reached"

    def test_lifespan_passes_through(self):
        seen = []

        async def inner(scope, receive, send):
            seen.append(scope["type"])

        async def run():
            await BearerAuthASGI(inner, "k")({"type": "lifespan"},
                                             None, None)
        asyncio.run(run())
        assert seen == ["lifespan"]


# ---- transport assembly + write policy -----------------------------------

class TestAssemblyAndWrites:
    def test_sse_app_wrapped_only_when_key_set(self):
        assert isinstance(build_sse_app("k"), BearerAuthASGI)
        assert not isinstance(build_sse_app(None), BearerAuthASGI)

    def test_sse_401_over_http(self):
        """Real HTTP semantics through the wrapped SSE app: an
        unauthenticated request never reaches an MCP endpoint."""
        starlette_tc = pytest.importorskip("starlette.testclient")
        client = starlette_tc.TestClient(build_sse_app("k3y"))
        r = client.get("/sse")
        assert r.status_code == 401
        assert r.json()["error"].startswith("unauthorized")

    def test_index_corpus_refuses_in_read_only(self, monkeypatch):
        monkeypatch.setattr(srv, "_read_only", True)
        out = srv.index_corpus.fn if hasattr(srv.index_corpus, "fn") else None
        fn = out or srv.index_corpus
        result = fn("any", [{"id": "1", "text": "t"}])
        assert "read-only" in result["error"]

    def test_policy_dataclass_shape(self):
        p = SecurityPolicy(api_key="a", read_only=True)
        assert (p.api_key, p.read_only) == ("a", True)
