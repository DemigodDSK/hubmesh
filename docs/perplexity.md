# Field report: hubmesh as a Perplexity connector

*July 2026 · hubmesh v0.3.1–v0.3.2 · Perplexity connectors (MCP over
SSE), solver model: Claude Sonnet 5 Thinking*

hubmesh's MCP server was wired into Perplexity as a custom connector
and driven through a 10-prompt test battery on a fictional corpus.
Everything below is from live sessions, cross-checked against server
logs (every claimed tool call was verified against the server's
`CallToolRequest` count — several answers that *looked* tool-backed
were actually answered from chat context, and are marked as such).

## Setup (two commands since v0.3.2)

```bash
pip install "hubmesh[mcp]"
python -m spacy download en_core_web_sm

hubmesh-mcp --transport sse --port 8000 --allow-tunnel
ngrok http 8000        # free tier is fine; note the https URL
```

Perplexity → Settings → Connectors → Add:

| Field | Value |
|---|---|
| Name | `hubmesh` |
| MCP server URL | `https://<your-ngrok-url>/sse` |
| Authentication | None |
| Transport | SSE |
| Network access | Public |

Then start a **new chat** (hard-reload the tab if the connector isn't
picked up) and index something:

> Use the hubmesh connector: index a corpus named "docs" with these
> documents: …

### What NOT to use (learned the hard way)

- **supergateway** — unnecessary (hubmesh serves SSE natively) and it
  crashes with "Already connected to a transport" on the second SSE
  connection after a session closes. If you saw 502s and restarted it
  repeatedly: that was why.
- **cloudflared quick tunnels** — plain HTTP passes but SSE bodies are
  buffered indefinitely; tool calls hang. Both QUIC and HTTP/2
  transports affected in our tests.
- Omitting `--allow-tunnel` — proxied requests arrive with the
  tunnel's Host header and get **421 Misdirected Request**.

### Error decoder

| Symptom | Actual cause |
|---|---|
| `[FETCHER_HTML_STATUS_CODE_ERROR]` on Add | Your server/tunnel is down (Perplexity's probe got an error page) |
| `[API_CLIENTS_ERROR] Server does not support automatic registration` | Endpoint dead → Perplexity fell back to an OAuth registration attempt. Restart server + tunnel, retry |
| Tool call starts, then "Error during tool execution" | Timeout — server cold start exceeded Perplexity's ~5–15s window. Fixed in v0.3.1 (startup warm-up); upgrade |
| "Answer stopped before finishing", no tool call in server logs | Perplexity's model stalled *thinking* — see finding 1 below |

## Test battery: 9/9

Corpus: 12 fictional documents (fictional so the model cannot answer
from world knowledge) containing a 4-hop acquisition/founder chain, a
name-collision trap, an alias/rebrand, funding numbers, and noise.

| Test | Result | Real tool calls |
|---|---|---|
| 2-hop bridge question | pass | 1 |
| 4-hop chain to a city | pass | 0 (answered from chat context — reused earlier retrievals, and said so) |
| Name-collision ("Who founded Aurora?") | pass — surfaced both candidates, asked which | 1 |
| Alias resolution (pre-rebrand company name) | pass | 1 |
| Numeric facts (funding rounds) | pass | 0 (chat context) |
| **Explicit iterative retrieval** | **pass — two seeded calls, hop 2 aimed at the entity discovered in hop 1** | 2 |
| Negative control (fact absent from corpus) | pass — retrieved first, *then* refused; no hallucination | 1 |
| Presupposition trap (see below) | pass on attempt 3 | 1 |
| `path_between` two entities | pass — and exposed a real design lesson (finding 2) | 1 |

The iterative test is the architectural headline: Perplexity's model
ran hubmesh's intended loop — retrieve, read, pass the discovered
bridge entity as `seed_entities`, mask consumed docs with
`exclude_docs` — without any custom prompting beyond "answer step by
step, retrieving once per hop."

## Finding 1: presupposition traps stall reasoning models — an escape hatch fixes it

One battery question deliberately presupposes an answer the corpus
does not contain: "Which university did the founder of the company
that **acquired** X study at?" — where the corpus states the education
of the *acquired* company's founder, but not the acquirer's.

Twice in a row, Perplexity's reasoning model connected, listed tools,
and then **died in its thinking phase without issuing a single tool
call** ("Answer stopped before finishing"; server logs show the
handshake and `ListToolsRequest` succeeding, then silence). Adding one
sentence — *"If the corpus does not state this, say so explicitly."* —
produced immediate convergence: a real retrieval, a correct refusal,
and the model *naming the trap it had avoided* (identifying that the
available university fact belongs to the other founder).

Hypothesis: a question presupposing a nonexistent answer leaves a
reasoning model with no licensed conclusion — it senses the tempting
answer is wrong but doesn't treat "there is no answer" as an available
completion until the prompt legitimizes it. Practical rule for
grounded-retrieval prompts: **give the model explicit permission to
report absence.**

## Finding 2: shortest path ≠ meaningful path

Asked to connect two founders, `path_between` returned a technically
correct 2-hop path through a place-name that appears incidentally in
both entities' documents — while the meaningful 4-hop chain (through
the acquisition and a shared advisor) was longer and went unreported.
Perplexity, to its credit, honestly described the link as "bridged
purely through the textual mention" rather than claiming a
relationship.

v0.3.2 addresses this: `path_between` now returns up to `k_paths`
*distinct routes* (detour-variants of an already-reported bridge are
filtered), each annotated with `hops` and `via_documents`, so the
calling agent can weigh the shortcut against the substantive chain.
Live re-test: both routes surface.

## Caveats

Single client (Perplexity), single solver model (Claude Sonnet 5
Thinking), a 12-document corpus, and one session per prompt — this is
a field report, not a benchmark. With a corpus this small, one
default-`top_k` retrieval sweeps most documents; forcing genuine
multi-hop behaviour required `top_k=3`. The retrieval-recall
benchmarks live in [BENCHMARKS.md](../BENCHMARKS.md).
