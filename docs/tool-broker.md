# Read-only tool broker and evidence ledger

`krummholz_llm/tool_broker.py` implements the host boundary described in
[ADR 002](ADR-002-evidence-seeking-training.md). It is model-independent
infrastructure. **No live search service or HTTP adapter is connected, and no
model weights were trained or approved by this work.** The frozen GPU comparison
and Android application's network permissions are unchanged.

The implemented loop is: validate one proposal, call a host-supplied adapter
within its budget, validate its result, and record any fetched evidence. Model
selection, model-specific tool-call rendering, answer generation and semantic
review remain separate.

## Proposal interface

The version is `krummholz-readonly-tools-v1`. Exactly two operations exist:

```json
{"version":"krummholz-readonly-tools-v1","name":"search_public_web","arguments":{"query":"an exact host-authorized public query"}}
```

```json
{"version":"krummholz-readonly-tools-v1","name":"fetch_public_page","arguments":{"result_id":"a handle returned by this broker's search"}}
```

Duplicate keys, extra fields, unknown tools, arbitrary fetch URLs and malformed
values are rejected. There are no shell, filesystem, environment-variable,
private-document, message-sending, purchase or account-management operations.
Models cannot add capabilities through source text or JSON arguments.

`PublicScope` is trusted application configuration. It supplies the profile,
an exact set of public queries, permitted hostnames and whether this request
requires fresh retrieval. Query text must match exactly: private additions,
extra words and whitespace cannot be silently normalized into authorization.
This is deliberately a narrow starting contract. The application can construct
eligible public queries from the user's request; a future richer query policy
needs its own tests and must remain outside the model's authority.

Use a separate broker per user request and invoke it serially. Profile names and
request-local IDs prevent accidental mixing; they are not tenant authentication.
The calling application still owns user authorization and decides whether a
query is appropriate to send publicly. This module does not infer privacy from
arbitrary natural-language text.

## Adapter contracts

Adapters are trusted host code, supplied as picklable objects with these methods:

| Method | Input fields, in addition to timeout_seconds | Exact output |
| --- | --- | --- |
| `search(request)` | `query`, `max_results` | `{"results":[{"url":"…","title":"…","snippet":"…"}]}` |
| `resolve(request)` | `hostname` | `{"addresses":["public IP", "…"]}` |
| `fetch(request)` | `url`, validated `addresses`, `tls_server_name`, `max_body_bytes`, `follow_redirects:false`, `send_credentials:false`, `send_cookies:false` | `{"status":200,"location":null,"text":"…","media_type":"text/plain"}` |

The search adapter and fetch adapter can be separate objects. DNS resolution
and one-hop fetching belong to the fetch adapter. No implementation is supplied
here: the tests use explicitly fictional responses.

A real fetch adapter **must connect only to the supplied IP addresses**, preserve
the URL hostname for TLS/SNI and certificate verification, enforce streaming
body/time limits, and disable automatic redirects, authentication and cookies.
It must not resolve the hostname again during the connection; otherwise DNS
rebinding could defeat the broker's validation. Auditing and integration-testing
that actual transport is required before claiming live retrieval works. A
trusted adapter that ignores its contract is not sandboxed by this module.

The broker checks HTTPS, the exact hostname scope, default TLS port and absent
userinfo/fragments. URL query strings are denied by default. A host may explicitly
allow known public parameter names through `allowed_url_query_parameters`; common
credential and signed-URL names are still rejected, including camelCase and
hyphenated forms. That allowlist must match the real endpoint's public semantics:
no string heuristic can determine whether every arbitrary value is a secret.
DNS answers must
contain only public addresses; mixed public/private answers, loopback, link-local,
private, carrier-grade NAT, multicast and mapped private IPv6 addresses are
rejected before fetching. The broker handles each redirect itself, repeating
host and DNS checks. Redirects cannot enlarge the caller's hostname scope.

Adapters return only bounded JSON across a process boundary. Each operation
runs in a separate child process. A timeout terminates and reaps that child;
late results are discarded. Provider logs and exception details are not returned
to the model. Unrelated environment credentials are removed in the child. A
search provider's future credentials belong in its trusted host configuration,
never in model arguments, prompts or evidence.

This is a capability boundary and process timeout, not a general operating-system
sandbox for malicious provider code. Configure only reviewed adapters. No
provider is selected by model output.

## Bounds and evidence

Defaults are six tool proposals, 30 seconds total and 10 seconds per proposal,
including its DNS/fetch steps; five search results; a 512-character query;
64 KiB per fetched body; 192 KiB cumulative fetched bodies; and three redirects.
The constructor enforces finite upper bounds on host settings. Process cleanup
can add up to half a second after the deadline. If a child cannot be reaped, the
broker disables further calls rather than launching more work.

Search results are discovery metadata, marked untrusted and non-citable.
Only a successful bounded fetch can create an evidence record. The host mints
both result handles and source IDs; a provider cannot supply its own source ID.
Records retain the final URL, redirect chain, body hash, retrieval time, media
type and original text. Text remains untrusted, including any embedded
instructions. HTML is not executed or rendered by this module.

The ledger lives in memory, and there is no automatic payload logging or file
save. Callers receive copies of its views. It records outcomes and budgets but
does not copy rejected private queries into an event log. A future display must
escape source/model text, and a future persistence layer must have an explicit
retention policy.

`assess_citations` checks only that each ID belongs to this request, each exact
quote appears in its fetched text, and—when required—the cited retrieval was
live and recent enough. Missing sources, unavailable providers, stale receipts,
invalid quotes and exhausted budgets remain explicit unresolved outcomes.
Fixture observations cannot satisfy a live-retrieval requirement.

A recent download does not make an old publication current. A matching quote
does not establish that it supports an answer. Accordingly, the most favorable
assessment is `evidence_ready_for_semantic_review`, with
`semantic_support:"not_determined"` and
`current_facts:"not_established_by_retrieval_time"`. Source IDs are request-local
handles, not authenticated publisher identities. The broker does not grade a
model answer, certify facts or approve a model.

## Local checks and next integration

Run the model-free tests:

```text
python -m unittest tests.test_tool_broker -v
```

The tests include actual child-process timeout/cleanup and credential-isolation
checks. All search, DNS and page responses are fictional; none contacts a real
service. Tests that set `provider_mode="live"` simulate that host configuration
solely to exercise freshness decisions, without making live calls.

After selecting a capable comparator, the next implementation is a separately
versioned tool-loop diagnostic using its exact publisher template. Normalize
the model's proposal into this broker interface, pass actual observations back
as untrusted tool results, and validate the final answer against those records.
Keep the original 36-case source-QA comparison unchanged. Use separate tasks
covering sufficient supplied evidence, current facts, unavailable/stale/conflicting
sources, malicious pages, calculations and unnecessary searching.

Connect a reviewed live search/HTTP adapter before claiming web search works.
Local-document retrieval and bounded calculations are subsequent distinct tools;
neither is silently exposed by this increment. Only after the unchanged base and
retrieval-assisted base are measured should source-grounded training trajectories
be used to test whether weight adaptation adds value.
