# ADR 005: Krummholz Library (reference corpus and daily current channel)

Date: October 27, 2026. Status: accepted direction (Hayven, October 27, 2026). No code yet.
Builds on ADR-003 (basis ranking, "how do I know this?"), ADR-004 (dynamic weights layers,
hallucination policy) and pass 3 (certainty and search, fetch, cite tool use).

## 1. Decision

Ship a signed, versioned, dated knowledge library with every Krummholz model and query it
through a read-only tool, instead of relying on what the weights remember. Two parts:

- **Reference library**: slow-moving, curated passages (history, science, law, geography,
  product domain packs). Updated in versioned releases.
- **Current channel**: structured, dated fact cards about the state of the world, rebuilt
  centrally once a day and shipped as signed deltas to every connected instance.

Facts still never go into weights (ADR-004). The library is the shared, Krummholz-curated
counterpart to each user's personal memory store.

Why: answers grounded in cited library text cannot revert to the publisher base model's
political alignment; citations become exact and checkable in code; knowledge updates
without retraining; the corpus can be audited; and a 4B model with the library should beat
an 8B model without it on factual questions, keeping the phone tier small. It also removes
the fixed "knowledge cutoff": the model reports facts as of the library's date.

Not chosen: hard-coding the 1,595 verified history answers as canned replies. Exact-match
lookup misses paraphrases and a wrong canned answer is a confident confabulation. The
passages are the ground truth; the verified questions become index hints that point to them.

## 2. Lookup order and basis

At answer time the model may consult, in order of preference by basis (ADR-003):
fresh live web result (when connected and the question is time-sensitive), current-channel
card (dated), reference library passage (dated release), user statement or personal memory,
then the weights. Every answer cites which one it used. The system prompt always carries
today's date and the library's as-of date, and the model is trained that a dated library
fact beats its own prior memory.

Tools (krummholz-readonly-tools, next version):
- `search_library {query}` returns non-citable hits with ids, titles and as-of dates.
- `fetch_library {result_id}` returns citable text or a fact card.
- `search_public_web` / `fetch_public_page` remain for anything the library lacks or for
  facts that change faster than daily (prices, weather, breaking events).

Certainty: a strong library hit raises certainty only when the fetched text actually
contains the answer (exact-quote citation). A weak or missing hit lowers it and escalates to
web search when connected, or to an honest "my library does not cover this" offline.

## 3. Current channel: fact cards, not articles

Each daily item is a structured claim, never narrative prose:

```json
{"card_id": "...", "as_of": "2026-10-27", "event_date": "2026-10-26",
 "subject": "...", "predicate": "...", "object": "...", "numbers": {"...": 0},
 "statement_by": null, "status": "confirmed | reported | superseded",
 "supersedes": null, "sources": [{"url": "...", "publisher": "...", "tier": "A", "fetched_at": "...", "sha256": "..."}],
 "quote": "short exact quote from a primary source, when licensable"}
```

Rules, enforced in code:
- **Facts and attributed statements only.** "The central bank set its rate at X on [date]";
  "Ministry Y stated Z". No characterizations, adjectives of judgment or motives.
- **Every number, date and name** in a card must appear in at least one of its sources
  (the same deterministic check as ADR-004 section 6).
- **Confirmed** requires at least two independent sources that agree on the specifics, or
  one Tier A primary source. Otherwise the card is **reported**, and the model must say
  "reported by X".
- **Supersession** links newer cards to the ones they replace (revised counts, changed
  office-holders), using ADR-003 conflict handling.
- Cards are small (about 50 to 150 tokens), so retrieval adds little prompt cost.

## 4. Source policy (decided by verifiability and independence, not reputation)

The same rule applies to every outlet and every government, including the United States.

**Tier A: primary and official records** (preferred; can confirm alone)
- Election results from national election authorities; legislation and gazettes
  (e.g. Congress.gov, legislation.gov.uk, EUR-Lex); court judgments from the issuing court.
- Central bank decisions; national statistics offices; World Bank, OECD, IMF, Eurostat and
  UN statistical releases.
- Hazards and health: USGS earthquakes, NOAA and national meteorological services, GDACS,
  WHO disease outbreak notices.
- Corporate and legal registries: SEC EDGAR and equivalent national filings; UN Security
  Council resolutions; official sanctions lists.

**Tier B: structured world state** (corroborating only; never the sole source for anything
contested)
- Wikidata (CC0 structured statements such as office-holders, populations, dates), used only
  for statements that carry a reference, and checked against Tier A where one exists.
  Wikipedia prose is not used in the current channel.

**Tier C: reporting** (for events with no primary record yet; facts extracted, text not
redistributed)
- A diverse pool of news organizations across countries, ownership groups and editorial
  traditions, including the major international wire services. Opinion, analysis and
  editorial sections are excluded.
- **Independence rule**: two reports count as independent only if they come from different
  ownership groups and neither cites the other as its source.
- **State media rule**: outlets controlled by a government (any government) count only as
  that government's statement ("the Chinese government stated", "the Russian government
  stated", "the U.S. government stated"), never as independent confirmation of facts about
  that government's own conduct.
- Social media is excluded except official accounts, recorded as statements by their owner.

The source registry (publisher, country, ownership group, state control, tier, license,
access terms) is a versioned config file. Changes to it are reviewed like code, with a reason
recorded for each, so the policy itself is auditable.

**Licensing**: facts are not copyrightable; cards store facts, links and only short quotes.
Full article text is never redistributed. Wire-service and publisher terms for automated
access are on counsel's list before Tier C goes live. Tier A and Wikidata are public domain,
CC0 or open-licensed.

## 5. Daily pipeline (central, unattended)

1. Fetch from the registry (respecting robots and terms), snapshot each document with hash
   and time.
2. Extract candidate cards: deterministic parsers for structured sources; an open-weight
   model (OLMo 3) for text, with every field verified against the source text in code.
3. Corroborate, deduplicate, supersede.
4. Lint: neutrality (no judgment adjectives or unattributed claims), schema, numbers and names.
5. Anomaly checks: a card contradicting many confirmed cards, a sudden surge on one topic,
   a single source producing an unusual share of the day, or any registry source failing
   signature or format checks.
6. Build the delta, sign it, publish. Instances download on connection (phones on Wi-Fi and
   charging), verify the signature, update the index incrementally.

**No daily human review.** The deterministic gates do the work. Cards that trip an anomaly
check are **held** while the rest of the release ships; held cards wait for a person (Hayven,
or a delegate later) and expire after 14 days if unreviewed. A weekly 20-card random audit
is optional and is the input to an RLHF-style correction stage in later training runs.

Estimated cost: a few CPU hours and a small GPU extraction job, about 1 to 5 dollars per day,
plus delivery. Deltas are expected to be a few megabytes per day.

## 6. Reference library

- Initial content: the 1,836 history passages already snapshotted with revision ids
  (`config/history-sources-v*.json`), public-domain government and science texts
  (`config/public-domain-text-sources.json`), and product packs (forestry and fisheries for
  Katahdin; client policies stay in organization documents for Acadia, ADR-004 L3).
- Size tiers: desktop and hosted up to about 1 GB of text with index; phone 50 to 200 MB with
  per-product packs.
- Curation follows the same neutrality rule: contested topics carry passages presenting each
  side's evidence, with attribution.
- Licensing: CC BY-SA passages ship as a separately licensed data file with attribution,
  which is simpler than training on them (ADR-003 open question).

## 7. Integrity and safety

- Releases and deltas are signed; instances refuse unsigned or mismatched data; rollback is
  one version back.
- Library text is untrusted data like any tool result: instructions inside it are never
  followed (existing injection training applies).
- Instances that cannot connect keep their last release and state its as-of date.
- Runtime guard (host code): if an answer contains phrasing from the propaganda phrase lists
  and no library or web citation, the host triggers a lookup instead of showing it.

## 8. Training implications (next run)

- Trajectories for `search_library` / `fetch_library` alongside web search, in the pass 3
  format, including: library newer than the model's memory (library wins), library silent
  (search the web or say so), library card status "reported" (attribute it), and stale card
  versus fresh web result (web wins).
- The certainty labels keep using measured accuracy; library-grounded answers should push
  most factual answers above the threshold without guessing.
- Evaluation: library hit rate and answer accuracy on probes v1/v2 with and without the
  library; latency and prompt cost on the S26.

## 9. Open items

- Counsel: Tier C publisher and wire terms; CC BY-SA library file attribution format.
- Build order: reference library and `search_library` first (no new sources needed), then
  the current channel with Tier A and B, then Tier C after counsel.
- Embedding model choice for on-device retrieval (small, permissively licensed), plus BM25.
