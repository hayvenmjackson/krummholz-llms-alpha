# ADR 002: Train for evidence seeking and grounded tool use

Date: October 26, 2026.
Status: accepted direction; data preparation and implementation are staged behind
baseline validation. No specialized weights have been trained by this decision.

## Intended behavior

Krummholz models should seek evidence when a question needs current information,
missing facts, an exact calculation, or an external record. They should choose an
appropriate available tool, use its actual result, and cite the evidence supporting
the answer. A model should still answer directly when the supplied text already
contains sufficient evidence, or when no external factual claim is needed.

Effectiveness determines model selection. Low-end hardware compatibility is not a
release requirement. Compression is evaluated against the original checkpoint;
smaller artifacts are accepted only when the relevant quality is preserved.

## Application and model responsibilities

Weights learn tool selection, argument construction, source assessment and grounded
responses. Application code provides the tools, validates requests, enforces access
and spending limits, and decides whether required evidence has been obtained.
A preference learned by a model is not a reliable enforcement mechanism by itself.

The first tool-enabled increment will offer read-only web search, source retrieval,
local document retrieval and bounded calculations. Tool proposals use a versioned
schema with a known tool name and validated arguments. They do not execute directly.
The host rejects unknown tools, extra fields, unauthorized resources and requests
outside the caller's scope. Sending messages, purchases and file changes remain
separate capabilities with their own authorization requirements.

For requests explicitly requiring current facts, the host requires a successful
fresh retrieval before presenting an answer as verified. It records source URLs,
retrieval time and the evidence used. Failed or unavailable searches lead to an
honest unresolved result, never an invented search result. Offline mode can answer
from supplied documents and clearly identifies when a request needs live access.

Search results and retrieved pages are untrusted content. The host separates them
from instructions, limits content size, rejects private-network retrieval targets,
checks redirects, and bounds tool calls and total time. Citations must refer to
actual retrieved material. Literal quotation checks are combined with semantic
review; a matching quote alone does not establish the answer is correct.

The current first-alpha implementation is an offline source-Q&A baseline. Its
manifest has no network permission and its protocol has no executable tool calls.
This decision does not pretend those capabilities already exist or silently add
network access to that release. The next implementation introduces explicit tool
interfaces and evaluates the complete loop before advertising tool-enabled models.

## Training data and sequence

1. Establish a capable unchanged baseline and rule out tokenizer, conversion and
   runtime defects. Compare candidates on unseen source-Q&A and tool-use tasks.
2. Assemble a small, documented corpus of relevant real-world public-domain text.
   Keep item-level rights evidence, original URL, dates, content hash, extraction
   method and exclusions. Publicly readable material is not automatically public
   domain. Record jurisdiction and distinguish CC0 from other open licenses.
3. Remove boilerplate, exact/near duplicates, personal contact details, sensitive
   personal content and unrelated material. Preserve provenance and required
   attribution separately from training examples. Split by original document and
   source family before chunking so near-duplicate passages cannot cross evaluation
   and training boundaries.
4. Create reviewed demonstrations of tool choice, valid arguments, real tool
   observations, evidence-backed answers, clarification and abstention. Keep
   source text and synthetic demonstrations separately labeled. Synthetic examples
   require factual checks and recorded generator terms; they are not relabeled as
   historical or observational data.
5. Run a bounded supervised adapter-training experiment for the selected base.
   Measure whether domain-text adaptation adds value before expanding it. Then use
   preference pairs where supported to favor verified evidence over confident
   guessing, correct tool use over fabricated calls, and appropriate direct answers
   over needless searching. Do not assume more tokens or more training improves
   quality without comparison.
6. Compare the unchanged base, retrieval/tool-assisted base and trained candidate
   on the same held-out tasks. Retain the base and previous adapter for rollback.
   Ship training changes only when the overall result improves without material
   regression in general competence or task boundaries.

## Evidence required before a tool-enabled release

- Correct tool selection and arguments, successful execution, and faithful use of
  results across fresh facts, calculations and document questions.
- Answers supported by sources, accurate citations, and appropriate treatment of
  missing or conflicting evidence. Review full claims, not keyword presence.
- Resistance to instructions embedded in retrieved content, including attempts to
  redirect tools or disclose unrelated information.
- Honest behavior on offline use, search failures, stale sources and exhausted
  time/cost budgets; no fabricated tool execution or source.
- Measured latency, memory, request cost and regression results on the intended
  deployment. S26 testing is evidence about that device, not a substitute for
  broader model evaluation.

The authored pilot evaluation is kept out of training and reviewed at the answer
level. A small passing suite is useful development evidence, not proof of broad
production reliability. A known materially incorrect candidate is not published
as a usable model simply to meet a packaging or schedule milestone.

## Primary references

- Toolformer demonstrates training models to select and use tools:
  https://arxiv.org/abs/2302.04761
- USGS explains U.S. public-domain material and third-party exceptions:
  https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted
- NOAA National Ocean Service describes reuse, credits and exceptions:
  https://oceanservice.noaa.gov/about/faq.html
- Project Gutenberg distinguishes public-domain text, copyrighted contributions
  and jurisdiction-specific status:
  https://www.gutenberg.org/policy/permission.html
