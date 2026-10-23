# ADR 001: Local-first grounded baseline before specialization

Date: October 23, 2026
Status: initial implementation decision for a bounded local construction milestone.

## Decisions

Use one application platform with Acadia, Katahdin and Atlantic profiles. Separate
their source namespaces, policy descriptions and evaluation cases. Start with
Pleias-RAG-1B for quality and the matching 350M model for a low-cost runtime check.
One successful small checkpoint may serve both the Android phone and a shared
hosted API. A larger or second engine must earn its cost with evaluation.

The local phone target is Samsung S26; exact variant, RAM, Android
version, sustained performance and runtime compatibility remain to be verified.
llama.cpp's raw completion path is the first implementation route because Pleias
publishes GGUF. LiteRT conversion remains a separate option, not an assumed
drop-in replacement. No generic chat template is used for the Pleias protocol.

Keep authorization outside the weights. This milestone has no external tools,
remote callbacks, cloud fallback, browser, messaging or mutation permissions.
Natural-language output never authorizes an action. Production tenant isolation,
credentials, approvals and storage controls are later application/service work,
not claims made by this namespace check.

Prefer open/public-domain lineage and document permitted synthetic/non-public
sources. Client information is not available for training without explicit
permission. A publisher's weight license does not establish full rights clearance
for all inherited data.

## Implementation approach

Investigate implementation defects, make feasible corrections, and rerun the
relevant checks. Document external constraints, unresolved evidence and unverified
quality alongside the results. Verification must preserve access controls and
the defined application boundaries.

## Resource and delivery constraints

Compare useful throughput, memory use and end-to-end cost before selecting
training or hosting infrastructure. GPU serving remains an option when
measurements justify it.

Begin runtime testing with one bounded case after checking available memory.
The 24-case suite should run only on a host with sufficient capacity. Runtime and
model provisioning, including quantization, are separate from harness execution.

November 15, 2026 is the target for an operable preliminary Android/hosted pilot.
This milestone adapts existing weights rather than training three foundation
models or delivering five polished native apps. Katahdin examples remain
provisional, synthetic, and non-predictive pending domain review.

## Acceptance and remaining gates

1. Confirm exact model/tokenizer framing, CPU inference and S26 runtime behavior.
2. Measure baseline citation accuracy, abstention, task results, memory and latency.
3. Select client-service/domain workflows and a separately reviewed held-out set.
4. Review complete source/permission manifests and synthetic generator terms.
5. Approve compute only after the architecture and first feasibility result.
6. Compare unchanged, retrieval-only and trained variants; preserve source splits.
7. Validate quantization, privacy, action controls and rollback behavior.
8. Approve public distribution, model/data cards, insurance/IP and country/service
   policy before customer-facing release. Legal review is managed separately.

No code test result establishes model safety, ecological forecasting validity,
absence of hallucinations, or production readiness. No source-level rights audit
of the entire inherited pretraining corpus has been completed here.

## Primary references

- https://huggingface.co/PleIAs/Pleias-RAG-1B
- https://huggingface.co/PleIAs/Pleias-RAG-350M
- https://huggingface.co/PleIAs/Pleias-1.2b-Preview
- https://arxiv.org/html/2504.18225v2
- https://github.com/Pleias/Pleias-RAG-Library
- https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://huggingface.co/litert-community/SmolLM3-3B
