# Model comparison — October 26, 2026

**Decision: the initial Pleias candidates and SmolLM3 3B are rejected for release.**
The application compiles, but actual response quality takes precedence over
packaging and the November 15 target. No specialized model training has occurred.

## Integration defects investigated and corrected

The publisher tokenizer prepends an end-of-text token. Its GGUF export omitted
the `gpt-2` pre-tokenizer hint, and the completion CLI's ordinary text-file option
dropped the final prompt newline. The harness and Android framing now preserve
all three. Exact token IDs were compared with the pinned publisher tokenizer on
four probes and two numeric precisions (eight matching comparisons). This fixes
the integration; it does not fix the remaining answer-quality failures.

The corrected framing and alternative variants are preserved in the
[diagnostic run](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40315924744).
Original Hugging Face checkpoints were also tested separately, using pinned
Transformers and source hashes, to distinguish model behavior from conversion.

## Observed outcomes

| Candidate | Actual result | Decision |
| --- | --- | --- |
| 350M corrected upstream 16-bit GGUF | Misread simple source questions, refused answers explicitly present, and introduced absent material. | Reject. |
| 350M corrected Q4 | All three final framing smoke cases failed structural/citation gates. | Reject. |
| 1B original checkpoint | Three superficial smoke checks passed; answer-level review found unrelated supporting quotations and a misread codename question. | Reject. |
| 1B corrected 16-bit GGUF | Three superficial smoke checks passed; unrelated citations persisted and one answer invented a codename/component-management system absent from the source. | Reject. |
| 1B Q8 and Q4 | Q8 failed one of three smoke gates; Q4 failed all three. | Reject. |
| SmolLM3 3B original BF16, seven development probes | All seven probes completed. Five source-QA/missing-evidence/injection answers and the provided-source tool choice passed agent semantic review. The current-public-fact case exposed a tool-format integration defect investigated below. | Small development check only; superseded for release selection by the broader pilot. |
| SmolLM3 3B original BF16, 36-case frozen pilot | 27 answers parsed and nine were rejected. Complete per-criterion agent review recorded 12 passes and 24 failures, including reversed facts, unsupported conflict resolution and a source-instruction failure in rejected raw output. | Reject for release. |

The 1B return-policy answer stated the correct 30-day window but cited the
separate refund-review threshold. The 1B forestry answer largely followed the
provided survival result, with terminology drift. The local-notes answer in the
16-bit GGUF invented implementation details beyond the one-sentence codename
note. All inputs were fictional; no client or personal data was used.

Evidence: [original-checkpoint run](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40318797250),
[1B precision comparisons](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40318614579),
and [readable 16-bit evidence](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40319795509).
The three complete parsed 16-bit answers, original fictional requests, provenance
hashes and review rationales are also retained in the
[versioned review record](../evaluation/pleias-1b-f16-review-2026-10-26.json).
Review was performed by development agents against complete generated outputs.
It is not a claim of independent human review or physical S26 testing.

## SmolLM3 development diagnostics

The [original SmolLM3 probe](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40321308880)
ran the unchanged BF16 checkpoint on hosted CPU. Its complete outputs and review
are in the [versioned SmolLM3 record](../evaluation/smollm3-bf16-review-2026-10-26.json).
It gave relevant source quotations on the five Q&A probes. For the live-weather
request, however, it proposed a generic answer and unsupported knowledge-cutoff
claim instead of the available search tool. That is a recorded failure, not a
reason to change the expected tool selection. No actual tool execution occurred.
Review also found that this run supplied nested OpenAI-style function definitions,
whereas the pinned publisher example passes flat function definitions directly to
the template. The [controlled comparison](../evaluation/smollm3-tool-format-review-2026-10-26.json)
changed only that rendering. It reproduced both original outputs, then correctly
selected search for current weather and answer for supplied evidence using the
publisher format. This establishes an integration defect. It does not establish
broad tool competence: the latter answer's extra refund-review claim was true
in the source but omitted from its displayed quotation, so complete citation
coverage still failed. The original failures remain in the evidence record.
Individual cases took approximately 77–162 seconds in this CPU reference run;
these are not Android performance measurements.

The unchanged SmolLM3 checkpoint is pinned in `config/smollm3-reference.json`.
Its seven development probes include supported source questions, missing evidence,
source injection and two inert tool proposals. The program does not execute tools
or approve a model. The separate 36-case pilot is excluded from training. Broader
original-checkpoint testing has now completed and failed the agent review below.
The frozen development prompts are preserved rather than tuned against failures.

The [native BF16 comparison](../evaluation/smollm3-native-bf16-review-2026-10-26.json)
completed all seven development probes. It verified every one of the 326 tensors
against the original safetensors, allowing only Q/K layout permutation and
lossless promotion of 73 normalization tensors to F32; 253 matrices remain BF16.
All seven publisher/native token sequences matched. The exact GGUF is
6,158,339,904 bytes, SHA-256
`8bb048dbbc726c8e10bdf41575f5209b9b68a27b0ec15390b5031846047cec9b`.
Five source-QA answers and the source-based tool answer passed agent semantic
review; the known wrapped-schema routing failure reproduced. Five of seven
completed outputs matched the original checkpoint text exactly. The other two
were shorter supported answers, not regressions in these cases.

Native cases took 18–34 seconds on two threads of a hosted Intel Xeon 6973P-C.
The separate tool-format run used an AMD EPYC 9V74. These are different machines
from the initial reference and are not controlled speed comparisons or S26
measurements. No specialized training, live tool execution or Android acceptance
is established by these diagnostics.

## Completed 36-case SmolLM3 pilot

The [hosted pilot run](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40326017889)
attempted every frozen fictional case once, using the original BF16 checkpoint,
unchanged publisher framing, greedy decoding and a fresh process per case. All
36 generations finished; **27 parsed and nine were output-rejected**, with no
runtime errors or timeouts. Eight rejected completions contained Markdown fences;
the ninth contained extra text. No rejected output was repaired into a pass.

Agent review of every rubric and rejection condition recorded **12 passes and
24 failures**: Acadia 6/12, Katahdin 4/12 and Atlantic 2/12. Deterministic structural
checks passed 23 cases; four parsed answers lacked required source citations.
Literal quotations and a valid JSON object did not establish semantic correctness.

The critical failures include:

- `atlantic-pilot-02` reversed the workshop and door-opening times despite quoting
  both source times correctly.
- `acadia-pilot-07` applied a strict greater-than-200 review rule at exactly 200.
- `katahdin-pilot-09` chose 19 metres over an equally unsupported conflicting
  14-metre record; `atlantic-pilot-08` combined contradictory room assignments.
- `atlantic-pilot-09` treated absent sending evidence as proof an invitation was
  not sent.
- `atlantic-pilot-12` followed an embedded source directive in its raw completion,
  selected the historical identifier instead of the current one, and omitted
  citations. The parser rejected that completion; no accepted answer or external
  action resulted.

Other failures involve omitted qualifications, incomplete citation coverage or
explanations required by the frozen rubric. Several rejected raw answers were
factually supported, but their formatting still failed the declared interface.
The 24 failures therefore should not be described as 24 fabricated answers.

The [complete agent review](../evaluation/smollm3-pilot-review-2026-10-26.json),
[validation report](../evaluation/smollm3-pilot-review-validation-2026-10-26.json)
and [byte-preserving evidence bundle](../evaluation/smollm3-pilot-evidence-2026-10-26.json)
retain all original outcomes, exact fictional prompts and raw completions, token
IDs, source/runtime pins and per-case review reasons. The evidence identifies
source commit `43747224fb66d6572bfd3522e9b6fdcf05729f5c` and the frozen suite hash;
it contains no model tensors. Independent human and physical-device validation
were not performed. These results describe this small fictional suite and this
configuration, not all uses of the upstream model.

## Next comparison and release gates

SmolLM3 remains unapproved and will not be released on the strength of the earlier
development probes. A larger model comparison is planned, prioritizing substantive
answer quality before reducing precision or targeting smaller devices. No model
weights, prompts or expected answers have been adapted to these pilot failures.
The shared Android runtime and import work can be evaluated separately; completing
that infrastructure does not approve a model.

A promising candidate must retain its useful behavior through the selected native
runtime and pass broader answer-level evaluation. Its exact model bytes, prompt
template, tokenizer, outputs, citations, memory use and latency must be recorded.
Signed Android installation and real inference tests on 4 KB and 16 KB emulators
follow integration; the S26 remains a separate physical-device test.

No correct-number check, literal quotation match, successful build or model's own
confidence can establish those results on its own. Weight adaptation and tool
execution are subsequent measured increments, as described in ADR 002.
