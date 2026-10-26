# Model comparison — October 26, 2026

**Decision: neither initial Pleias candidate is approved for the Android alpha.**
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
| SmolLM3 3B original BF16 | Reference probe prepared and dispatched; full output review pending. | Unselected. |

The 1B return-policy answer stated the correct 30-day window but cited the
separate refund-review threshold. The 1B forestry answer largely followed the
provided survival result, with terminology drift. The local-notes answer in the
16-bit GGUF invented implementation details beyond the one-sentence codename
note. All inputs were fictional; no client or personal data was used.

Evidence: [original-checkpoint run](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40318797250),
[1B precision comparisons](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40318614579),
and [readable 16-bit evidence](https://github.com/hayvenmjackson/krummholz-llms-alpha/actions/runs/40319795509).
Review was performed by development agents against complete generated outputs.
It is not a claim of independent human review or physical S26 testing.

## Next comparison and release gates

The unchanged SmolLM3 checkpoint is pinned in `config/smollm3-reference.json`.
Its seven development probes include supported source questions, missing evidence,
source injection and two inert tool proposals. The program does not execute tools
or approve a model. The separate 36-case pilot is excluded from training.

A promising candidate must retain its useful behavior through the selected native
runtime and pass broader answer-level evaluation. Its exact model bytes, prompt
template, tokenizer, outputs, citations, memory use and latency must be recorded.
Signed Android installation and real inference tests on 4 KB and 16 KB emulators
follow integration; the S26 remains a separate physical-device test.

No correct-number check, literal quotation match, successful build or model's own
confidence can establish those results on its own. Weight adaptation and tool
execution are subsequent measured increments, as described in ADR 002.
