# Public-domain scientific text intake — candidate register

Prepared 26 October 2026. **Original proposal: eight candidates; no training or redistribution approval.** The machine-readable register is [public-domain-text-sources.json](../config/public-domain-text-sources.json). The subsequent [bounded acquisition and cleaning audit](public-domain-pilot-audit.md) records the six acquired CC0 documents and selected passages. The original register remains a proposal snapshot with approval flags false; actual reviewed selections are recorded separately.

The next useful step is a small, audited evidence set for the behavior in [ADR-002](ADR-002-evidence-seeking-training.md). It does not replace the pending model-quality baseline or establish that the inherited model's corpus meets our provenance policy.

## Candidate documents

| Register ID | Scientific use | Rights evidence and present status | Provisional split |
| --- | --- | --- | --- |
| usgs-roadxstr-2026 | Road/stream crossing field methods for forestry and fisheries. | [USGS item PDF](https://pubs.usgs.gov/tm/18/b1/tm18B1.pdf): indexed frontmatter gives a U.S. public-domain notice with third-party exclusions. Direct fetch returned 403; hold for full review. | Train |
| usgs-landcover-2015 | Historical land-cover analysis; retain the 1973–2000 period. | [USGS item PDF](https://pubs.usgs.gov/of/2015/1018/pdf/ofr2015-1018.pdf): same qualified notice; direct fetch returned 403. Hold for full review. | Test |
| noaa-stock-reference-points-2025 | Fisheries reference-point and assessment methods. | [NOAA item record](https://repository.library.noaa.gov/view/noaa/72013) explicitly identifies CC0; section and privacy review pending. | Train |
| noaa-modeling-uncertainty-2015 | Model uncertainty, skill and communication. | [NOAA item record](https://repository.library.noaa.gov/view/noaa/12956) explicitly identifies CC0; section and privacy review pending. | Dev |
| noaa-water-level-uncertainty-2025 | Instrument, survey and geodetic uncertainty. | [NOAA item record](https://repository.library.noaa.gov/view/noaa/70293) explicitly identifies CC0; section and privacy review pending. | Train |
| noaa-nwlon-requirements-2024 | Observing-system accuracy, continuity and documentation. | [NOAA item record](https://repository.library.noaa.gov/view/noaa/66446) explicitly identifies CC0; section and privacy review pending. | Train |
| noaa-doppler-principles-1983 | Stable radar physics and measurement limitations only. | [NOAA item record](https://repository.library.noaa.gov/view/noaa/49662) explicitly identifies CC0; obsolete operational instructions excluded. | Train |
| plos-metabarcoding-signal-noise-2023 | Sampling bias, non-detection and ecological evidence. | [This article's publisher notice](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0285674) explicitly specifies CC0, corroborated by [NOAA](https://repository.library.noaa.gov/view/noaa/53260). Section and privacy review pending. | Test |

Exact titles, dates, publishers, document URLs, scope and exclusions are in the register. These are candidate technical texts, not eight cleared datasets. The two water-level reports share one source family and must stay together.

## Rights boundary

[NOAA's policy](https://repository.library.noaa.gov/Content%20and%20Copyright) distinguishes public access from permission to reproduce or reuse; unlabelled documents must not be assumed reusable. A repository label does not automatically clear a credited quotation, image, table or external dataset.

[CC0](https://creativecommons.org/publicdomain/zero/1.0/) is a worldwide dedication to the extent law permits; it does not remove privacy, publicity, patent or trademark rights. The two USGS notices establish only a candidate U.S. government-work basis, not a worldwide dedication. Their proposed use outside the U.S. remains a separate review gate. No CC-BY material is included: an attribution license is not public domain. Preserve attribution and evidence even where a license does not require it, and never suggest agency endorsement.

## Bounded intake and cleaning

Start with at most eight documents and 150,000 retained words; actual counts are unknown. Prefer the six explicit CC0 items for the first review. Hold the two forestry items until direct source/credit inspection succeeds. This is a focused adaptation/evaluation pilot, not enough material to justify training a foundation model from scratch.

1. Snapshot each exact document and item-level rights notice, with edition, retrieval time, final URL, bytes and SHA256. Approve page/section ranges and record reviewer and jurisdiction scope before extraction enters the corpus.
2. Extract narrative and useful equations with page/section offsets. Verify OCR, signs, units, subscripts and table structure against the source. Drop unreadable passages rather than inventing missing symbols.
3. Remove site navigation, repeated headers, contact blocks, acknowledgements and bibliography text. Exclude third-party material without separate clearance, logos/images, linked genetic records, partner datasets and supplements. Keep bibliographic attribution in the provenance manifest.
4. Screen names in narrative, email/phone/address strings, account-like identifiers, author birth years/ORCIDs, participant lists, vessel/operator records and sensitive site/property locations. Manually inspect flagged passages and a retained sample; regex alone is not a privacy audit. Remove or quarantine questionable examples while preserving needed scientific context.
5. Normalize Unicode/whitespace for deduplication; retain an original-to-clean offset map. Use exact normalized hashes plus near-duplicate word shingles across this intake and the existing corpus. Quarantine shared passages and credited quotations; do not let a paraphrase evade a source-family split.

No private email, client material or laptop/phone personal data is part of this proposal. A scientific PDF can still contain personal information; public availability is not a reason to retain it.

## Split and evaluation plan

Freeze source-family assignments before generating examples: five train documents, one dev document and two test documents, across seven families. Keep revisions, translations, copied sections, derived questions and all synthetic variants with their originating family. Near-duplicate families must be merged before the freeze; a held document contributes nothing until cleared. These are provisional assignments, not finished evaluations.

Dev/test text stays out of training, training retrieval indexes and teacher generation for training. Use a sealed set of human-written questions and source-backed answers; record question author and when a result was exposed. This small split cannot support broad capability claims. Base-model exposure to these publications is unknown, so “unseen during our adaptation” must not be described as “never seen in pretraining.”

Measure answer correctness, exact numerical/unit handling, citation support, recognition of insufficient evidence, appropriate fresh lookup and correct tool permissions. Include stale editions, conflicting passages, missing evidence, invalid units and malicious instructions embedded in retrieved text. Compare unchanged base, retrieval-only and any later adapter on the same frozen tests. No quality threshold is asserted from three smoke cases.

## Text versus tool trajectories

Use cleared narrative to teach terminology and explanation style. Treat time-dependent facts—current forecasts, stock status, catch limits, operational requirements—as prompts to retrieve current authoritative evidence, not facts to memorize from old publications.

Build the first **40 reviewed examples** before expanding to a maximum of **300 trajectory records**. Each should retain source ID/hash/section, partition, date, question, evidence/tool results, answer, citation and reviewer result. Synthetic wording and simulated tool results must be labelled. Verify facts against approved text and calculations against deterministic code; do not accept a teacher model's assertion as evidence. Track the teacher's permission to generate training data separately.

Useful trajectories: identify missing information; select an allowed read-only search; assess date and authority; cite the retrieved passage; call a calculator for an explicit quantitative step; answer with stated limits or ask for missing evidence. Balance useful answers with justified lookup and abstention so “use tools” does not become unnecessary searching or blanket refusal. Include hostile source text as data, never as authorization.

Training a tool-call proposal does not grant execution rights. Runtime permission checks remain outside the model; the current Android baseline stays offline with external tools denied. Live web access, tool schemas and an action broker require their own implementation and verification. No lookup or downstream action is executed by this intake proposal.

