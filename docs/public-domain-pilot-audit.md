# Public-domain text pilot: acquisition and cleaning audit

Completed 26 October 2026. **Six full PDFs acquired; 30 selected passages retained from five documents; one document held. No training, synthetic generation or corpus publication occurred.** This is a small reviewed intake batch, not a sufficient training corpus or whole-document rights clearance.

The six PDFs total **12,092,086 bytes and 250 pages**. Every PDF matches the checksum published in its NOAA repository record. Primary item-level CC0 notices were saved with exact requested/final URLs, timestamps, byte counts and SHA256 hashes. The PLOS PDF came from its publisher and matches the NOAA copy's SHA256. The two USGS candidates were outside this six-item acquisition and remain on hold.

| Source ID | Partition | Retained passages | Retained words | Decision |
| --- | --- | ---: | ---: | --- |
| noaa-stock-reference-points-2025 | Train | 6 | 565 | Selected methods prose |
| noaa-water-level-uncertainty-2025 | Train | 7 | 829 | Selected methods and limitations |
| noaa-nwlon-requirements-2024 | Train | 8 | 475 | Selected requirements concepts |
| noaa-modeling-uncertainty-2015 | Dev | 5 | 412 | Reserved for development |
| plos-metabarcoding-signal-noise-2023 | Test | 4 | 500 | Reserved for testing |
| noaa-doppler-principles-1983 | Train | 0 | 0 | Held: unreliable OCR and old operational context |

**Totals:** 2,781 words; train 1,869, dev 412, test 500. Source families retain their proposed partitions; both water-level reports remain together. Only the `text` fields in the train partition are training candidates. Metadata and held-out text must not be included in training or training example generation.

## What was removed or held

The raw extraction contains 79,124 whitespace-delimited words. Only explicit, reviewed line intervals enter the cleaned files. Everything outside those intervals is excluded: author/contact and affiliation blocks, acknowledgements, participant lists, correspondence, references, figures, tables, front/back matter, supplements, linked datasets, and quoted third-party passages. Bibliographic provenance remains separate from the training text.

Specific exclusions include the water-level report's quoted IHO definitions, the uncertainty workshop's credited third-party table on PDF page 22, and the radar manual's diagram credited to another author on PDF page 7. The radar PDF also contains damaged OCR and descriptions of obsolete operational equipment; it contributes no text. No exact addresses, emails, phone numbers, ORCIDs, named participants, vessel/operator records or precise field-site coordinates were retained in the selected prose.

The initial pypdf plain-text extraction introduced broken spaces and replacement characters. Switching to pinned pdfplumber extraction corrected those issues in the modern documents. A small list of visually reviewed line-wrap repairs is recorded explicitly, along with exact citation removals. Source-span hashes prevent these edits from silently applying to changed text. Equations, subscript-heavy passages and broken cross-page continuations were excluded instead of reconstructed.

All 30 final passages were read. Seven source pages were rendered and visually inspected, including one page per retained document and two radar pages. Pattern checks found no remaining contact, URL, rights-credit or replacement-character flags in retained text. Exact hashes and five-word-shingle containment checks found no duplicate selections. These checks cover this selected pilot, not every document in the existing corpus or the inherited model's training data. They are not a guarantee that no privacy or rights issue could remain.

## Small excerpt spot review

These short excerpts identify the reviewed material without exposing excluded details:

| Source and PDF page | Excerpt | Check |
| --- | --- | --- |
| Stock-reference guidance, 33 | “Process error can account for the natural variability of stock biomass” | Complete paragraph retained; citation names removed; scientific qualification preserved. |
| Water-level uncertainty, 17 | “Metadata should include: sensor type and brand/model” | Generic metadata requirements, no actual person or station record. |
| NWLON requirements, 10 | “Requirements are outcome-oriented, concise, actionable, customer-oriented” | Retained general design principles; statutory text and footnotes excluded. |
| Modeling uncertainty, 16 | “Evaluation of model skill has multiple facets.” | Dev-only passage; removed bibliography pointers remain recoverable through source offsets. |
| Metabarcoding, 13 | “Our results illustrate that noise in metabarcoding datasets, like signal, is non-random” | Test-only passage; line-wrap repairs checked against the rendered page. |

## Reproduction and local artifacts

Implementation: [intake_public_domain_pilot.py](../tools/intake_public_domain_pilot.py). Exact reviewed selections and exclusions: [public-domain-pilot-review.json](../config/public-domain-pilot-review.json). Candidate policy and source URLs: [public-domain-text-sources.json](../config/public-domain-text-sources.json).

Verified environment: Python 3.12.14, pypdf 6.10.0, pdfplumber 0.11.9 and Node 24.19.0. Node's standard HTTPS fetch was needed because urllib received HTTP 403 from NOAA; no cookies, account credentials, authentication bypass or weakened TLS checks were used. No GitHub acquisition job was created.

```text
python -B tools/intake_public_domain_pilot.py --self-test
python -B tools/intake_public_domain_pilot.py --fetch
python -B tools/intake_public_domain_pilot.py --clean
```

All acquisition and corpus files remain under the gitignored `data/public-domain-pilot/` directory:

- `acquisition.json`: full-text and rights-evidence receipts, hashes and official checksum checks.
- `raw/` and `evidence/`: original PDFs and primary rights pages; not cleaned for publication.
- `extracted/`: numbered page/line text for review; not training input.
- `clean/train.jsonl`, `clean/dev.jsonl`, `clean/test.jsonl`: reviewed passages, separate by source family.
- `manifest.json`: machine-readable counts, source-span provenance, review decisions, exclusions and output hashes.
- `review-images/`: local visual checks; not publication assets.

The self-test passed. Cleaning twice produced identical hashes for all three JSONL files. An intentionally altered source hash was rejected before changing any output. All six repository checksum checks passed, and Git confirmed raw, evidence and clean files are ignored. No private emails or client material was accessed.

The coordinating agent separately read all 30 selected passages for semantic and privacy issues: no personal contacts or individual identifiers were found, and relevant methods and limitations were preserved. This second reading was not an independent rights audit or full-PDF review. Some selections rely on abbreviations, earlier context or section pointers; preserve source metadata and do not generate questions that require definitions or context absent from the retained selection. The review record pins the three reviewed clean-file hashes.

The next data step is broader cross-corpus deduplication and any required domain review, followed by a small set of verified evidence/tool-use examples. Base-model quality, provenance and the separate training gates still apply. Current operational facts must be retrieved from current sources; these dated texts are not a replacement for that lookup.
