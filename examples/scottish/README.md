# Scottish English quickstart audio

Five complete utterances (19.456 seconds total; about 1.9 MB) from
[OpenSLR SLR83](https://www.openslr.org/83/), obtained from the local cache of
[ylacombe/english_dialects](https://huggingface.co/datasets/ylacombe/english_dialects),
revision `ed3d69abb765fd304ccdcc646ec6ca3c49740d15`, split `train`.
Female/male labels below are the source subset labels. Speaker numbers are the
source's anonymised IDs, not identities.

| Audio and matching TXT basename | Subset | Line ID | Speaker ID | Seconds |
|---|---|---|---:|---:|
| `scf_04310_02119798587` | `scottish_female` | BI0069 | 4310 | 4.181 |
| `scf_07049_00705812248` | `scottish_female` | BI0107 | 7049 | 3.840 |
| `scf_03397_02044513859` | `scottish_female` | EN0168 | 3397 | 3.669 |
| `scm_07508_01418625966` | `scottish_male` | SC0007 | 7508 | 3.840 |
| `scm_02484_00589991076` | `scottish_male` | BI0057 | 2484 | 3.925 |

Each basename has a `.wav` and a UTF-8 `.txt`. References match both the cached
`text` field and the corresponding file ID in the upstream
[line index](https://www.openslr.org/resources/83/line_index_all.csv). Original
wording, case and punctuation are preserved; TXT files add one final newline.
[source_references.csv](source_references.csv) contains only those five index records.
[examples.json](examples.json) records source IDs, pinned Parquet locations, row
indices, reference text, native audio properties and SHA-256 hashes.

The WAVs were copied byte-for-byte from the cached Hugging Face audio payloads:
48 kHz, mono, 16-bit PCM. No conversion, resampling, trimming, denoising or volume
changes were made for this bundle. The CLI resamples to 16 kHz in memory during
transcription. Original OpenSLR ZIP-member byte identity was not independently
checked; no earlier conversion history beyond the Hugging Face repackage is
asserted. These five illustrative clips are not a new evaluation split.

## Attribution and licence

Copyright 2018, 2019 Google, Inc. Corpus: *Crowdsourced high-quality UK and Ireland
English Dialect speech data set*, OpenSLR identifier SLR83. Hugging Face repackage:
`ylacombe/english_dialects` (publisher: `ylacombe`). No endorsement is implied.

Audio and reference texts in this folder are distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0). The upstream legal text is included in [LICENSE](LICENSE).
Retain attribution and change notices when redistributing; adaptations must use
the same licence or a compatible licence as specified by its terms.

Citation: Isin Demirsahin, Oddur Kjartansson, Alexander Gutkin and Clara Rivera.
2020. [Open-source Multi-speaker Corpora of the English Accents in the British
Isles](https://aclanthology.org/2020.lrec-1.804/). LREC, pages 6532–6541.

This example-data licence is separate from the repository code. It does not grant
a code licence; the repository currently has no separate code `LICENSE` file.
No Common Voice, Shetland or police recordings are included here.

See the repository [quickstart](../../README.md#run-a-bundled-example) for one-clip
and five-clip commands. Generated outputs belong in ignored `runs/` folders.
