# Venue-customized submission versions

Each subfolder is a **self-contained, submission-ready** condensation of the full
paper (`../Imdad_Final_Research_Paper.{tex,pdf}` is the full-length / arXiv
version). The short versions lead with the outlier-free → INT8-deployment result
and compress the register experiment to one paragraph, to fit conference limits.

Every folder includes its own style/class files and figures, so it compiles
standalone:

```bash
cd <venue> && pdflatex <file>.tex && pdflatex <file>.tex   # 2 passes for refs
```

| Folder | Venue | Template | Built | Limit | Authorship |
| --- | --- | --- | --- | --- | --- |
| `icassp/` | IEEE ICASSP | `IEEEtran` (conference) | **3 pp** | 4 + 1 ref | named |
| `dcase/` | DCASE Workshop | official `dcase2024.sty` | **3 pp** | 5 (4+1) | named |
| `interspeech/` | Interspeech | official `Interspeech2024.cls` | **3 pp** | 4 + refs | **anonymized** |
| `workshop/` | NeurIPS-style ML workshop | `neurips_2024.sty` | **4 pp** | ~4–9 | named (preprint) |

## Per-venue notes

- **ICASSP** — non-anonymous (single-blind). Sits at 3 pp with a full page of
  headroom; you can expand the register/results discussion toward the 4-page
  limit if desired. Replace the IEEE copyright notice block per the CFP.
- **DCASE** — uses the official 2024 workshop template (an ICASSP/`spconf`-style
  format). DCASE workshop review is non-anonymous. 5-page limit incl. references.
- **Interspeech** — built in **double-blind review mode** (the default of
  `Interspeech2024.cls`): the author block renders as "Anonymous submission",
  PDF metadata is "Under review / Author name(s) withheld", and the body has no
  self-identifying text (no affiliation, no code URL). The author/affiliation
  metadata is filled in the source but suppressed. **For camera-ready:**
  uncomment `\interspeechcameraready` and restore the code-release URL in the
  conclusion.
- **Workshop** — NeurIPS style in `[preprint]` mode (authors shown). For a
  double-blind workshop, remove the `[preprint]` option; for camera-ready use
  `[final]`. Title is deployment/efficiency-forward.

## Updating before submission

1. Confirm the **target year's** template (these use 2024 kits; swap the
   `.cls`/`.sty` if the venue posts a newer one — the body is portable).
2. Add the venue's required copyright/CFP boilerplate.
3. Re-read each venue's anonymization policy and verify the build matches.
4. The condensed bibliographies carry ~8 key references; expand if the venue
   expects fuller related work.
