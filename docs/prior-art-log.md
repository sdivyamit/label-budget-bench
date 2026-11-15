# Prior-art log

Standing obligation (plan §3): re-run the saved arXiv searches every Monday until submission and log every new
hit here. If a paper appears that crosses label budget × label source × model family with calibration or
threshold drift under shift, cite it, sharpen the difference, and re-check the framing within 48 hours (plan §13).

Saved searches (arXiv cs.CL and cs.LG):
1. "label budget"
2. "LLM annotations" + "fine-tun"
3. "calibration" + "text classification"
4. "encoder" + "decoder" + "classification"

Added on 12 Nov 2026 because the results point that way:
5. "distribution shift" + "fine-tuned" + ("frontier" or "LLM API") + "classification"
6. "score-returning" or "decision model" + "classifier" (hosted probability models such as Jev)

## Entries

| Date | What was done | New papers | Effect on the paper |
|---|---|---|---|
| 2026-11-09 | Full Week-1 sweep (kill criterion). Result: conditional go. The 13 compared works are in the plan's §3 table: Airbnb engineering post (Nov 2026); arXiv 2602.06370; arXiv 2606.25782 "Do Encoders Suffice?"; INLG 2026 "Impact of Model Size…"; Cunha et al. arXiv 2504.01930; Pecher, Srba, Bielikova (EMNLP 2025); Pangakis & Wolken arXiv 2406.17633; Lu & Smith arXiv 2504.15432; ACL 2026 Balanced Brier Score; ConformalGuard arXiv 2610.14365; arXiv 2504.08716; arXiv 2509.00731; arXiv 2505.23299; plus the off-the-shelf detectors. | — | Paper framed as the interaction; adopts those papers' metrics. |
| 2026-11-12 | No new search. Two candidate claims checked against the §3 table (one-line summaries only, not the full papers). (1) "Earlier papers claimed small models are better, we show they lose under shift": not supported as worded. The works that compare small fine-tuned models with frontier APIs (Airbnb post, 2602.06370, 2606.25782) did not test a shifted set. Shift is tested only by 2509.00731 (encoders about 78% vs LoRA Qwen2.5-7B 96%, no frontier API) and ConformalGuard (cross-corpus phishing, BERT-base vs Claude Sonnet 4 and GPT-4o, about threshold control). (2) "Nobody tested Jev": not correct. The Airbnb post includes Jev, on one private task with 100k+ labels and no shift. | — | Both are gaps the study fills, not contradictions of published claims. To do before writing: read ConformalGuard and the Airbnb post in full, in particular who wins under shift in ConformalGuard. |

| 2026-11-15 | Final pre-submission search, run a day early (Sun) on S.R.'s request. arXiv API, cs.CL OR cs.LG, submitted 9–15 Nov 2026, abstract field: all six saved searches (hits: 1 "label budget" 2; 2 "LLM annotations" + fine-tun 0; 3 calibration + "text classification" 0; 4 encoder + decoder + classification 1; 5 shift + fine-tuned + frontier/LLM API + classification 0; 6 score-returning/decision model + classifier 0), plus six broader screens by title (calibration 65; LLM annotations/labels/distillation 21; shift/OOD + classifier 2; hallucination/toxicity/moderation detection 7; fine-tuned/encoder vs API/zero-shot classification 1; conformal/threshold + classification 9). Filter sanity check: 300 cs.CL and 705 cs.LG submissions in the window. 14 abstracts read in full. | Related, none crossing the axes: 2611.12521 (calibration and explainability audit of open Jev-style "System 1" decision models, 0.8B–27B, under label-preserving perturbations; no label budget or label source); 2611.14861 (Jev as a multilingual judge at 1/31–1/61 of GPT-6.1's cost; its probabilities help flag errors); 2611.13339 (definition sensitivity of LLM annotators); 2611.15197 (LLM annotations with budgeted human auditing for mean estimation); 2611.15911 (analytical cost model for routing between small and frontier LLMs). The six saved-search hits are off-topic (molecular data acquisition, active learning, graph pretraining). | None. No paper crosses label budget × label source × model family with calibration or threshold drift under shift; the "no surveyed work combines…" framing stands. 2611.12521 and 2611.14861 agree with our use of Jev (graded probability used as a score, no calibration claim made). No citation added (main paper at the 20-page limit); authors may cite 2611.12521 next to Jev if a cut frees space. Related Work's "last search" date set to 15 November 2026. Papers submitted on 15 Nov appear in arXiv listings from 16–17 Nov, so a re-run on Mon 16 Nov before submitting would cover that day. |

Searches closed at submission. Re-run weekly only if the paper goes to the next venue.
