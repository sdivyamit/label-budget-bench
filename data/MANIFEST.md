# Data manifest

Generated 2026-11-11 by `data/manifest.py`. Do not edit by hand.

## Sources

| Source | License | Access | Files (sha256, first 12) | Accessed |
|---|---|---|---|---|
| [ragtruth](https://github.com/ParticleMedia/RAGTruth) | MIT | open | `response.jsonl` e4c2e4ac24ff (21,458,735 B)<br>`source_info.jsonl` 0dffc26ea9f3 (15,117,971 B) | 2026-11-10 |
| [llm_aggrefact](https://huggingface.co/datasets/lytang/LLM-AggreFact) | CC BY-ND 4.0 | gated (accept terms on HF; HF_TOKEN) | `dev-00000-of-00001.parquet` ca9a5bda9561 (35,453,136 B)<br>`test-00000-of-00001.parquet` ba6976c3da99 (17,298,205 B) | 2026-11-10 |
| [beavertails_30k](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | CC BY-NC 4.0 | open | `train.jsonl.gz` 2e24b85dd599 (4,948,694 B)<br>`test.jsonl.gz` 0ce619bf4c42 (545,222 B) | 2026-11-10 |
| [wildguardtest](https://huggingface.co/datasets/allenai/wildguardmix) | ODC-BY | gated (accept terms on HF; HF_TOKEN) | `wildguard_test.parquet` 6ccc2909c1ae (2,258,878 B) | 2026-11-10 |
| [toxigen_annotated](https://huggingface.co/datasets/toxigen/toxigen-data) | MIT (microsoft/TOXIGEN repository) | open | `annotated_train.parquet` 79e838cbf4a3 (689,294 B)<br>`annotated_test.parquet` c372fb0c883f (79,702 B) | 2026-11-10 |
| [implicit_hate_stg1](https://github.com/SALT-NLP/implicit-hate) | MIT | complete authors' survey, then download implicit-hate-corpus.zip into data/raw/implicit_hate/ | `implicit_hate_v1_SAP_posts.tsv` a80dc84acec1 (64,160 B)<br>`implicit_hate_v1_stg1.tsv` 8c9380566263 (635,647 B)<br>`implicit_hate_v1_stg1_posts.tsv` 00811aca26d0 (2,190,629 B)<br>`implicit_hate_v1_stg2.tsv` 0ccbae023162 (200,525 B)<br>`implicit_hate_v1_stg2_posts.tsv` 116522bdc341 (705,957 B)<br>`implicit_hate_v1_stg3.tsv` 9405cbf802f4 (397,340 B)<br>`implicit_hate_v1_stg3_posts.tsv` d6c4ff29c985 (887,584 B) | 2026-11-10 |

## Processed tasks

### t1_rag_hallucination

Positive class: response contains content not supported by the provided context.

| Split | Items | Positive rate |
|---|---|---|
| pool | 8,651 | 0.314 |
| val | 996 | 0.308 |
| test | 1,775 | 0.205 |
| shift | 3,000 | 0.686 |

N levels: [100, 250, 500, 1000, 2000, 5000, 8651]; frontier sub-pool: 5,000; detailed-prompt test subsample: 1,000.

Pool by stratification group: QA 4,386 (0.31 pos); Summary 4,265 (0.31 pos).

Cleaning: ragtruth_dropped_quality = 173; ragtruth_heldout_task_type = Data2txt; ragtruth_heldout_responses = 6195.

### t2_response_safety

Positive class: assistant response is harmful.

| Split | Items | Positive rate |
|---|---|---|
| pool | 20,000 | 0.574 |
| val | 1,000 | 0.574 |
| test | 3,021 | 0.574 |
| shift | 1,709 | 0.166 |

N levels: [100, 250, 500, 1000, 2000, 5000, 10000, 20000]; frontier sub-pool: 5,000; detailed-prompt test subsample: 1,000.

Cleaning: beavertails_conflicting_pairs_dropped = 3; beavertails_duplicate_pairs_dropped = 65; beavertails_train_test_overlap_dropped = 12; beavertails_test_prompts_seen_in_train = 2997.

### t3_implicit_toxicity

Positive class: text is hateful/toxic.

| Split | Items | Positive rate |
|---|---|---|
| pool | 5,920 | 0.376 |
| val | 1,000 | 0.376 |
| test | 752 | 0.422 |
| shift | 2,108 | 0.389 |

N levels: [100, 250, 500, 1000, 2000, 5000, 5920]; frontier sub-pool: 5,000; detailed-prompt test subsample: 752.

Pool by stratification group: asian 596 (0.33 pos); black 596 (0.41 pos); chinese 598 (0.36 pos); jewish 573 (0.43 pos); lgbtq 603 (0.39 pos); mental_dis 599 (0.33 pos); middle_east 592 (0.39 pos); muslim 579 (0.37 pos); physical_dis 577 (0.37 pos); women 607 (0.38 pos).

Cleaning: toxigen_threshold = toxicity_human >= 3.0; toxigen_test_texts_in_train_dropped = 120; toxigen_heldout_groups = ['latino', 'mexican', 'native_american']; toxigen_heldout_items = 2108.
