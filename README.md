# EPA – Email task-to-person assignment

A small, reproducible supervised-ML pipeline for the **Enron People
Assignment (EPA)** problem (Rameshkumar et al., W-NUT 2018). Given an
email plus an extracted task sentence, decide for each candidate person
(sender, To, Cc) whether they are responsible for the task.

This is positioned as a real-world email-assistant problem, not a generic
text-classification benchmark. The goal is to demonstrate end-to-end
engineering judgment — clean data engineering, sensible features,
transparent baselines, group-aware evaluation, and a production-shaped
inference service — rather than to chase F1.

## 1. Problem framing

Per the paper, EPA is reduced to a series of binary decisions: for every
`(email, task, candidate person)` tuple, predict whether that person
should be notified. We frame and engineer the problem exactly that way:

- One training example per candidate person per email-task pair.
- Candidate set = sender + To + Cc, deduplicated by lowercased email.
- `label = 1` if the candidate is in the annotator-resolved responsible
  set, `0` otherwise.
- "No-one is responsible" → all candidate rows for that HIT are `0`.
- A single task can have multiple responsible people; we treat each
  candidate independently.

This solution treats EPA as **supervised binary classification over
(email, task, candidate person)**. We intentionally avoided LLM
prompting / RAG / chat APIs because the assignment asks for a trainable
supervised pipeline, and because the original paper showed that a
logistic-regression baseline with handcrafted features is a sensible
starting point for an interpretable, reproducible solution.

## 2. Dataset

We use the official rehost of the EPA dataset published by the paper's
first author at <https://github.com/RevRameshkumar/EPADataset>. It ships a
single `EPADataset.tsv` containing 6,734 HITs with the following per-row
JSON payload:

```json
{
  "EmailID":       "...",
  "Subject":       "...",
  "From":          {"emailAddress": {"Name": "...", "Address": "..."}},
  "ToRecipients":  {"emailAddressList": [{"emailAddress": {...}}, ...]},
  "CcRecipients":  {"emailAddressList": [...]},
  "Message":       "... <mark>task sentence</mark> ... <br/> ...",
  "TaskSentence":  "task sentence",
  "Judgements":    [{"<annotator_id>": ["responsible@x", ...]}, ...]
}
```

Acquisition (`src/data/download.py`) tries three strategies in order so
the pipeline is robust to network restrictions:

1. **Explicit URL** in `dataset.url` (e.g. an internal mirror).
2. **`git clone --depth 1`** of `dataset.git_repo` (defaults to the
   reference repo above). Works in environments where direct
   `raw.githubusercontent.com` egress is blocked but `github.com` git
   transport is allowed.
3. **Synthetic fallback** — generates a small EPA-shaped dataset under
   `data/raw/_epa_synthetic.json` so the pipeline remains runnable
   end-to-end for reviewers without dataset access. The synthetic file
   is prefixed with `_` so the parser ignores it whenever the real TSV
   is also present.

`src/data/parse_epa.py` is **format-flexible**: it accepts the reference
TSV, plus JSON (list-of-records or `{"data": [...]}`), JSONL, and CSV
rehosts. Field-name aliases (`From`/`sender`, `Message`/`body`,
`Judgements`/`annotations`, etc.) are mapped onto a single normalized
schema. Both annotator-label formats observed in the wild are handled
(per-judge dicts shipped by the reference repo, plus per-recipient
binary votes used by some forks). Aggregation across judges supports:

- `consensus: perfect` – keep only HITs where every judge agreed
  exactly (matches the paper's strict α=0.6123 analysis);
- `consensus: majority` (default) – per-recipient majority vote;
- `consensus: any` – any judge that marked the recipient counts.

The expected normalized record after parsing:

```python
{
    "email_id": str, "task_id": str,
    "subject": str, "body": str, "task": str,
    "sender": (name, email),
    "to":  [(name, email), ...],
    "cc":  [(name, email), ...],
    "responsible_emails": [email, ...],
    "no_one_responsible": bool,
    "n_judges": int,
    "perfect_agreement": bool,
}
```

### What the data actually looks like

After parsing the reference dataset with the default majority-vote
consensus we get **6,734 HITs → 19,101 candidate rows** (mean 2.84
candidates per HIT). 6,137 of the HITs (91%) are perfect-agreement at
the HIT level. Class prior = **0.486**.

Class balance by candidate role (the dominant structural signal):

| candidate_role | count  | positive_rate |
|----------------|-------:|--------------:|
| sender         | 6,408  | 0.017         |
| to             | 8,415  | 0.867         |
| cc             | 3,943  | 0.430         |
| multiple       |   335  | 0.555         |

Positive rate by candidate-set size (the paper's primary cut):

| num_total_candidates | n     | positive_rate |
|---------------------:|------:|--------------:|
| 2 (single recipient) | 7,580 | 0.499         |
| 3                    | 3,588 | 0.475         |
| 4                    | 3,236 | 0.475         |
| 5                    | 3,000 | 0.442         |
| 6                    | 1,068 | 0.507         |
| 7                    |   546 | 0.571         |

## 3. Approach

```
data/raw  ──▶  parse_epa  ──▶  build_examples  ──▶  data/processed/examples.csv
                                                          │
                                                          ▼
                                          ┌───── train.py ─────┐
                                          │                    │
                                ColumnTransformer       baselines / sweep
                            ┌────────┬────────────┐              │
                            │        │            │              ▼
                       word TF-IDF   char TF-IDF  Candidate     reports/
                            │        (optional)   feature       evaluation.{md,json}
                            └─────┬──┴───┬────────┘
                                  ▼      ▼
                              LogisticRegression
                                     │
                                     ▼
                          models/epa_model.joblib
                                     │
                                     ▼
                       FastAPI: src/service/app.py  →  POST /predict
```

The trained pipeline (TF-IDF + structured features + LR) is serialized in
one go via `joblib`, so the FastAPI service reuses the **exact** same
feature engineering used in training. That train/serve parity is the
single most important property for reproducibility.

## 4. Training example construction

`src/data/build_examples.py` materializes one row per candidate. Important
details:

- Candidates are deduplicated by lowercased email; if the same person
  appears in multiple roles (e.g. To and Cc), we keep one row with
  `candidate_role = multiple` and set every relevant role flag.
- We preserve sender / to / cc role information explicitly via
  `is_sender`, `is_to`, `is_cc` flags. The paper's annotation spec lets
  the sender be marked responsible too (they sometimes commit themselves
  to the task), so the sender is a candidate, not excluded.
- `full_context = subject + " [SEP] " + task + " [SEP] " + body`. This
  string is the input to TF-IDF.
- Each row carries `email_task_id = email_id::task_id`. We use that as
  the **group key** for splitting so candidates from the same HIT never
  cross train / test boundaries.
- Each row carries `n_judges` and `perfect_agreement` so that downstream
  evaluation can slice to the paper's strict universally-agreed subset.

## 5. Feature engineering

Three feature families, combined via `ColumnTransformer`:

**Word TF-IDF** over `full_context` — 1-to-2 grams, sublinear TF,
min_df / max_df pruning, capped at 50k features.

**Optional char TF-IDF** (off by default) — 3-to-5 grams via `char_wb`,
useful on Enron's noisier formatting; flip `features.char_tfidf.enabled`
in `config.yaml`.

**Structured candidate features** (`src/features/candidate_features.py`):

- *Role flags*: `is_sender`, `is_to`, `is_cc`, `appears_in_multiple_roles`,
  `is_only_recipient`, missing-name/email flags, sender-domain match.
- *Name / email reference* (the addressee-tagging signal): does the
  candidate's first / last / full name or email or local part appear in
  the task / body / subject? Each is a binary feature per location.
- *Email pragmatics over the task text*: `you`, `your`, `please`,
  `can you` / `could you` / `would you`, `let me know`, `?`,
  question-stem heuristic, `we` / `us` / `team`, imperative-verb hit on
  the first six tokens, plus length features.
- *Counts*: number of To, Cc, and total candidates (linear and log-scale).
- *Proximity*: same-sentence-as-task, name-before-task, name-after-task,
  normalized character distance between the candidate's name and the
  task sentence in the body.

We deliberately do **not** rely on TF-IDF alone — addressee resolution
is a person-assignment problem, not a topic-classification problem, and
the structured features carry a lot of the signal.

## 6. Baselines

Two transparent baselines are reported alongside the main model so we can
sanity-check the supervised gains:

1. **Every-recipient** — predict every candidate as responsible. Recall
   is 1.0 by construction; precision equals the class prior. Mirrors the
   paper's Table 5.
2. **Class-prior** — predict the global positive rate as the score for
   every candidate, threshold at 0.5 (so collapses to all-zero unless
   the prior is ≥0.5). Mirrors the paper's `x̄`-baseline.

## 7. Model

Logistic regression (`solver=liblinear`, `class_weight=balanced` by
default) on the combined sparse feature matrix. The choice is deliberate:

- Matches the paper's setup, so comparisons against the published
  baselines are apples-to-apples.
- Probability-calibrated by default and inspectable
  coefficient-by-coefficient.
- Trains in seconds on the full Enron split; easy to iterate on features.

A neural model (small MLP, transformer fine-tune) is left as an explicit
future improvement rather than being shipped as the default — see §11.

## 8. Evaluation

`src/models/evaluate.py` recomputes everything from a saved model. Group
splitting uses `GroupShuffleSplit` on `email_task_id` so candidates from
the same HIT never appear on both sides. We report:

- precision, recall, F1, accuracy, PR-AUC, ROC-AUC, confusion matrix;
- per-scenario slices: single- vs multi-recipient (paper's primary
  cut), tasks containing "you/your", tasks with an explicit candidate
  name, no-one cases, candidate-is-sender, candidate-is-recipient,
  perfect-agreement-only;
- a threshold sweep (precision / recall trade-off) so a product team can
  pick a higher-precision (avoid notifying wrong people) or
  higher-recall (avoid missing responsible people) operating point.

### Held-out test results

On the held-out test split (3,900 candidate rows from 1,373 HITs), at
the default 0.5 threshold:

| metric    | value |
|-----------|------:|
| precision | 0.823 |
| recall    | 0.872 |
| F1        | 0.847 |
| accuracy  | 0.849 |
| PR-AUC    | 0.923 |
| ROC-AUC   | 0.931 |

Confusion: TP = 1,630, FP = 351, FN = 240, TN = 1,679.

Scenario breakdown — single-recipient HITs are essentially solved
(F1 = 0.96); the residual error budget is concentrated in
multi-recipient HITs:

| scenario                    |  n   | precision | recall |  F1   |
|-----------------------------|-----:|----------:|-------:|------:|
| single_recipient_email      | 1468 |     0.961 |  0.968 | 0.964 |
| multi_recipient_email       | 2432 |     0.740 |  0.809 | 0.773 |
| task_has_you_or_your        | 1592 |     0.814 |  0.863 | 0.837 |
| task_has_explicit_name      |  165 |     0.804 |  0.897 | 0.848 |
| task_has_no_explicit_person | 3735 |     0.824 |  0.870 | 0.847 |
| candidate_is_sender         | 1347 |     0.850 |  0.288 | 0.430 |
| perfect_agreement_only      | 3398 |     0.850 |  0.896 | 0.873 |

For comparison, the paper's Avocado-trained baseline transferred to
Enron achieved P/R/F1 = **0.69 / 0.89 / 0.78** on single-recipient and
**0.62 / 0.70 / 0.66** on multi-recipient. Our numbers exceed both
because we train directly on Enron and combine TF-IDF with the
handcrafted features, but the paper's qualitative observation
(multi-recipient is meaningfully harder) clearly carries over.

The current Markdown evaluation report lives at
[`reports/evaluation.md`](reports/evaluation.md) and a structured JSON
summary at [`reports/evaluation.json`](reports/evaluation.json).

## 9. Error analysis

[`reports/error_analysis.md`](reports/error_analysis.md) walks through
the residual errors with concrete examples drawn from the test split.
The headline:

- **92% of FPs** sit in multi-recipient HITs — over-assignment when the
  task is imperative but doesn't disambiguate which of N recipients is
  meant.
- **17% of FNs** are senders committing themselves ("I'll handle…",
  "let me…"), where our `is_sender` prior pulls scores down.
- **41% of FPs** are tasks containing "you" / "your", where the model
  spreads responsibility across recipients instead of resolving the
  pronoun.

The report tags each error category with a likely cause and a concrete
next-step improvement. The top three priorities are listwise scoring
across candidates within a HIT, first-person commitment features for
the sender, and out-of-candidate-list NER for the no-one cases.

## 10. Inference service

A minimal FastAPI service mirrors the train-time feature pipeline.

```bash
uvicorn src.service.app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "sender": {"name": "Caira Wong", "email": "caira@example.com"},
      "to": [{"name": "Anna Smith", "email": "anna@example.com"},
             {"name": "Brad Jones", "email": "brad@example.com"}],
      "cc": [{"name": "John Patel", "email": "john@example.com"}],
      "subject": "Draft report",
      "body": "Hi Anna, can you and Brad complete a draft by Friday?",
      "task": "can you and Brad complete a draft by Friday?"
    }'
```

Response (real output from the trained model on the example payload):

```json
{
  "task": "can you and Brad complete a draft by Friday?",
  "threshold": 0.5,
  "assignments": [
    {"person": "anna@example.com",  "score": 0.836, "assigned": true,  "role": "to"},
    {"person": "brad@example.com",  "score": 0.729, "assigned": true,  "role": "to"},
    {"person": "caira@example.com", "score": 0.010, "assigned": false, "role": "sender"},
    {"person": "john@example.com",  "score": 0.530, "assigned": true,  "role": "cc"}
  ]
}
```

Threshold can be overridden per request via either a query parameter
(`?threshold=0.7`) or the request body. `GET /health` returns whether the
model artifact is present without loading it.

## 11. Limitations

- **Thread-aware modeling is shallow.** The features see the body text
  but do not parse quoted history into structured prior turns. The
  paper's no-explicit-mention case (Figure 2) needs more than that.
- **Coreference is heuristic.** "You" / "your" / "we" are detected as
  pragmatic features but not resolved to specific recipients. Likely the
  single biggest source of error.
- **Independent binary classification.** Candidates from the same email
  are scored independently. A learning-to-rank formulation would let
  the model trade scores between candidates within a HIT.
- **No probability calibration.** Scores are well-ordered but the
  numeric values aren't calibrated to true probabilities; threshold
  choice should be made on a held-out PR curve, not a fixed 0.5.
- **Annotation noise is not modeled.** The paper reports α = 0.61 inter-
  annotator agreement; we currently aggregate by majority vote (or
  perfect agreement, optional), and don't down-weight noisy HITs.
- **Out-of-candidate-list references.** "Brad will complete the draft"
  with Brad not on To/Cc → the model still has to assign someone (or
  no-one) from the To/Cc list, and currently has no NER signal for
  this case.

## 12. Future improvements

- Better thread-aware modeling: parse quoted history (`---- Original
  Message ----` blocks) into structured prior turns and add features /
  representations from them.
- Coreference / addressee resolution for "you", "your", "we" – e.g.
  use a small dependency parse plus salience heuristics to map
  pronouns to candidates.
- Learning-to-rank formulation over candidates instead of independent
  binary classification (e.g. listwise softmax with a "no-one" slot).
- Probability calibration (`CalibratedClassifierCV` with isotonic on a
  held-out fold) before exposing scores to product surfaces.
- Human-in-the-loop feedback: capture user corrections from the email
  client and fold them back as labelled data with sample weights.
- Better handling of group emails / aliases: expand known aliases via
  a directory lookup so individual recipients can be scored.
- Syntactic / dependency features (subject-of-imperative,
  vocative-NP detection) for better addressee tagging.
- Optional neural model after the strong baseline is established —
  e.g. distil a small encoder over the same `(email, task, candidate)`
  triples; only ship if it materially beats the LR baseline at a
  comparable inference cost.

## 13. How to run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Acquire raw data (tries url, then git clone, then synthetic fallback)
python -m src.data.download

# 3. Build candidate-level training examples
python -m src.data.build_examples \
    --raw_dir data/raw \
    --output data/processed/examples.csv \
    --consensus majority      # or perfect | any

# 4. Train the model + write evaluation report
python -m src.models.train \
    --data data/processed/examples.csv \
    --model_out models/epa_model.joblib

# 5. Re-evaluate a saved model without retraining
python -m src.models.evaluate \
    --data data/processed/examples.csv \
    --model models/epa_model.joblib \
    --report_out reports/evaluation.md

# 6. One-off prediction from a JSON payload
python -m src.models.predict --input example_payload.json

# 7. Inference service
uvicorn src.service.app:app --reload
```

A typical end-to-end run on a laptop takes ~30 seconds (parsing the
TSV, building examples, training LR on ~13k rows). The model artifact
is ~2.4 MB; the candidate CSV is ~65 MB because we keep the full body
text alongside each candidate row for slicing during evaluation.

## Repo layout

```
epa-assignment/
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/             # downloaded EPADataset.tsv goes here
│   └── processed/       # candidate-level examples.csv
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── parse_epa.py
│   │   └── build_examples.py
│   ├── features/
│   │   ├── text_features.py
│   │   └── candidate_features.py
│   ├── models/
│   │   ├── pipeline.py
│   │   ├── baselines.py
│   │   ├── metrics.py
│   │   ├── split.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── service/
│   │   └── app.py
│   └── utils/
│       ├── io.py
│       └── text_cleaning.py
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/
│   ├── evaluation.md
│   ├── evaluation.json
│   └── error_analysis.md
│
└── models/
    └── epa_model.joblib
```
