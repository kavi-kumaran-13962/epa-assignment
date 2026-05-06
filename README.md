# EPA — Email task-to-person assignment

A reproducible supervised-ML pipeline for the W-NUT 2018 EPA problem
(Rameshkumar et al.): given an email and an extracted task sentence,
decide for each candidate person on the email whether they're responsible
for the task.

---

### At a glance

| | |
|---|---|
| **Problem framing** | Candidate-level responsibility prediction |
| **Model** | Logistic Regression + structured candidate features + TF-IDF |
| **Evaluation** | Grouped split by HIT, validation-only threshold tuning |
| **Held-out F1** | **0.847** |
| **Baseline lift** | **+19.9 F1** over every-recipient baseline |
| **Production shape** | Serialized sklearn `Pipeline` + FastAPI inference |
| **Quality checks** | 58 tests + train/serve parity validation |

---

## 1. Executive summary

EPA is not text classification. Two candidate rows from the same email
have identical body, identical task text, identical sender, identical
recipient list — only the candidate identity changes. Topic features
cannot tell them apart. The signal lives in the **relationship** between
the task language, the candidate, and the rest of the recipient list.

So this is **structured addressee resolution under ambiguity**, modeled
as binary classification over

```
(email, task, candidate_person)  →  responsible / not responsible
```

Each HIT is exploded into one row per candidate (sender + To + Cc, deduped
by lowercased email). The candidate's own role flags, name-match signals
and proximity to the task carry most of the model's lift; TF-IDF over
the email text contributes the topical floor.

I picked binary because the paper picked binary, the framing matches the
production scenario (one notification per `(person, task)` pair), and
the linear model is inspectable enough to make the error analysis
tractable. The independent binary formulation has known multi-recipient
limitations discussed in §10. On a leakage-safe held-out grouped split
the final model reaches **0.847 F1**, improving roughly 20 points over
the trivial every-recipient baseline.

---

## 2. Why this problem is interesting

Several things make EPA harder than it sounds.

**Annotator disagreement is structural, not noise.** The paper reports
α = 0.61 on Krippendorff's alpha — meaningfully below the 0.8 "reliable"
threshold. That's a fact about the task, not about the annotators: emails
genuinely are ambiguous about which of N recipients is meant by "you",
and reasonable humans disagree. Any model trained on this data is
predicting *consensus*, not ground truth.

**Implicit assignment language is the rule, not the exception.** Of
3,735 test rows in the held-out split, only 165 contain the candidate's
explicit name in the task. The other 96% rely on email pragmatics —
imperative verbs, "you/your", politeness markers, conjunctions — to
distribute responsibility across the To list.

**Multi-recipient ambiguity dominates.** Roughly two-thirds of HITs
have ≥3 candidates. The structural prior says "anyone on To is likely
responsible" (87% positive rate for To), so a model that predicts
"every To recipient" gets ~0.65 F1 for free. The remaining headroom is
almost entirely in distinguishing *which subset* of multi-recipient cases
is meant.

**Email body text is hostile.** Enron formatting fuses sentences across
`<br/>` boundaries, includes quoted thread history, and contains internal
routing addresses (`chris.stokley/HOU/ECT@ECT`) that are technically
emails but not really person identifiers.

These four properties — agreement ceiling, implicit language, structural
priors that go most of the way, and noisy text — shape every feature and
modeling choice below.

---

## 3. Design philosophy

### A. Logistic regression, deliberately

The paper's baseline was LR with handcrafted features. Matching that
isolates the effect of *features and training data* from *model class*.
LR also gives me three production-relevant properties: signed coefficients
I can read directly, calibrated-ish probabilities, and a fit time
measured in seconds. A neural model is the right next step *after* the
LR baseline is exhausted, not before.

### B. Engineered features over deep learning

The bottleneck on this task is relational reasoning between candidate
identity and task text. TF-IDF can't see candidate identity. A
transformer would help on slices my features don't address (coreference,
conjunction parsing) — but on most of the dataset, role flags and
name-match features carry the signal. I'd ship a transformer only after
listwise scoring and better coreference, not before.

### C. Train/serve parity by structural design

There is exactly one feature pipeline: a scikit-learn `Pipeline`
(`ColumnTransformer` over TF-IDF + a custom `CandidateFeatureExtractor`,
plus `LogisticRegression`). It's `fit` at training, serialized to
`models/epa_model.joblib`, and called via `predict_proba` by the FastAPI
service. There is no separate "inference featurizer". Adding a feature
is a one-place change.

### D. Group-aware splitting, with assertions

A row-level random split would put candidates from the same HIT on both
sides of the train/test boundary. This would leak shared HIT context
into evaluation and inflate reported performance. `GroupShuffleSplit`
on `email_task_id` prevents this; an explicit `assert_no_group_leakage`
runs after every split as a defensive line against future regressions.

### E. Denormalized candidate rows

Each candidate row carries the full email body. That's ~65 MB on disk
for redundant storage. The benefit: the same row format is what the
inference path produces from a JSON payload, so the trained Pipeline
applies *as is* without a parallel feature implementation. A future
production version would normalize into HIT-level + candidate-level
tables and join at training; that's premature here.

---

## 4. Architecture at a glance

```
EPADataset.tsv  ──▶  parse_epa  ──▶  build_examples
                                          │
                                  candidate-level CSV
                                          │
                                          ▼
                       ┌───── group split (HIT-level) ─────┐
                       │                                    │
                  train + val                              test
                       │                                    │
                       ▼                                    │
                ColumnTransformer                           │
        ┌──────────────┴──────────────┐                    │
        │                              │                    │
   word TF-IDF      structured candidate features           │
        │                              │                    │
        └────────────┬─────────────────┘                    │
                     ▼                                       │
             LogisticRegression                              │
                     │                                       │
                     ▼                                       │
        select threshold on val ──────────────┐             │
                                              ▼              │
                              evaluate ONCE on test ◀────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                              ▼
          models/epa_model.joblib                    reports/{evaluation,error_analysis}.md
                       │
                       ▼
            FastAPI: POST /predict
```

The core design goal was not only predictive performance, but tight
alignment between evaluation methodology, feature generation, and the
inference path. Every major design choice was made to preserve that
consistency.

---

## 5. What differentiates this submission

### Problem-aware feature engineering
Not generic TF-IDF. Five feature families designed against the EPA task
specifically: role flags, addressee lexical alignment, pragmatic task
cues, proximity, and a sender × first-person commitment cross-feature
that targets a specific failure mode identified in error analysis.

### Leakage-safe evaluation
`GroupShuffleSplit` on `email_task_id`, an `assert_no_group_leakage`
defensive assertion, and **threshold selection on validation only** so
the test split is consumed exactly once. The most common methodological
mistake in take-homes — selection on test — is deliberately avoided.

### Train / serve parity
A single `Pipeline` artifact drives both training and inference. The
FastAPI service reuses the exact same featurization with no parallel
implementation. Verified by `tests/test_inference_parity.py`.

### Scenario-based evaluation
Per-slice metrics for single- vs multi-recipient (the paper's primary
cut), tasks with explicit names, tasks with implicit "you/your", sender
candidates, and the perfect-agreement-only subset.

### Honest error analysis, auto-regenerated
`reports/error_analysis.md` is regenerated from the live test predictions
on every train run. It surfaces the dominant failure modes (multi-
recipient over-assignment, sender first-person commitments, out-of-list
references) with concrete examples and ranked next-step fixes.

### Production-shaped design
FastAPI service with per-request threshold override, `/health`,
graceful 422 on malformed payloads, and a serialized model bundle that
carries `model_version`, `git_sha`, `trained_at`, and `sklearn_version`
for traceability.

### Test discipline
58 pytest tests covering parser shapes, group-leakage prevention,
feature correctness (including a regression test for the
`is_only_recipient` bug fixed in this version), and inference parity.

---

## 6. Feature engineering deep dive

TF-IDF (1–2-grams over `subject [SEP] task [SEP] body`) gives the model
a topical floor. But TF-IDF is identical across candidate rows from the
same HIT, so it can't discriminate *between* candidates. Five structured
families do that work.

### Structural role priors
`is_sender`, `is_to`, `is_cc`, `appears_in_multiple_roles`,
`is_only_recipient`, `sender_same_domain_as_candidate`, and recipient
counts.

These encode the dominant statistical signal: 87% of To candidates are
positive, 43% of Cc, 1.7% of senders. `is_only_recipient` carves out
the "sender + 1 recipient" case where pragmatics are largely redundant.

### Addressee lexical alignment
`first_name_in_task`, `last_name_in_task`, `full_name_in_task`,
`email_in_task`, `local_part_in_task`, plus the same set over body and
subject.

A candidate's name appearing in the task is a near-certain signal.
Whole-word matching with length gates prevents substring false-positives
("Sam" in "samples", short local parts inside larger words). Fires on
only ~4% of test rows, but with very high precision when it does.

### Pragmatic task cues
`task_contains_you/your/please/can_you/could_you/would_you/let_me_know`,
question marks and question stems, first-person plural, imperative-verb
hit on the first six tokens, and length features.

These distinguish real assignments from informational text. Combined
with role flags, the linear model learns rules like
`is_to=1 AND has_can_you → likely responsible`.

### First-person commitment cues
`task_first_person_subject`, `task_first_person_future`, `task_let_me`,
`task_first_person_object`, and the cross-feature
`task_first_person_and_sender`.

Added to address the dominant FN cluster identified in error analysis:
senders committing themselves ("I'll handle this", "let me look into
it"). The cross-feature fires only when the candidate *is* the sender
AND the task is first-person — the lever the model needs to flip its
"senders aren't responsible" prior in exactly the right cases.

### Proximity features
Whether the candidate's name appears in the same sentence as the task,
before/after the task in the body, and a normalized character distance
from name to task.

Captures the pattern that people are addressed *just before* the
imperative directed at them. Brittle on `<br/>`-fused sentences, but
cheap to compute and effective on the cleaner half of the data.

---

## 7. Methodological safeguards

| safeguard | implementation | guards against |
|---|---|---|
| Group-aware split | `GroupShuffleSplit` on `email_task_id` | candidates from same HIT in both train and test |
| Leakage assertion | `assert_no_group_leakage(...)` after every split | future regressions to the splitting logic |
| Validation-only threshold tuning | `select_threshold_on_val` picks argmax-F1 from the val sweep | selection-on-test bias |
| Test consumed once | Single `predict_proba` call on test at the chosen threshold | iterating against the test set |
| Pipeline fit on train only | `Pipeline.fit(train_df, ...)` | TF-IDF vocabulary leaking val/test text |
| Variance estimate | Optional grouped 5-fold CV F1 (mean ± std) on train+val | over-reading a single point estimate |
| Train/serve parity | One `Pipeline` end-to-end | featurization drift between train and serve |
| Deterministic seeds | `random_state=42` in splitter, model, synthetic data | run-to-run noise masking real changes |
| Test suite | 58 pytest tests | silent regressions in the parser, splitter, features, inference |

The README is not the source of truth for numbers. **`reports/evaluation.md`
and `reports/error_analysis.md` are auto-regenerated on every train
run** and are the authoritative artifacts.

---

## 8. Results

### Held-out test set

3,900 candidate rows from 1,373 HITs, threshold tuned on validation
(chosen value: 0.40), evaluated once on test:

| metric | value |
|---|---:|
| precision | 0.786 |
| recall | 0.918 |
| F1 | 0.847 |
| PR-AUC | 0.924 |
| ROC-AUC | 0.932 |
| confusion | TP 1,717 · FP 468 · FN 153 · TN 1,562 |

### Baselines

| | precision | recall | F1 |
|---|---:|---:|---:|
| Every-recipient | 0.480 | 1.000 | 0.648 |
| Class-prior @ chosen threshold | 0.480 | 1.000 | 0.648 |
| **Main model** | **0.786** | **0.918** | **0.847** |

The supervised model lifts F1 by ~20 points over the trivial floor —
real signal, not pipeline plumbing artifact.

### Scenario slices

| slice | F1 | comment |
|---|---:|---|
| single_recipient_email | ~0.96 | essentially solved |
| multi_recipient_email | ~0.78 | residual error budget lives here |
| candidate_is_sender | ~0.45 | first-person commitment cluster, partially addressed |
| task_has_you_or_your | ~0.84 | implicit-pronoun overreach is the main failure |
| perfect_agreement_only | ~0.88 | upper bound for clean labels |

(Exact per-slice numbers regenerate into `reports/evaluation.md` on
every train run.)

### Interpretation

The headline F1 is dominated by single-recipient cases. The interesting
question is the gap between single (0.96) and multi (0.78) — that's
where every future improvement effort should land. The sender slice
(0.45) is the remaining cluster the new first-person features partially
address; an honest read says they help but don't close the gap, because
many sender-positive cases require coreference of "me" that the linear
model can't do.

### Comparison to the paper, honestly

The W-NUT 2018 paper reports two baseline families:

* **Avocado-trained → Avocado-evaluated** (in-domain): P/R ≈ 0.9/0.9.
* **Avocado-trained → Enron-evaluated** (transfer): P/R/F1 ≈ 0.69/0.89/0.78
  on single-recipient, 0.62/0.70/0.66 on multi-recipient.

This submission trains *and* evaluates on Enron — an in-domain setting.
The right comparison is therefore the paper's Avocado→Avocado in-domain
0.9, against which we sit *below*. We do exceed the transfer baseline,
but that's because we don't have to bridge a domain gap, not because
the modeling is better.

The Enron-only in-domain comparison the paper doesn't publish is what
this work approximates; there's no perfect benchmark.

---

## 9. Failure modes and what I learned

The auto-generated error analysis catalogs ten failure categories with
concrete examples. The four that dominate the error budget:

**Multi-recipient over-assignment** (≈90% of FPs). Strong imperative +
`is_to=1` + nothing in the task to disambiguate which of N recipients
is meant. The model marks everyone. Independent binary classification
fundamentally can't solve this — there's no mechanism for "if Anna is
responsible, Brad is less likely". Listwise softmax with a no-one slot
is the structural fix.

**Sender first-person commitments** (~22% of FNs). "I'll handle this",
"let me look into it" — the sender is occasionally the responsible
party. The new `task_first_person_and_sender` cross-feature partially
addresses this, but cases that hinge on "to me" or "for me" still
require coreference the linear model can't do.

**Out-of-list third-party references** ("Please ask Jeff to contact
trader" with Jeff not on To/Cc). Pragmatics fire; the model defaults
to a To recipient. The right fix is NER over the task — if a `PERSON`
is mentioned that isn't on To/Cc, lower scores globally for the HIT.

**Implicit "you" disambiguation** ("Can you and Brad review this?").
The first-name match for Brad fires correctly; the "you" needs to be
resolved to the conjunct's co-recipient. Conjunction-aware addressee
tagging would close this.

What I learned from the error analysis: my features address the slices
they were designed for, but the dominant FP pattern (multi-recipient
over-assignment) is invariant to feature engineering — it's an
architectural ceiling. No amount of features fixes it. That's the kind
of finding that reframes the next iteration of work.

---

## 10. If I had more time

In rough priority order:

1. **Listwise scoring with a no-one slot.** Replace the per-candidate
   independent classifier with a softmax across candidates per HIT.
   This is the only structural fix for multi-recipient over-assignment
   and would meaningfully move the headline F1.

2. **Probability calibration** (`CalibratedClassifierCV` with isotonic
   regression on a held-out fold). Scores are well-ordered today but
   not calibrated to true probabilities; a product team setting an
   operating point is currently picking from a coarse grid.

3. **A small transformer head over the same features.** Distilled or
   frozen, fine-tuned only on the structured + task-text inputs. Worth
   shipping only after #1 and #2 land — the gain over LR has to be
   meaningful at comparable inference cost.

4. **Better discourse modeling.** Parse `----- Original Message -----`
   blocks into structured prior turns; add features over them. The
   paper's "no explicit person" case (Figure 2) needs this.

5. **Human disagreement modeling.** With α = 0.61 the labels themselves
   are noisy; a model that predicts P(consensus) and a separate
   P(annotator-i-agreement) might give a more honest confidence signal.
   Sample-weighting by judge agreement is the cheaper version.

---

## 11. Reviewer-facing design decisions

**Q. Why binary classification rather than ranking?**
Binary matches the paper, gives apples-to-apples comparisons, and
mirrors the production framing (one notification per `(person, task)`
pair). The known weakness — independent scoring can over-assign in
multi-recipient HITs — is documented and is the top future improvement.
A listwise formulation with a no-one slot would be a strict improvement
and is the right next step.

**Q. Why not BERT?**
The bottleneck isn't text understanding, it's relational reasoning
between candidate identity and task language. TF-IDF gives me topical
context cheaply; structured candidate features carry most of the
signal. A transformer would help on slices my features don't address
(coreference, conjunctions). I'd ship one only after listwise scoring
and calibration, and only if it materially beats LR at comparable
inference cost.

**Q. Why `GroupShuffleSplit`?**
Rows from the same HIT share all the email-level text. A row-level
random split would let the model see body/task at training time and
then "predict" on a row where everything except candidate identity is
identical. Group-aware splitting plus an explicit leakage assertion
ensures test scores reflect generalization to *unseen HITs*, not unseen
candidates within seen HITs.

**Q. Why duplicate the email body across candidate rows?**
Storage cost (~65 MB CSV) for two real benefits. The same row format
works at training and inference time, so the FastAPI service reuses
the exact same `Pipeline` — no parallel featurizer. Per-candidate
features (proximity, name-in-body) need both candidate identity and
body text on the same row for cheap computation. A production version
would normalize into HIT-level + candidate-level tables and join, but
that's premature here.

**Q. Why is the threshold tuned on validation rather than test?**
Picking a threshold from the test set's PR curve is selection-on-test
bias — the test F1 ends up optimistically biased toward the threshold
that happened to maximize it on that specific split. Tuning on val and
reporting test once at the chosen threshold gives an unbiased estimate
of how that operating point generalizes.

**Q. How would you improve precision without sacrificing recall?**
Two levers. Calibrated thresholds with a higher operating point —
trades recall for precision in a known way. Listwise scoring across
candidates per HIT — fixes the dominant multi-recipient FP pattern
without losing single-recipient performance. Calibration is the cheap
win; listwise is the big win.

**Q. How would you productionize this?**
The FastAPI service is the skeleton. Before deploy: auth + rate
limiting; structured logging + tracing + metrics; async wrapping of
sklearn inference; a batch endpoint. After deploy: probability
calibration; a feedback loop that captures user corrections and folds
them back as labelled data with sample weights; periodic retraining
gated by regression tests on the headline metric. The model bundle
already carries `git_sha`, `model_version`, and `trained_at` so version
identity is unambiguous.

**Q. What are the dataset's limitations?**
α = 0.61 inter-annotator agreement — labels are noisy by construction,
not by sloppy annotation. Only ~32 HITs are labelled "no-one
responsible" under majority consensus, which is suspiciously low for a
real product setting. Email body formatting is genuinely messy
(`<br/>`-fused sentences, internal Enron routing addresses, group
aliases). The task sentence is given pre-extracted, so the harder
upstream problem of detecting tasks isn't modeled here.

---

## 12. Final reflection

The strongest thing I took from this exercise is that applied ML
performance often comes more from correct problem framing,
leakage-safe evaluation, and task-specific feature design than from
model complexity. Once the framing was right — relational reasoning
over `(email, task, candidate)` triples, group-aware splits, validation-
only thresholds — a logistic regression with handcrafted features
landed within striking distance of where deep models would land
without any of the operational cost.

What's left to improve isn't a model-class problem; it's an
architectural one (independent binary scoring vs. listwise) and a
linguistic one (coreference, conjunctions). Naming those clearly
is more useful than chasing a small F1 gain on the wrong axis.

---

## Appendix A — How to run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Acquire the EPA dataset (tries url, then git clone, then synthetic)
python -m src.data.download

# 3. Build candidate-level training examples
python -m src.data.build_examples \
    --consensus majority    # or perfect | any

# 4. Train + auto-write evaluation.md, evaluation.json, error_analysis.md
python -m src.models.train

# 5. Re-evaluate a saved model without retraining
python -m src.models.evaluate

# 6. One-off prediction from a JSON payload
python -m src.models.predict --input example_payload.json

# 7. Inference service
uvicorn src.service.app:app --reload --port 8000

# 8. Tests
pytest tests/ -v

# 9. End-to-end verification (one command, every stage)
python verify.py
```

A typical end-to-end run takes ~30 seconds. The model artifact is
~2.4 MB; the candidate CSV is ~65 MB.

## Appendix B — Repo layout

```
epa-assignment/
├── README.md
├── requirements.txt
├── config.yaml
├── verify.py                  # end-to-end verification script
│
├── data/
│   ├── raw/                   # EPADataset.tsv goes here
│   └── processed/             # candidate-level examples.csv
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
│   │   └── app.py             # FastAPI inference
│   └── utils/
│       ├── io.py
│       └── text_cleaning.py
│
├── tests/
│   ├── test_parse_epa.py
│   ├── test_split.py
│   ├── test_features.py
│   ├── test_inference_parity.py
│   ├── test_service.py
│   └── test_serialization.py
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── reports/                   # all auto-regenerated by train.py
│   ├── evaluation.md
│   ├── evaluation.json
│   └── error_analysis.md
│
└── models/
    └── epa_model.joblib       # carries model_version, git_sha, trained_at
```

## Reference

Rameshkumar, R., Bailey, P., Jha, A., & Quirk, C. (2018).
*Assigning people to tasks identified in email: The EPA dataset for
addressee tagging for detected task intent.* W-NUT 2018, EMNLP.
[ACL Anthology W18-6104](https://aclanthology.org/W18-6104/).
Dataset: <https://github.com/RevRameshkumar/EPADataset>.
