"""Train the main EPA model and report evaluation alongside baselines.

This script is the main training entry point::

    python -m src.models.train \
        --data data/processed/examples.csv \
        --model_out models/epa_model.joblib

It produces:

* the saved sklearn ``Pipeline`` (TF-IDF + structured features + LR);
* a Markdown evaluation report at ``reports/evaluation.md`` with overall
  metrics, per-scenario breakdown, threshold sweep on validation, and
  baseline comparison;
* a structured JSON summary at ``reports/evaluation.json``;
* an auto-generated error analysis at ``reports/error_analysis.md``
  drawn from the held-out test set so it stays in lockstep with the
  current model;
* a saved bundle that includes a model version, training timestamp,
  and (best-effort) git SHA for traceability.

Methodological safeguards (also documented in the README):

* Group-aware splitting on ``email_task_id`` (no candidate from the same
  HIT can appear in both train and test).
* Explicit ``assert_no_group_leakage`` after every split.
* The decision threshold is selected on the **validation set** only.
* The test set is consumed exactly once for final unbiased reporting.
* A grouped 5-fold CV F1 is reported (mean ± std) so the headline isn't
  a single point estimate.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from src.models.baselines import ClassPriorBaseline, EveryRecipientBaseline
from src.models.metrics import (
    compute_metrics,
    evaluate_scenarios,
    threshold_sweep,
)
from src.models.pipeline import build_pipeline
from src.models.split import assert_no_group_leakage, group_train_test_split
from src.utils.io import ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)


__model_version__ = "0.2.0"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_examples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col, default in [
        ("subject", ""),
        ("body", ""),
        ("task", ""),
        ("full_context", ""),
        ("candidate_email", ""),
        ("candidate_name", ""),
        ("sender_email", ""),
        ("sender_name", ""),
        ("candidate_role", ""),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
    for col in [
        "is_sender",
        "is_to",
        "is_cc",
        "num_to_recipients",
        "num_cc_recipients",
        "num_total_candidates",
        "no_one_responsible",
        "label",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "full_context" in df.columns and df["full_context"].str.len().sum() == 0:
        df["full_context"] = (
            df["subject"].astype(str)
            + " [SEP] "
            + df["task"].astype(str)
            + " [SEP] "
            + df["body"].astype(str)
        )
    return df


def _git_sha() -> Optional[str]:
    """Best-effort git SHA for traceability; returns None outside a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Threshold tuning on validation
# ---------------------------------------------------------------------------


def select_threshold_on_val(
    pipe: Any, val_df: pd.DataFrame, grid: List[float]
) -> Tuple[float, pd.DataFrame]:
    """Pick the threshold that maximises F1 on the validation split.

    Returns ``(chosen_threshold, sweep_dataframe)``. If the validation
    split is empty (e.g. ``val_size: 0`` in config) we fall back to 0.5.
    """
    if len(val_df) == 0:
        logger.warning("Empty validation split; defaulting threshold to 0.5.")
        empty = pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1", "accuracy"]
        )
        return 0.5, empty
    y_val = val_df["label"].to_numpy()
    y_score = pipe.predict_proba(val_df)[:, 1]
    sweep = threshold_sweep(y_val, y_score, grid)
    best_row = sweep.loc[sweep["f1"].idxmax()]
    return float(best_row["threshold"]), sweep


# ---------------------------------------------------------------------------
# Grouped 5-fold CV
# ---------------------------------------------------------------------------


def grouped_cv_f1(
    train_val_df: pd.DataFrame,
    cfg: Dict[str, Any],
    threshold: float,
    n_splits: int = 5,
) -> Tuple[float, float, List[float]]:
    """Run grouped k-fold CV on the training+val portion of the data.

    Used as a sanity check on the headline F1 — gives a mean ± std so the
    test-set point estimate isn't reported alone.
    """
    if len(train_val_df) < n_splits * 100:
        logger.info("Too few rows for %d-fold CV; skipping.", n_splits)
        return float("nan"), float("nan"), []

    groups = train_val_df[cfg["split"]["group_key"]].astype(str).to_numpy()
    y = train_val_df["label"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)

    f1s: List[float] = []
    for fold, (tr_idx, te_idx) in enumerate(
        splitter.split(train_val_df, y, groups=groups), start=1
    ):
        tr = train_val_df.iloc[tr_idx]
        te = train_val_df.iloc[te_idx]
        pipe, _ = build_pipeline(cfg)
        pipe.fit(tr, y[tr_idx])
        score = pipe.predict_proba(te)[:, 1]
        pred = (score >= threshold).astype(int)
        f = f1_score(y[te_idx], pred, zero_division=0)
        f1s.append(float(f))
        logger.info("CV fold %d/%d: F1 = %.4f", fold, n_splits, f)

    arr = np.asarray(f1s)
    return float(arr.mean()), float(arr.std()), f1s


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _format_metrics_block(name: str, m: Dict[str, Any]) -> str:
    keys = [
        "precision", "recall", "f1", "accuracy",
        "pr_auc", "roc_auc", "tp", "fp", "fn", "tn", "n",
    ]
    lines = [f"### {name}"]
    for k in keys:
        if k in m:
            v = m[k]
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def _df_to_md(df: pd.DataFrame, max_rows: int = 50) -> str:
    if len(df) == 0:
        return "_(no rows)_"
    df = df.head(max_rows)
    return df.to_markdown(index=False, floatfmt=".4f")


def write_evaluation_report(
    out: Path,
    cfg: Dict[str, Any],
    chosen_threshold: float,
    overall: Dict[str, Any],
    baselines: Dict[str, Dict[str, Any]],
    scenarios: pd.DataFrame,
    val_sweep: pd.DataFrame,
    cv_f1: Tuple[float, float, List[float]],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    parts: List[str] = []
    parts.append("# EPA evaluation report\n")
    parts.append(
        "Auto-generated by `src.models.train`. "
        "**Threshold tuned on validation, evaluated once on the held-out test set "
        f"at threshold = {chosen_threshold:.2f}.**\n"
    )

    parts.append("## Main model — Logistic Regression with handcrafted + TF-IDF features\n")
    parts.append(_format_metrics_block("Held-out test set", overall))

    cv_mean, cv_std, _ = cv_f1
    if not np.isnan(cv_mean):
        parts.append(
            f"\n**Grouped 5-fold CV F1 (train+val):** "
            f"{cv_mean:.4f} ± {cv_std:.4f}"
        )

    parts.append("\n## Baselines\n")
    for name, m in baselines.items():
        parts.append(_format_metrics_block(name, m))
        parts.append("")

    parts.append("\n## Scenario breakdown\n")
    parts.append(
        "Slices that mirror the paper's discussion of single- vs. "
        "multi-recipient performance, plus addressee-tagging cases the "
        "take-home asked us to look at explicitly.\n"
    )
    parts.append(
        _df_to_md(
            scenarios[
                [
                    c
                    for c in [
                        "scenario", "n_examples", "support_pos", "support_neg",
                        "precision", "recall", "f1", "pr_auc",
                    ]
                    if c in scenarios.columns
                ]
            ]
        )
    )

    parts.append("\n## Threshold sweep (validation set)\n")
    parts.append(
        "The decision threshold was selected from this sweep by argmax F1 on the "
        "**validation** set. The test set is used exactly once for unbiased reporting "
        "at the chosen threshold.\n"
    )
    parts.append(
        _df_to_md(
            val_sweep[
                [c for c in ["threshold", "precision", "recall", "f1", "accuracy"]
                 if c in val_sweep.columns]
            ]
        )
    )

    out.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", out)


# ---------------------------------------------------------------------------
# Auto-generated error analysis
# ---------------------------------------------------------------------------


_YOU_RE = re.compile(r"\byou\b|\byour\b", re.IGNORECASE)


def _summarize_test_errors(
    test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray,
) -> Dict[str, Any]:
    df = test_df.copy()
    df["score"] = y_score
    df["pred"] = y_pred
    df["true"] = y_true

    fp = df[(df.pred == 1) & (df.true == 0)]
    fn = df[(df.pred == 0) & (df.true == 1)]
    multi = df["num_total_candidates"] > 2

    n_fp = len(fp)
    n_fn = len(fn)

    fp_multi = int(((df.pred == 1) & (df.true == 0) & multi).sum())
    fp_single = n_fp - fp_multi
    fn_sender = int(((df.pred == 0) & (df.true == 1) & (df.is_sender == 1)).sum())
    fn_cc = int(((df.pred == 0) & (df.true == 1) & (df.is_cc == 1)).sum())
    fp_you = int(
        ((df.pred == 1) & (df.true == 0)
         & df["task"].fillna("").str.contains(_YOU_RE)).sum()
    )

    top_fp = fp.sort_values("score", ascending=False).head(6)
    top_fn = fn.sort_values("score").head(6)

    return {
        "n_test": len(df),
        "n_fp": n_fp,
        "n_fn": n_fn,
        "fp_multi": fp_multi,
        "fp_single": fp_single,
        "fn_sender": fn_sender,
        "fn_cc": fn_cc,
        "fp_you": fp_you,
        "top_fp": top_fp,
        "top_fn": top_fn,
    }


def _format_examples(examples: pd.DataFrame) -> str:
    if len(examples) == 0:
        return "_(none)_"
    rows = ["| score | candidate | role | multi? | task |", "|---:|---|---|:---:|---|"]
    for _, r in examples.iterrows():
        multi = "✓" if r.get("num_total_candidates", 0) > 2 else "–"
        task = (str(r["task"])[:90] + "…") if len(str(r["task"])) > 90 else str(r["task"])
        # Escape pipes in task text so it doesn't break the markdown table.
        task = task.replace("|", "\\|")
        rows.append(
            f"| {r['score']:.3f} | {r['candidate_email']} | "
            f"{r['candidate_role']} | {multi} | {task} |"
        )
    return "\n".join(rows)


def write_error_analysis(
    out: Path,
    overall: Dict[str, Any],
    summary: Dict[str, Any],
    chosen_threshold: float,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n_test = summary["n_test"]
    n_fp = summary["n_fp"]
    n_fn = summary["n_fn"]
    pos = overall.get("support_pos", 0)
    neg = overall.get("support_neg", 0)
    fp_pct_neg = 100.0 * n_fp / max(neg, 1)
    fn_pct_pos = 100.0 * n_fn / max(pos, 1)
    fp_multi_pct = 100.0 * summary["fp_multi"] / max(n_fp, 1)
    fn_sender_pct = 100.0 * summary["fn_sender"] / max(n_fn, 1)
    fp_you_pct = 100.0 * summary["fp_you"] / max(n_fp, 1)

    parts: List[str] = []
    parts.append("# Error analysis\n")
    parts.append(
        "Auto-regenerated by `src.models.train` from the latest model. "
        f"Numbers below are computed on the held-out test split at threshold "
        f"= {chosen_threshold:.2f}.\n"
    )

    parts.append("## Headline numbers\n")
    parts.append(
        f"| metric | value |\n|---|---|\n"
        f"| precision | {overall.get('precision', float('nan')):.3f} |\n"
        f"| recall | {overall.get('recall', float('nan')):.3f} |\n"
        f"| F1 | {overall.get('f1', float('nan')):.3f} |\n"
        f"| ROC-AUC | {overall.get('roc_auc', float('nan')):.3f} |\n"
        f"| PR-AUC  | {overall.get('pr_auc', float('nan')):.3f} |\n"
        f"| support | {n_test:,} ({pos:,}+ / {neg:,}−) |\n"
    )
    parts.append(
        f"Confusion: TP = {overall.get('tp', 0):,}, "
        f"FP = {overall.get('fp', 0):,}, "
        f"FN = {overall.get('fn', 0):,}, "
        f"TN = {overall.get('tn', 0):,}.\n"
    )

    parts.append("## Where the errors live\n")
    parts.append(
        f"| bucket | count | fraction of bucket | comment |\n"
        f"|---|---:|---:|---|\n"
        f"| Total FPs | {n_fp:,} | {fp_pct_neg:.0f} % of negatives | over-assigning to a recipient |\n"
        f"| Total FNs | {n_fn:,} | {fn_pct_pos:.0f} % of positives | missing a responsible person |\n"
        f"| FPs in multi-recipient HITs | {summary['fp_multi']:,} | "
        f"**{fp_multi_pct:.0f} %** of FPs | dominant FP pattern |\n"
        f"| FPs in single-recipient HITs | {summary['fp_single']:,} | "
        f"{100 - fp_multi_pct:.0f} % of FPs | tiny once the structural signal locks in |\n"
        f"| FNs where candidate is sender | {summary['fn_sender']:,} | "
        f"{fn_sender_pct:.0f} % of FNs | senders committing themselves |\n"
        f"| FNs where candidate is Cc | {summary['fn_cc']:,} | "
        f"{100.0 * summary['fn_cc'] / max(n_fn, 1):.0f} % of FNs | "
        f"multi-recipient miss; \"and X\" elision |\n"
        f"| FPs where task contains \"you/your\" | {summary['fp_you']:,} | "
        f"{fp_you_pct:.0f} % of FPs | implicit-pronoun overreach |\n"
    )

    parts.append("\n## Top false positives (highest score, true label 0)\n")
    parts.append(_format_examples(summary["top_fp"]))

    parts.append("\n## Top false negatives (lowest score, true label 1)\n")
    parts.append(_format_examples(summary["top_fn"]))

    parts.append(
        "\n## Error categories and concrete next steps\n"
        "\n"
        "| # | Error type | Example pattern | Likely cause | Possible improvement |\n"
        "|---|---|---|---|---|\n"
        "| 1 | Multi-recipient over-assignment | imperative + 3+ To-recipients; we mark all | "
        "pragmatics + role flags both fire; nothing tells the model only one is meant | "
        "listwise softmax across candidates per HIT (with a no-one slot) |\n"
        "| 2 | Sender first-person commitment | \"I'll handle this\", \"let me look\" | "
        "is_sender prior is strongly negative | "
        "added `task_first_person_*` features in this version; "
        "watch the new sender-FN rate |\n"
        "| 3 | Out-of-list third-party reference | \"Please ask Jeff to contact trader\" "
        "(Jeff not on To/Cc) | model defaults to To recipient | "
        "NER over the task; if a PERSON isn't on To/Cc, lower scores globally |\n"
        "| 4 | Conjunction \"X and you\" | \"Can you and Brad review this?\" | "
        "first-name match for Brad fires; \"you\" still unresolved | "
        "small dependency parse; resolve `you` from conjunct + To list |\n"
        "| 5 | Implicit plural \"you\" | \"Please complete a draft\" with multi-To | "
        "no signal disambiguating which subset | listwise scoring (same as #1) |\n"
        "| 6 | Sentence-segmentation noise | runs of `<br/>` fuse two sentences | "
        "proximity features misfire | better sentence segmenter on body |\n"
        "| 7 | Group / alias recipient | `webmaster@…`, `sales@…` | "
        "no real first/last name | detect aliases; expand via directory or skip |\n"
    )

    parts.append(
        "\n## What we'd prioritise next\n"
        "1. **Listwise scoring with a no-one slot.** Largest leverage on multi-recipient FPs.\n"
        "2. **Calibrate scores** (`CalibratedClassifierCV` isotonic on a held-out fold) "
        "before exposing thresholds to product surfaces.\n"
        "3. **Out-of-candidate-list NER** for the no-one cases.\n"
        "4. **Conjunction-aware addressee tagging** for \"X and you\" patterns.\n"
    )

    out.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Wrote auto-generated error analysis to %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model_out", default=None)
    parser.add_argument("--report_out", default=None)
    parser.add_argument(
        "--skip_cv",
        action="store_true",
        help="Skip 5-fold cross-validation (saves ~30 seconds).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_path = Path(args.data or cfg["paths"]["examples_csv"])
    model_out = Path(args.model_out or cfg["paths"]["model_path"])
    report_out = Path(
        args.report_out or Path(cfg["paths"]["reports_dir"]) / "evaluation.md"
    )
    error_out = Path(cfg["paths"]["reports_dir"]) / "error_analysis.md"

    df = _read_examples(data_path)
    if len(df) == 0:
        raise SystemExit(f"No examples found in {data_path}")
    logger.info(
        "Loaded %d candidate rows (%d positive, prior=%.4f)",
        len(df),
        int(df["label"].sum()),
        df["label"].mean(),
    )

    train_df, val_df, test_df = group_train_test_split(
        df,
        group_key=cfg["split"]["group_key"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )
    assert_no_group_leakage(
        train_df, val_df, test_df, group_key=cfg["split"]["group_key"]
    )
    logger.info(
        "Group split sizes: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    pipe, _ = build_pipeline(cfg)
    pipe.fit(train_df, y_train)

    # Threshold selection on VALIDATION (not test).
    chosen_threshold, val_sweep = select_threshold_on_val(
        pipe, val_df, list(cfg["evaluation"]["threshold_grid"])
    )
    logger.info("Selected threshold = %.2f from validation sweep", chosen_threshold)

    # Single test-set evaluation at the chosen threshold.
    y_score = pipe.predict_proba(test_df)[:, 1]
    y_pred = (y_score >= chosen_threshold).astype(int)
    overall = compute_metrics(y_test, y_pred, y_score)

    # Baselines on the same test split for sanity.
    every = EveryRecipientBaseline().fit(train_df, y_train)
    every_pred = every.predict(test_df)
    every_score = every.predict_proba(test_df)[:, 1]
    prior = ClassPriorBaseline().fit(train_df, y_train)
    prior_pred = prior.predict(test_df, threshold=chosen_threshold)
    prior_score = prior.predict_proba(test_df)[:, 1]
    baselines: Dict[str, Dict[str, Any]] = {
        "Every-recipient baseline": compute_metrics(y_test, every_pred, every_score),
        f"Class-prior baseline (p={prior.prior_:.3f})": compute_metrics(
            y_test, prior_pred, prior_score
        ),
    }

    scenarios = evaluate_scenarios(test_df, y_test, y_pred, y_score)

    # Optional grouped 5-fold CV on train+val for variance estimate.
    cv_mean, cv_std, cv_folds = float("nan"), float("nan"), []
    if not args.skip_cv:
        try:
            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            cv_mean, cv_std, cv_folds = grouped_cv_f1(
                train_val_df, cfg, chosen_threshold, n_splits=5
            )
            logger.info("Grouped 5-fold CV F1: %.4f ± %.4f", cv_mean, cv_std)
        except Exception as e:  # noqa: BLE001
            logger.warning("CV skipped due to error: %s", e)

    # Persist the model bundle with version + traceability metadata.
    ensure_dir(model_out.parent)
    bundle = {
        "pipeline": pipe,
        "config": cfg,
        "feature_columns": list(test_df.columns),
        "default_threshold": chosen_threshold,
        "training_class_prior": float(prior.prior_),
        # Traceability.
        "model_version": __model_version__,
        "git_sha": _git_sha(),
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(bundle, model_out)
    logger.info("Saved model to %s", model_out)

    # Reports.
    write_evaluation_report(
        report_out, cfg, chosen_threshold, overall, baselines, scenarios, val_sweep,
        (cv_mean, cv_std, cv_folds),
    )

    summary = _summarize_test_errors(test_df, y_test, y_pred, y_score)
    write_error_analysis(error_out, overall, summary, chosen_threshold)

    json_path = report_out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "overall": overall,
                "baselines": baselines,
                "scenarios": scenarios.to_dict(orient="records"),
                "validation_threshold_sweep": val_sweep.to_dict(orient="records"),
                "chosen_threshold": chosen_threshold,
                "training_class_prior": float(prior.prior_),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "cv_f1_mean": cv_mean,
                "cv_f1_std": cv_std,
                "cv_f1_folds": cv_folds,
                "model_version": __model_version__,
                "git_sha": bundle["git_sha"],
                "trained_at": bundle["trained_at"],
                "sklearn_version": bundle["sklearn_version"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote summary JSON to %s", json_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
