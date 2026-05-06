"""Stand-alone evaluation entry point.

Loads a saved model and a candidate-level CSV, recomputes overall +
scenario metrics + threshold sweep on validation + auto error analysis,
and writes Markdown + JSON reports. Useful for evaluating a previously
trained model without retraining.

Methodologically identical to ``train.py``: threshold is selected on the
validation split, then the test split is consumed exactly once.

Usage::

    python -m src.models.evaluate \
        --data data/processed/examples.csv \
        --model models/epa_model.joblib \
        --report_out reports/evaluation.md
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import joblib

from src.models.baselines import ClassPriorBaseline, EveryRecipientBaseline
from src.models.metrics import (
    compute_metrics,
    evaluate_scenarios,
    threshold_sweep,
)
from src.models.split import assert_no_group_leakage, group_train_test_split
from src.models.train import (
    _read_examples,
    _summarize_test_errors,
    select_threshold_on_val,
    write_error_analysis,
    write_evaluation_report,
)
from src.utils.io import load_config, setup_logging

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--report_out", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the val-tuned threshold (use sparingly).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_path = Path(args.data or cfg["paths"]["examples_csv"])
    model_path = Path(args.model or cfg["paths"]["model_path"])
    report_out = Path(
        args.report_out or Path(cfg["paths"]["reports_dir"]) / "evaluation.md"
    )
    error_out = Path(cfg["paths"]["reports_dir"]) / "error_analysis.md"

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]

    df = _read_examples(data_path)
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

    # Threshold: prefer CLI override, then the bundle's tuned value, then
    # re-tune on validation as a fallback.
    if args.threshold is not None:
        chosen_threshold = float(args.threshold)
        # Build an empty sweep for report consistency.
        import pandas as pd  # local import to avoid module-level dep
        val_sweep = pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1", "accuracy"]
        )
    elif "default_threshold" in bundle:
        chosen_threshold = float(bundle["default_threshold"])
        # Recompute the val sweep so the report shows where this threshold
        # came from in the precision/recall trade-off curve.
        _, val_sweep = select_threshold_on_val(
            pipe, val_df, list(cfg["evaluation"]["threshold_grid"])
        )
    else:
        chosen_threshold, val_sweep = select_threshold_on_val(
            pipe, val_df, list(cfg["evaluation"]["threshold_grid"])
        )

    y_test = test_df["label"].to_numpy()
    y_score = pipe.predict_proba(test_df)[:, 1]
    y_pred = (y_score >= chosen_threshold).astype(int)
    overall = compute_metrics(y_test, y_pred, y_score)

    every_pred = (
        EveryRecipientBaseline().fit(train_df, train_df["label"]).predict(test_df)
    )
    prior_clf = ClassPriorBaseline().fit(train_df, train_df["label"])
    prior_pred = prior_clf.predict(test_df, threshold=chosen_threshold)

    baselines = {
        "Every-recipient baseline": compute_metrics(y_test, every_pred),
        f"Class-prior baseline (p={prior_clf.prior_:.3f})": compute_metrics(
            y_test, prior_pred
        ),
    }
    scenarios = evaluate_scenarios(test_df, y_test, y_pred, y_score)

    write_evaluation_report(
        report_out,
        cfg,
        chosen_threshold,
        overall,
        baselines,
        scenarios,
        val_sweep,
        (float("nan"), float("nan"), []),  # CV is a train.py artifact only
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
                "model_version": bundle.get("model_version"),
                "git_sha": bundle.get("git_sha"),
                "trained_at": bundle.get("trained_at"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote evaluation report to %s", report_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
