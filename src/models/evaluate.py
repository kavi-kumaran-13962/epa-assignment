"""Stand-alone evaluation entry point.

Loads a saved model and a candidate-level CSV, recomputes overall +
scenario metrics + threshold sweep, and writes a Markdown report. Useful
for evaluating a previously trained model without retraining.

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
from src.models.split import group_train_test_split
from src.models.train import _read_examples, write_report
from src.utils.io import load_config, setup_logging

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_path = Path(args.data or cfg["paths"]["examples_csv"])
    model_path = Path(args.model or cfg["paths"]["model_path"])
    report_out = Path(args.report_out or Path(cfg["paths"]["reports_dir"]) / "evaluation.md")

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    threshold = float(args.threshold if args.threshold is not None else bundle.get(
        "default_threshold", cfg["evaluation"]["default_threshold"]
    ))

    df = _read_examples(data_path)
    train_df, val_df, test_df = group_train_test_split(
        df,
        group_key=cfg["split"]["group_key"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )
    y_test = test_df["label"].to_numpy()
    y_score = pipe.predict_proba(test_df)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    overall = compute_metrics(y_test, y_pred, y_score)

    every_pred = EveryRecipientBaseline().fit(train_df, train_df["label"]).predict(test_df)
    prior_clf = ClassPriorBaseline().fit(train_df, train_df["label"])
    prior_pred = prior_clf.predict(test_df, threshold=threshold)

    baselines = {
        "Every-recipient baseline": compute_metrics(y_test, every_pred),
        f"Class-prior baseline (p={prior_clf.prior_:.3f})": compute_metrics(
            y_test, prior_pred
        ),
    }
    scenarios = evaluate_scenarios(test_df, y_test, y_pred, y_score)
    sweep = threshold_sweep(y_test, y_score, list(cfg["evaluation"]["threshold_grid"]))

    write_report(report_out, cfg, overall, baselines, scenarios, sweep)
    json_path = report_out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "overall": overall,
                "baselines": baselines,
                "scenarios": scenarios.to_dict(orient="records"),
                "threshold_sweep": sweep.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote evaluation report to %s", report_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
