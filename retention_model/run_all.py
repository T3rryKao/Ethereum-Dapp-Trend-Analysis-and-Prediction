"""One-command Q5 workflow. Existing data, results and artifacts stay untouched."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import sklearn

import features
import labels
import splits
import train

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute_pipeline(output: Path, prediction_model=train.DEFAULT_PREDICTION_MODEL, weighted_training=False):
    """Generate inputs, then run the fixed chronological Train/Test experiment."""
    print("1. Generate M12 labels and audit tables", flush=True)
    raw = labels.load_data()
    labs = labels.build_labels(raw)
    audit, audit_summary = labels.audit_tables(raw, labs)
    labs.to_csv(output / "labels_m12.csv", index=False)
    audit.to_csv(output / "label_cohort_audit.csv", index=False)
    audit_summary.to_csv(output / "label_audit_summary.csv", index=False)
    print(audit_summary.to_string(index=False))
    print("2. Generate M1–M3 features and labeled dataset", flush=True)
    feats = features.early_features(raw)
    dataset = features.build_dataset(raw)
    feats.to_csv(output / "features_k3.csv", index=False)
    dataset.to_csv(output / "dataset_k3.csv", index=False)
    if weighted_training:
        from weighted_training import run_experiment
        print("3. Separate weighting experiment: training-only validation, then full-pool refit", flush=True)
        result = run_experiment(dataset, feats, output, prediction_model)
    else:
        print("3. Train/Test only: fit all seven fixed predictors, evaluate and save", flush=True)
        result = train.run_experiment(dataset, feats, output, prediction_model)
    print("Sparse zeros assume complete source coverage through December 2025.")
    print(f"Outputs: {output}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Fresh run directory; default runs_train_test/<UTC timestamp>")
    parser.add_argument("--prediction-model", choices=list(train.make_models()),
                        default=train.DEFAULT_PREDICTION_MODEL,
                        help="Predefined learner for scoring; default ridge, never test-selected")
    parser.add_argument("--weighted-training", action="store_true",
                        help="Separate weight-selection experiment with one training-only chronological validation block")
    args = parser.parse_args(argv)
    output = (args.output_dir or ROOT / ("runs_weighted" if args.weighted_training else "runs_train_test") /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite an existing run directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run_summary.txt").open("x") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"Q5 full workflow | seed={train.SEED} | run={output}", flush=True)
                result = execute_pipeline(output, args.prediction_model, weighted_training=args.weighted_training)
                result.update(seed=train.SEED, schema_version=splits.SCHEMA_VERSION,
                              workflow_version=result.get("experiment_workflow_version", splits.WORKFLOW_VERSION),
                              input_path=str(labels.DATA_PATH),
                              input_sha256=hashlib.sha256(labels.DATA_PATH.read_bytes()).hexdigest(),
                              python=platform.python_version(),
                              numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)
                (output / "run_metadata.json").write_text(json.dumps(result, indent=2))
                print("Q5 workflow completed successfully.")
            except Exception:
                traceback.print_exc()
                raise
    return output


if __name__ == "__main__":
    main()
