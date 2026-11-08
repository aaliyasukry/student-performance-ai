from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "final"
DATA_DIR = PROJECT_ROOT / "data" / "model_ready"
FINAL_TEST_PERIOD = "2026_T2"


def metrics(actual, predicted):
    error = predicted - actual
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rows": len(actual),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": 1 - ss_res / ss_tot,
        "Within_10_pct": float(np.mean(np.abs(error) <= 10) * 100),
    }


def evaluate(model_file, data_file, target, select_final_test):
    artifact = joblib.load(MODEL_DIR / model_file)
    pipeline = artifact["pipeline"]
    data = pd.read_csv(DATA_DIR / data_file)
    data = select_final_test(data)
    data = data[data[target].notna()]
    features = list(pipeline.feature_names_in_)
    predicted = pipeline.predict(data[features])
    return metrics(data[target].to_numpy(dtype=float), predicted)


def main():
    results = {
        "Overall average": evaluate(
            "overall_random_forest_academic_only.joblib",
            "overall_prediction_dataset.csv",
            "target_calculated_average",
            lambda d: d[d["academic_period"] == FINAL_TEST_PERIOD],
        ),
        "Pooled subject mark": evaluate(
            "subject_random_forest_academic_only.joblib",
            "subject_prediction_dataset.csv",
            "target_mark",
            lambda d: d[d["dataset_split"] == "FINAL_TEST"],
        ),
    }

    print(f"Final out-of-time test: {FINAL_TEST_PERIOD}")
    for name, m in results.items():
        print(
            f"{name:<22} rows={m['rows']:>5}  MAE={m['MAE']:.4f}  "
            f"RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}  "
            f"within_10={m['Within_10_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
