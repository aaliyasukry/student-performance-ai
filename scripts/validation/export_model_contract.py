from pathlib import Path
import json
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OVERALL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "overall_random_forest_academic_only.joblib"
)

SUBJECT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "subject_random_forest_academic_only.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "integration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_object(obj, name):
    report = {
        "name": name,
        "python_type": str(type(obj)),
    }

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Type:", type(obj))

    # Dictionary-based saved artifact
    if isinstance(obj, dict):
        print("\nDictionary keys:")

        for key in obj.keys():
            print(f" - {key}")

        report["dictionary_keys"] = list(obj.keys())

        useful_keys = [
            "features",
            "feature_names",
            "numeric_features",
            "categorical_features",
            "model",
            "pipeline",
            "preprocessor",
            "metadata",
        ]

        for key in useful_keys:
            if key in obj:
                value = obj[key]

                print()
                print(f"{key}:")
                print(value)

                if isinstance(value, (list, tuple)):
                    report[key] = list(value)
                elif isinstance(value, dict):
                    report[key] = value
                else:
                    report[f"{key}_type"] = str(type(value))

        estimator = (
            obj.get("pipeline")
            or obj.get("model")
        )

    else:
        estimator = obj

    # Scikit-learn feature_names_in_
    if hasattr(estimator, "feature_names_in_"):
        features = list(estimator.feature_names_in_)

        print()
        print("feature_names_in_:")
        for i, feature in enumerate(features, start=1):
            print(f"{i:03d}. {feature}")

        report["feature_names_in_"] = features
        report["feature_count"] = len(features)

    # Pipeline inspection
    if hasattr(estimator, "named_steps"):
        print()
        print("Pipeline steps:")

        pipeline_steps = {}

        for step_name, step in estimator.named_steps.items():
            print(
                f" - {step_name}: "
                f"{type(step).__module__}.{type(step).__name__}"
            )

            pipeline_steps[step_name] = (
                f"{type(step).__module__}."
                f"{type(step).__name__}"
            )

        report["pipeline_steps"] = pipeline_steps

        # Try finding features from pipeline steps
        for step_name, step in estimator.named_steps.items():

            if hasattr(step, "feature_names_in_"):
                step_features = list(step.feature_names_in_)

                print()
                print(
                    f"feature_names_in_ from step "
                    f"'{step_name}':"
                )

                for i, feature in enumerate(
                    step_features,
                    start=1
                ):
                    print(f"{i:03d}. {feature}")

                report[
                    f"{step_name}_feature_names_in"
                ] = step_features

    # Raw estimator parameter inspection
    if hasattr(estimator, "get_params"):

        params = estimator.get_params(deep=False)

        selected_params = {}

        interesting = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "random_state",
            "n_jobs",
        ]

        for key in interesting:
            if key in params:
                selected_params[key] = params[key]

        if selected_params:
            print()
            print("Selected model parameters:")

            for key, value in selected_params.items():
                print(f" - {key}: {value}")

            report["selected_parameters"] = (
                selected_params
            )

    return report


def main():

    if not OVERALL_MODEL.exists():
        raise FileNotFoundError(
            f"Overall model not found:\n{OVERALL_MODEL}"
        )

    if not SUBJECT_MODEL.exists():
        raise FileNotFoundError(
            f"Subject model not found:\n{SUBJECT_MODEL}"
        )

    print("Loading frozen production models...")

    overall = joblib.load(OVERALL_MODEL)
    subject = joblib.load(SUBJECT_MODEL)

    overall_report = inspect_object(
        overall,
        "OVERALL RANDOM FOREST"
    )

    subject_report = inspect_object(
        subject,
        "SUBJECT RANDOM FOREST"
    )

    output = {
        "overall_model": overall_report,
        "subject_model": subject_report,
    }

    output_path = (
        OUTPUT_DIR
        / "model_feature_contract.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            default=str
        )

    print()
    print("=" * 80)
    print("MODEL CONTRACT EXPORT COMPLETE")
    print("=" * 80)

    print()
    print("Saved report:")
    print(output_path)


if __name__ == "__main__":
    main()
