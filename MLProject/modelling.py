from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ===========
# Konfigurasi
# ===========
LOGGER = logging.getLogger(__name__)

TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "Credit Scoring - CI"
)
DATA_DIR_NAME: str = "namadataset_preprocessing"
TARGET_COLUMN: str = "credit_risk"
RANDOM_STATE: int = 42


def resolve_data_dir(dir_name: str = DATA_DIR_NAME) -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (Path(dir_name), here / dir_name, here.parent / dir_name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Folder '{dir_name}' tidak ditemukan.")


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data_dir = resolve_data_dir()
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    LOGGER.info("Data dimuat — latih: %s, uji: %s.", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def configure_tracking() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    LOGGER.info("MLflow Tracking URI: %s | Eksperimen: %s",
                TRACKING_URI, EXPERIMENT_NAME)


def train() -> None:
    configure_tracking()

    X_train, y_train, X_test, y_test = load_dataset()

    with mlflow.start_run(run_name="random_forest_basic"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": model.n_estimators,
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as artifact_dir:
            artifact_path = Path(artifact_dir)
            report_path = artifact_path / "classification_report.txt"
            report_path.write_text(
                classification_report(y_test, predictions), encoding="utf-8"
            )
            mlflow.log_artifact(str(report_path), "evaluation")

            confusion_path = artifact_path / "confusion_matrix.csv"
            pd.DataFrame(confusion_matrix(y_test, predictions)).to_csv(
                confusion_path, index=False
            )
            mlflow.log_artifact(str(confusion_path), "evaluation")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(1),
        )
        LOGGER.info("Akurasi set uji: %.4f", metrics["accuracy"])

    LOGGER.info("Pelatihan Basic selesai. Buka %s untuk melihat dashboard.",
                TRACKING_URI)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()


if __name__ == "__main__":
    main()
