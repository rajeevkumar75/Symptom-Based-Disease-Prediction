import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from src.config.configuration import Configuration
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logger


app = Flask(__name__, template_folder="templates", static_folder="templates/static")

_DEFAULT_CONFIG_PATH = "config/config.yaml"
_MAX_SYMPTOMS = int(os.getenv("APP_MAX_SYMPTOMS", "64"))


def _normalize_symptom(value: str) -> str:
    value = (value or "").strip().lower()
    # Normalize common separators to match training column naming.
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


@dataclass(frozen=True)
class _Artifacts:
    prediction_pipeline: PredictionPipeline
    feature_columns: Tuple[str, ...]
    symptom_lookup: Dict[str, str]  # normalized -> canonical column name


@lru_cache(maxsize=1)
def _load_artifacts() -> _Artifacts:
    config_path = os.getenv("APP_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    config = Configuration(config_path)
    prediction_pipeline = PredictionPipeline(config)

    transformation_config = config.get_data_transformation_config()
    validation_config = config.get_data_validation_config()

    cleaned_data_path = transformation_config["cleaned_data_path"]
    target_column = validation_config["target_column"]

    df = pd.read_csv(cleaned_data_path)
    feature_columns = tuple(c for c in df.columns if c != target_column)
    symptom_lookup = {_normalize_symptom(c): c for c in feature_columns}

    logger.info(
        "Artifacts loaded successfully",
        extra={
            "config_path": str(config_path),
            "cleaned_data_path": str(cleaned_data_path),
            "feature_count": len(feature_columns),
        },
    )
    return _Artifacts(
        prediction_pipeline=prediction_pipeline,
        feature_columns=feature_columns,
        symptom_lookup=symptom_lookup,
    )


def _coerce_symptoms(payload: Any) -> List[str]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = payload.get("symptoms", [])
    if isinstance(payload, str):
        # Allow comma-separated string as a convenience.
        return [s.strip() for s in payload.split(",") if s.strip()]
    if isinstance(payload, (list, tuple)):
        return [str(s).strip() for s in payload if str(s).strip()]
    return []


def _symptoms_to_features(symptoms: Iterable[str], artifacts: _Artifacts) -> Tuple[pd.DataFrame, List[str], List[str]]:
    row: Dict[str, int] = {col: 0 for col in artifacts.feature_columns}
    recognized: List[str] = []
    unrecognized: List[str] = []

    seen = set()
    for s in symptoms:
        normalized = _normalize_symptom(s)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        col = artifacts.symptom_lookup.get(normalized)
        if col is None:
            unrecognized.append(s)
            continue
        row[col] = 1
        recognized.append(col)

    return pd.DataFrame([row]), recognized, unrecognized


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    try:
        artifacts = _load_artifacts()
        return jsonify(
            {
                "status": "ok",
                "feature_count": len(artifacts.feature_columns),
            }
        )
    except Exception:
        logger.exception("Health check failed (artifacts not ready)")
        return jsonify({"status": "error", "message": "Artifacts not ready. Run training pipeline first."}), 503


@app.route("/symptoms", methods=["GET"])
def symptoms():
    try:
        artifacts = _load_artifacts()
        return jsonify(
            {
                "count": len(artifacts.feature_columns),
                "symptoms": list(artifacts.feature_columns),
            }
        )
    except Exception:
        logger.exception("Failed to load symptoms list")
        return jsonify({"error": "Could not load symptoms list"}), 500


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"error": "Invalid request", "details": "Malformed JSON or bad parameters"}), 400


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) if request.is_json else None
        symptoms_list = _coerce_symptoms(payload)
        if not symptoms_list and request.form:
            # Optional: accept HTML form submission with repeated fields or a single comma-separated field.
            symptoms_list = request.form.getlist("symptoms") or _coerce_symptoms(request.form.get("symptoms"))

        if not symptoms_list:
            return jsonify({"error": "No symptoms provided"}), 400
        if len(symptoms_list) > _MAX_SYMPTOMS:
            return jsonify({"error": f"Too many symptoms provided (max {_MAX_SYMPTOMS})"}), 400

        artifacts = _load_artifacts()
        features_df, recognized, unrecognized = _symptoms_to_features(symptoms_list, artifacts)
        if not recognized:
            return (
                jsonify(
                    {
                        "error": "No provided symptoms matched the model feature set",
                        "unrecognized_symptoms": unrecognized,
                    }
                ),
                400,
            )

        predictions = artifacts.prediction_pipeline.predict(features_df)
        disease = predictions[0] if len(predictions) else None

        return jsonify(
            {
                "disease": None if disease is None else str(disease),
                "recognized_symptoms": recognized,
                "unrecognized_symptoms": unrecognized,
                "symptom_count": len(symptoms_list),
            }
        )
    except BadRequest:
        raise
    except Exception:
        logger.exception("Prediction request failed")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "5000"))
    app.run(host=host, port=port, debug=debug)
