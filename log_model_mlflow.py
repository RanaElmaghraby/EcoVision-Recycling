
import json
from pathlib import Path

import mlflow
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "ecosort_artifacts"
MLARTIFACTS_DIR = PROJECT_DIR / "mlartifacts"
MLARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "EcoSort_EfficientNetV2S.keras"
CLASSES_PATH = ARTIFACTS_DIR / "class_names.json"
KNOWLEDGE_PATH = ARTIFACTS_DIR / "recycling_knowledge.json"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"

# Configure local MLflow tracking
DB_PATH = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

EXPERIMENT_NAME = "EcoSort_RecycleIR"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=MLARTIFACTS_DIR.resolve().as_uri()
    )

mlflow.set_experiment(EXPERIMENT_NAME)

# Load metadata
with open(CLASSES_PATH, encoding="utf-8") as f:
    class_names = json.load(f)

with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
    recycling_knowledge = json.load(f)

with open(HISTORY_PATH, encoding="utf-8") as f:
    history = json.load(f)

# Load existing model; no training
print("Loading saved model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)
print("Model loaded successfully!")

# Log to MLflow
with mlflow.start_run(run_name="EfficientNetV2S_existing_model") as run:
    mlflow.log_param("architecture", "EfficientNetV2S")
    mlflow.log_param("pretrained_weights", "ImageNet")
    mlflow.log_param("num_classes", len(class_names))
    mlflow.log_param("classes", ", ".join(class_names))
    mlflow.log_param("model_source", "Previously trained on Kaggle")
    mlflow.log_param("retraining_performed", False)
    mlflow.log_param("tensorflow_version", tf.__version__)

    epochs = max(
        (len(values) for values in history.values()),
        default=0
    )
    mlflow.log_param("epochs_trained", epochs)

    for metric_name, values in history.items():
        if values:
            mlflow.log_metric(
                f"final_{metric_name}",
                float(values[-1])
            )

    if history.get("val_accuracy"):
        mlflow.log_metric(
            "best_val_accuracy",
            float(max(history["val_accuracy"]))
        )

    if history.get("val_loss"):
        mlflow.log_metric(
            "best_val_loss",
            float(min(history["val_loss"]))
        )

    mlflow.tensorflow.log_model(
        model=model,
        name="waste_classifier"
    )

    mlflow.log_artifact(str(CLASSES_PATH), artifact_path="metadata")
    mlflow.log_artifact(str(KNOWLEDGE_PATH), artifact_path="metadata")
    mlflow.log_artifact(str(HISTORY_PATH), artifact_path="training_history")

    print("\nMLflow logging completed!")
    print("Run ID:", run.info.run_id)
    print("Experiment:", EXPERIMENT_NAME)
    print("Tracking URI:", mlflow.get_tracking_uri())


