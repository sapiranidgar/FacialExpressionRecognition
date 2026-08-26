"""Constants used only by the scripts/ entry points: JSON keys, plot titles, and print templates.

Constants shared with the fer package itself (dataset id, model type
identifiers, filenames, etc.) live in fer.constants instead - this module is
strictly for the orchestration/reporting layer that scripts/*.py adds on top.
"""

from __future__ import annotations

# --- results/winner.json schema (evaluate_clean.py) ---
WINNER_KEY_VICTIM = "victim"
WINNER_KEY_SUBSTITUTE = "substitute"
WINNER_KEY_METRICS = "metrics"

# --- per-class metrics table column (evaluate_clean.py) ---
PER_CLASS_TABLE_KEY_CLASS = "class"

# --- plot titles (evaluate_clean.py, train_cnn.py, train_resnet.py) ---
CONFUSION_MATRIX_TITLE_TEMPLATE = "{name} Confusion Matrix"
PER_CLASS_METRICS_TITLE_TEMPLATE = "{name} Per-Class Precision / Recall / Accuracy"
SIMPLE_CNN_TRAINING_TITLE = "Simple CNN Training"
RESNET_TRAINING_TITLE = "ResNet18 Fine-tune Training"

# --- training-curves filename prefix (train_resnet.py; kept distinct from
# fer.constants.RESNET_RESULT_NAME, which names a different file family) ---
RESNET_TRAINING_CURVES_PREFIX = "resnet18"

# --- console messages (evaluate_clean.py) ---
MSG_MODEL_SUMMARY_TEMPLATE = "\n{name}: overall accuracy={accuracy:.4f} f1_macro={f1_macro:.4f}"
MSG_VICTIM_TEMPLATE = "\nVictim model (best clean accuracy): {victim}"
MSG_SUBSTITUTE_TEMPLATE = "Substitute model (for phase-2 transfer attack): {substitute}"
MSG_WROTE_FILE_TEMPLATE = "Wrote {path}"

# --- console messages (train_cnn.py, train_resnet.py) ---
MSG_SIMPLE_CNN_TEST_METRICS_TEMPLATE = "Simple CNN test accuracy: {accuracy:.4f}, f1_macro: {f1_macro:.4f}"
MSG_RESNET_TEST_METRICS_TEMPLATE = "ResNet18 test accuracy: {accuracy:.4f}, f1_macro: {f1_macro:.4f}"

# --- step-progress messages (train_resnet.py) ---
MSG_BUILT_LOADERS = "built loaders"
MSG_BUILT_MODEL = "built model"
MSG_BUILT_TRAINER = "built trainer"
MSG_STARTED_TRAINING = "started training"
MSG_FINISHED_TRAINING = "finished training"
MSG_EVALUATING_MODEL = "evaluating model"
