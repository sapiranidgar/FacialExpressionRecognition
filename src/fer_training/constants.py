"""Constants specific to fer_training: Kaggle settings, result naming, and tuning.

Constants shared with other packages (model-type identifiers, metrics-dict
keys, etc.) live in common.constants instead - this module is strictly for
fer_training's own orchestration layer.
"""

from __future__ import annotations

from common.constants import MODEL_TYPE_RESNET, MODEL_TYPE_SIMPLE_CNN

# --- Repo-relative paths ---
CONFIG_FILE_RELATIVE_PATH = "configs/config.yaml"
ENV_FILE_NAME = ".env"

# --- Dataset directory convention ---
FER2013_SUBDIR_NAME = "fer2013"
TRAIN_SUBDIR_NAME = "train"
TEST_SUBDIR_NAME = "test"

# --- Kaggle dataset + credentials ---
KAGGLE_DATASET = "msambare/fer2013"
KAGGLE_CONFIG_DIR_NAME = ".kaggle"
KAGGLE_CONFIG_FILE_NAME = "kaggle.json"
KAGGLE_USERNAME_ENV_VAR = "KAGGLE_USERNAME"
KAGGLE_KEY_ENV_VAR = "KAGGLE_KEY"

# --- Result naming for the clean-accuracy comparison (evaluate_clean.py) ---
CLEAN_EVAL_SUBDIR = "clean_eval"
WINNER_FILE_NAME = "winner.json"
SIMPLE_CNN_RESULT_NAME = "simple_cnn"
RESNET_RESULT_NAME = "resnet18_finetuned"
CONFUSION_MATRIX_FILE_SUFFIX = "_confusion_matrix.png"
PER_CLASS_METRICS_CSV_SUFFIX = "_per_class_metrics.csv"
PER_CLASS_METRICS_PLOT_SUFFIX = "_per_class_metrics.png"
TRAINING_CURVES_FILE_SUFFIX = "_training_curves.png"

# --- Hyperparameter grid search (fer_training.tuning) ---
TUNING_SUBDIR = "tuning"
TUNING_RESULTS_CSV_SUFFIX = "_grid_search_results.csv"
DEFAULT_TRIAL_EPOCHS = 8
TRIAL_EARLY_STOPPING_PATIENCE = 3
TRIAL_LR_SCHEDULER_PATIENCE = 2
TRIAL_LR_SCHEDULER_FACTOR = 0.5
CONFIG_SECTION_SIMPLE_CNN = "simple_cnn"
CONFIG_SECTION_RESNET_FINETUNE = "resnet_finetune"
MODEL_TYPE_TO_CONFIG_SECTION = {
    MODEL_TYPE_SIMPLE_CNN: CONFIG_SECTION_SIMPLE_CNN,
    MODEL_TYPE_RESNET: CONFIG_SECTION_RESNET_FINETUNE,
}
CONFIG_KEY_LR = "lr"
CONFIG_KEY_WEIGHT_DECAY = "weight_decay"
