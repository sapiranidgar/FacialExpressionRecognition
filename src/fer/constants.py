"""Central registry of constants shared across the fer package.

Every literal that has meaning beyond its immediate line - filenames, dataset
identifiers, environment variable names, default hyperparameters not sourced
from config.yaml, and repeated plotting parameters - lives here instead of
being duplicated or hidden inline in the modules that use it.
"""

from __future__ import annotations

# --- Devices ---
DEVICE_CPU = "cpu"

# --- Repo-relative paths ---
CONFIG_FILE_RELATIVE_PATH = "configs/config.yaml"
ENV_FILE_NAME = ".env"

# --- Kaggle dataset + credentials ---
KAGGLE_DATASET = "msambare/fer2013"
KAGGLE_CONFIG_DIR_NAME = ".kaggle"
KAGGLE_CONFIG_FILE_NAME = "kaggle.json"
KAGGLE_USERNAME_ENV_VAR = "KAGGLE_USERNAME"
KAGGLE_KEY_ENV_VAR = "KAGGLE_KEY"

# --- Model type identifiers (used to select transforms/dataloaders/checkpoints) ---
MODEL_TYPE_SIMPLE_CNN = "simple_cnn"
MODEL_TYPE_RESNET = "resnet"

# --- Result naming for the clean-accuracy comparison (evaluate_clean.py) ---
CLEAN_EVAL_SUBDIR = "clean_eval"
WINNER_FILE_NAME = "winner.json"
SIMPLE_CNN_RESULT_NAME = "simple_cnn"
RESNET_RESULT_NAME = "resnet18_finetuned"
CONFUSION_MATRIX_FILE_SUFFIX = "_confusion_matrix.png"
PER_CLASS_METRICS_CSV_SUFFIX = "_per_class_metrics.csv"
PER_CLASS_METRICS_PLOT_SUFFIX = "_per_class_metrics.png"
TRAINING_CURVES_FILE_SUFFIX = "_training_curves.png"

# --- Image normalization ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAYSCALE_NORM_MEAN = [0.5]
GRAYSCALE_NORM_STD = [0.5]

# --- SimpleCNN architecture ---
SIMPLE_CNN_CHANNELS = (32, 64, 128, 256)
SIMPLE_CNN_CONV_KERNEL_SIZE = 3
SIMPLE_CNN_CONV_PADDING = 1
SIMPLE_CNN_POOL_SIZE = 2
SIMPLE_CNN_DEFAULT_DROPOUT = 0.3

# --- ResNet18 fine-tuning defaults ---
RESNET_DEFAULT_UNFROZEN_LAYERS = ["layer4", "fc"]

# --- compute_metrics() return dict keys (the contract between
# fer.training.metrics and every caller that reads its output) ---
METRICS_KEY_ACCURACY = "accuracy"
METRICS_KEY_F1_MACRO = "f1_macro"
METRICS_KEY_CONFUSION_MATRIX = "confusion_matrix"
METRICS_KEY_PER_CLASS = "per_class"
PER_CLASS_KEY_PRECISION = "precision"
PER_CLASS_KEY_RECALL = "recall"
PER_CLASS_KEY_ACCURACY = "accuracy"
PER_CLASS_KEY_SUPPORT = "support"

# --- Plotting ---
CONFUSION_MATRIX_FIGSIZE = (8, 8)
TRAINING_CURVES_FIGSIZE = (10, 4)
PER_CLASS_METRICS_FIGSIZE = (10, 5)
PER_CLASS_METRIC_NAMES = [PER_CLASS_KEY_PRECISION, PER_CLASS_KEY_RECALL, PER_CLASS_KEY_ACCURACY]
PER_CLASS_BAR_WIDTH = 0.25
SCORE_AXIS_MAX = 1.05
TICK_LABEL_ROTATION_DEGREES = 45

# --- Progress bars ---
TQDM_BATCH_UNIT = "batch"
TQDM_EVAL_DESCRIPTION = "evaluating"
