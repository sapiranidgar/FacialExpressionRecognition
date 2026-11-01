"""Constants shared by every consumer of the common library.

fer_training-only constants (Kaggle settings, result-file naming,
tuning-specific constants) live in fer_training.constants instead - this
module holds only what's genuinely needed by more than one package.
"""

from __future__ import annotations

# --- Devices ---
DEVICE_CPU = "cpu"

# --- Model type identifiers (used to select transforms/dataloaders/checkpoints) ---
MODEL_TYPE_SIMPLE_CNN = "simple_cnn"
MODEL_TYPE_RESNET = "resnet"

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
# common.training.metrics and every caller that reads its output) ---
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
