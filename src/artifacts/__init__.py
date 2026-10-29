"""Shared on-disk storage: downloaded data, trained checkpoints, and generated results.

Pure storage - no logic of its own. Each sub-package exposes a single path
constant (DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR) that other packages import
to locate these directories reliably, rather than each package independently
guessing a relative path. The actual contents of these directories (dataset
images, .pt weight files, generated plots/CSVs/JSON) are gitignored and
created at runtime.
"""
