"""Shared library: model architectures, the training loop, preprocessing, and dataset building.

Used by every other package in this repo (fer_training, fer_server, attacks,
defenses). Deliberately has zero dependency on any other package here, and
zero coupling to fer_training's YAML config system - every public function
and class takes plain arguments (paths, floats, ints, or small
locally-defined settings dataclasses like TrainConfig/PreprocessingConfig),
so it can be reused by consumers that have nothing to do with fer_training's
specific config.yaml schema.
"""
