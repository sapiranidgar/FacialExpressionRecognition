# FacialExpressionRecognition

This repository represents implementation of a facial recognition model and its attack attempts.

## The Task
Build a CNN (or fine-tune a pretrained model) to classify emotions from human face images.
In the next step, implement an attack on this model.

See [PLAN.md](PLAN.md) for the full design and project structure.

## Setup

```
pip install -r requirements.txt
pip install -e .
```

To download the dataset (FER2013 via Kaggle), you need a Kaggle account and
an API token at `~/.kaggle/kaggle.json` (create one at
https://www.kaggle.com/settings -> "Create New Token"). Then:

```
python scripts/download_dataset.py
```

## Usage

```
python scripts/train_cnn.py       # trains the from-scratch simple CNN
python scripts/train_resnet.py    # fine-tunes ResNet18
python scripts/evaluate_clean.py  # compares both, picks the attack victim
pytest tests/
```

### The Model

- Two models are trained and compared on clean accuracy: a simple CNN
  trained from scratch, and a ResNet18 fine-tuned from ImageNet weights.
  The winner becomes the target for phase 2's attacks; the other serves as
  the substitute model for the transfer-based black-box attack.

### The Attacks

- Not implemented yet. Planned: white-box evasion (FGSM, PGD), black-box
  evasion (transfer-based and query-based via Square Attack), and a
  BadNets-style backdoor poisoning attack. See [PLAN.md](PLAN.md).
