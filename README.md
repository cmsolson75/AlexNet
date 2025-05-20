
# AlexNet Legacy Implementation

This repository contains a Hydra-configured, PyTorch Lightning-based implementation of the AlexNet architecture for CIFAR-10 and CIFAR-100 datasets. The project is designed for experimental and diagnostic purposes, not production use or performance benchmarking.

---

## Project Structure
````
alexnet/
├── configs/             # Hydra configuration files
│   ├── checkpoint/
│   ├── dataset/
│   ├── model/
│   ├── optimizer/
│   └── trainer/
├── data/                # Data loading logic
├── models/              # AlexNet architecture
├── training/            # Lightning training wrapper

````


## Environment Setup

This project uses **Python 3.10**. Development was done using **conda** for environment management and **pip** for dependency resolution.

### 1. Clone the Repository

```bash
git clone https://github.com/https://github.com/cmsolson75/AlexNet.git
cd AlexNet
```

### 2. Create and Activate Environment

```bash
conda create -n alexnet python=3.10
conda activate alexnet
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---


## Dependencies

All runtime dependencies are specified in `requirements.txt`. Key libraries:

* `torch`
* `torchvision`
* `lightning`
* `wandb`
* `omegaconf`
* `hydra-core`
* `torchmetrics`
* `tqdm`

---

## Configuration System

Hydra is used to compose configuration files located under `alexnet/configs/`.

Primary config:
`alexnet/configs/config.yaml`

Modular components:

* `dataset/` — Dataset-specific parameters (e.g. batch size, location)
* `model/` — AlexNet architecture variants
* `optimizer/` — SGD and LR scheduler
* `trainer/` — Lightning trainer settings
* `checkpoint/` — Checkpointing frequency and naming

Override configs at runtime as needed:

```bash
python train.py model.pool.type=average optimizer.lr=0.01
```


---

## Training
Before starting, log into Weights & Biases (only required once per machine):

```
wandb login
```
Then launch training
```bash
python train.py
```

This will:

* Load config from `alexnet/configs/config.yaml`
* Create experiment-specific output directory:
  `outputs/<project>/<run_name>/`
* Log training to Weights & Biases
* Save checkpoints via Lightning

---

## Evaluation

```bash
python evaluate.py \
    --model-path path/to/model.pth
```

This uses the test loader for the dataset defined in the active config.

---

## Unwrap Trained Model

Use this to strip the Lightning wrapper and extract the raw AlexNet model:

```bash
python unwrap_model.py \
    --ckpt-path path/to/lightning.ckpt \
    --output-path path/to/alexnet.pth
```

---

## Notes

* Run names include a deterministic hash of the config and a timestamp.
* Checkpoints and logs are stored in machine-independent paths.
* All configurations are reproducible with fixed seeds.

---

