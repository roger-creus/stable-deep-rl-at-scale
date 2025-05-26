
# 🧠 Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning

This repository contains the code for the paper:  
**"Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning"**  
by *Roger Creus Castanyer, Johan Obando-Ceron, Lu Li, Pierre-Luc Bacon, Glen Berseth, Aaron Courville, and Pablo Samuel Castro*  
*Work done at Mila Quebec AI Institute and University of Montreal.*

---

## 📦 Installation

To set up the environment for running the main scripts:

```bash
conda create -n deeprl python=3.10 -y
conda activate deeprl
pip install -r requirements.txt
```

To run Atari experiments (`dqn.py`, `rainbow.py`), install additional dependencies:

```bash
pip install -r requirements_atari.txt
pip install -U typing_extensions
```

---

## 📁 Project Structure

```
src/
├── pqn.py          # PQN algorithm
├── ppo.py          # PPO algorithm
├── dqn.py          # DQN algorithm
├── rainbow.py      # Rainbow algorithm
├── cifar.py        # CIFAR-10 experiments (supervised learning & non-stationary SL)
├── models/
│   ├── agent.py    # Base classes for PQN and PPO agents
│   ├── encoder.py  # CNN encoders (AtariCNN, ImpalaCNN)
│   └── mlp.py      # MLP variants (FC, Residual, Multi-skip, DenseNet)
└── utils/
    ├── compute_hns.py          # Compute Human Normalized Scores
    ├── args.py                 # CLI arguments for PPO and PQN
    ├── representation_dynamics.py # Metrics and logging for representation dynamics
    ├── utils.py                # General utilities (optimizers, scaling, etc.)
    └── wrappers.py            # Atari environment wrappers
```

---

## 🚀 Example Usage

> All CLI arguments for PPO and PQN are found in `src/utils/args.py`  
> DQN, Rainbow, and CIFAR-10 scripts also expose similar CLI argument interfaces.
> Running any script without command line arguments will use the default baseline configuration settings.

### ▶️ Run PQN (baseline)
```bash
python src/pqn.py
```

### 🔬 Run PQN with Gradient Interventions
```bash
python src/pqn.py --mlp_type=multiskip_residual --use_ln --optimizer=kron
```

### 🧗 Run PPO with Gradient Interventions
```bash
python src/ppo.py --mlp_type=multiskip_residual --use_ln --optimizer=kron
```

### 🎮 Run DQN with Gradient Interventions
```bash
python src/dqn.py --mlp_type=multiskip_residual --use_ln --optimizer=kron
```

### 🌈 Run Rainbow with Gradient Interventions
```bash
python src/rainbow.py --mlp_type=multiskip_residual --use_ln --optimizer=kron
```

### 🖼️ Run CIFAR-10 Baseline (SL & Non-Stationary SL)
```bash
python src/cifar.py --mlp_type=multiskip_residual --use_ln --optimizer=kron
```

---

## 📄 Citation

If you use this codebase in your research, please cite the original paper.

