# PlaNet-RL

A PyTorch implementation of a simplified PlaNet (Planning Network) agent trained on the CartPole-v1 environment. This project replicates the core components of the PlaNet architecture including a Recurrent State-Space Model (RSSM), latent-space planning, and variational sequence modeling.

---

## 🧠 Project Highlights

- Implements core ideas from the PlaNet paper in a modular fashion
- Uses a recurrent latent model (RSSM) to learn compact world models
- Performs model-based planning using latent rollouts

---

## 📁 Folder Structure

```
PlaNet-RL/
│
├── model/             # Core neural network components
│   ├── encoder.py
│   ├── rssm.py
│   └── decoder.py
│
├── agent/             # Planning agent logic
│   ├── planner.py
│   └── agent.py
│
├── envs/              # Environment loader
│   └── make_env.py
│
├── train.py           # Training loop
├── test.py            # Evaluation and GIF rendering
├── utils.py           # Replay buffer, logging, utilities
├── config.yaml        # Config file
├── README.md          # Project documentation
├── .gitignore         # Ignore config/logs/pycache
├── training.log       # Mock training logs
└── results/
    ├── rewards_plot.png
    └── sample_episode.gif
```

---

## 🚀 Training

```bash
python train.py
```

Training logs and reward plots will be saved to `training.log` and `results/` respectively.

---

## 📊 Evaluation

```bash
python test.py
```

This will generate a `sample_episode.gif` visualizing an episode played by the trained agent.

---

## 📦 Requirements

- Python 3.8+
- PyTorch >= 1.10
- Gym >= 0.21
- imageio

Install requirements:
```bash
pip install -r requirements.txt
```

---

## ✏️ Author

This project was independently implemented and tested locally for research and educational purposes.
