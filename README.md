# kungfu-master-a3c
A3C-based reinforcement learning agent trained to play Kung Fu Master using Gymnasium and PyTorch with parallel environments and CNN-based actor-critic architecture.


# 🥋 Kung Fu Master Agent (A3C)

A Reinforcement Learning project implementing **Asynchronous Advantage Actor-Critic (A3C)** to play Kung Fu Master using Gymnasium.

---

## 🧠 Features

* CNN-based Actor-Critic network
* Parallel environment training
* Frame stacking + preprocessing
* Entropy-regularized policy learning

---

## 🏗️ Architecture

* Input: 4 stacked grayscale frames (42x42)
* 3 Convolution layers
* Fully connected layers
* Outputs:

  * Policy (actions)
  * Value function

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Train

```bash
python src/train.py
```

---

## 🧪 Test

```bash
python src/test.py
```

---

## 📦 Output

* Model saved in `checkpoints/model.pth`
* Gameplay videos in `results/videos`

---

## 🚀 Future Work

* PPO implementation
* Better reward shaping
* GPU parallelization

---

## 📜 License

MIT License
