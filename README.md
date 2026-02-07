# 🏀 BasketBallRL

A reinforcement learning project that trains an AI agent to shoot basketballs into a hoop using **PPO (Proximal Policy Optimization)** from Stable Baselines3.

![Reward Evolution](reward_evolution.png)

## 📋 Project Overview

This project implements a custom basketball shooting environment using **Gymnasium** and **Pygame**, where an AI agent learns to shoot a ball from random positions into a fixed hoop. The agent is trained using the PPO algorithm to optimize the shooting angle and power.

## 🎮 Environment Description

### State Space (Observation)
The agent receives a 2-dimensional observation:
- **Ball X Position**: Current horizontal position of the ball (0-1000)
- **Distance to Hoop**: Euclidean distance from the ball to the hoop center

### Action Space
The agent outputs a 2-dimensional continuous action:
- **Angle** (-1 to 1): Mapped to shooting angle (20° to 80°)
- **Power** (-1 to 1): Mapped to shooting power (10 to 28 units)

### Reward Structure
- ✅ **+100** for scoring (ball passes through the hoop while descending)
- 📏 **-0.1 × min_distance** reward shaping based on how close the ball got to the hoop
- ❌ **-5** penalty for going out of bounds

### Environment Parameters
| Parameter | Value |
|-----------|-------|
| Screen Size | 1000 × 700 pixels |
| Hoop Position | (800, 150) |
| Hoop Radius | 20 pixels |
| Ball Radius | 8 pixels |
| Gravity | 0.3 |
| FPS | 60 |

## 📁 Project Structure

```
BasketBallRL/
├── basketball_env.py    # Custom Gymnasium environment
├── train_best.py        # Training script with PPO
├── test_best.py         # Testing/demo script
├── reward_evolution.png # Training reward graph
├── README.md
├── logs_v2/             # TensorBoard logs
│   ├── monitor.csv
│   ├── PPO_1/
│   ├── PPO_2/
│   └── PPO_3/
└── models_v2/           # Saved models
    └── best_shooter_v2.zip
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install gymnasium stable-baselines3 pygame numpy tensorboard
```

### Training

Run the training script to train a new model:

```bash
python train_best.py
```

**Training Configuration:**
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MlpPolicy (Multi-layer Perceptron)
- **Learning Rate**: 0.0001
- **Total Timesteps**: 150,000
- **Batch Size**: 64
- **n_steps**: 2048
- **Gamma**: 0.99
- **GAE Lambda**: 0.95

### Testing

Run the demo to see the trained agent in action:

```bash
python test_best.py
```

This will launch a visual demonstration with 50 consecutive shots, displaying:
- Starting X position
- Action taken (angle, power)
- Observation state
- Reward obtained

### Monitoring with TensorBoard

View training progress with TensorBoard:

```bash
tensorboard --logdir=logs_v2
```

## 📊 Training Results

The training logs show the evolution of the mean episode reward across multiple training runs. The agent progressively learns to:
1. Aim towards the hoop direction
2. Calibrate the shooting power
3. Score consistently from various starting positions

## 🛠️ Technical Details

### Physics Simulation
- Projectile motion with constant gravity
- Velocity components calculated from angle and power
- Collision detection with hoop using distance threshold

### Scoring Logic
The ball is considered scored when:
1. It passes within the hoop radius + ball radius
2. The ball is descending (vy > 0)

## 📚 Dependencies

- `gymnasium` - RL environment framework
- `stable-baselines3` - PPO implementation
- `pygame` - Visual rendering
- `numpy` - Numerical computations
- `tensorboard` - Training visualization

## 👨‍💻 Author

M1 AI - Reinforcement Learning Project

## 📄 License

This project is for educational purposes.