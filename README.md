# 🤖 DQN Agent in a Custom Godot Platformer

This project features a **Deep Q-Network (DQN)** agent integrated into a custom 2D platformer built with **Godot Engine (v4.4.1)**. The RL logic is written in **GDScript**.

## 🧠 DQN Summary

DQN is a reinforcement learning algorithm that uses a neural network to estimate **Q-values** (expected rewards for actions). It enables agents to learn optimal behavior through interaction and trial-and-error.

## 🔍 Agent Input

- 8 raycasts (obstacle sensing)
- Player's `(x, y)` position
- Nearest coin's `(x, y)` position
- Is player on ground (boolean)

**Total Inputs: 13**

## 🎮 Action Space

- `0` → Move Left  
- `1` → Move Right  
- `2` → Jump

## 🧠 Network Architecture

- Input: 13 nodes  
- Hidden Layers: 2 × 64 neurons  
- Output: 3 actions (Q-values)

[Screenshot 2025-06-09 155738](https://github.com/user-attachments/assets/51bf3004-4f91-4a61-a116-08bd31de2c6a)

## 🧪 Environment Preview

![Screenshot 2025-06-09 121145](https://github.com/user-attachments/assets/e857e0be-f412-416f-aa12-856f16ce7436)


## ⚙️ Sample DQN Params

```gdscript
epsilon := 0.8
gamma := 0.99
batch_size := 64
target_update_frequency := 1000
