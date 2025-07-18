# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Commands

### Training
- **Run RL training**: `bash examples/run_openvla_oft_rl.sh`
- **Run LoRA-enabled training**: `bash examples/run_openvla_oft_rl_lora.sh`
- **Run evaluation only**: Set `trainer.val_only=True` in the script, then run the same command
- **Main training entry point**: `python -m verl.trainer.main_ppo` with Hydra configuration

### Configuration
- **Main config file**: `examples/run_openvla_oft_rl.sh` - contains all key parameters
- **Runtime environment**: `align.json` - sets environment variables including WANDB_API_KEY
- **Trainer configs**: `verl/trainer/config/ppo_trainer.yaml` - default PPO configuration

## Architecture Overview

### Core Framework
This is a **Vision-Language-Action (VLA) reinforcement learning framework** built on top of veRL and OpenVLA-OFT. It implements online RL training for robotic manipulation tasks using simple 0/1 reward signals.

### Key Components

**Training Pipeline**:
- `verl/trainer/main_ppo.py` - Main PPO training orchestrator
- `verl/trainer/ppo/ray_trainer.py` - Ray-based distributed training
- `verl/workers/rollout/rob_rollout.py` - Robot rollout worker for environment interaction

**Model Integration**:
- `verl/utils/vla_utils/openvla_oft/` - OpenVLA-OFT model implementations
- `verl/utils/libero_utils.py` - LIBERO simulation environment utilities
- `verl/workers/actor/dp_rob.py` - Actor worker for robot tasks

**Configuration System**:
- Uses Hydra for configuration management
- Default configs in `verl/trainer/config/`
- Runtime overrides via command-line arguments in shell script

### Data Flow
1. **SFT Model** → loaded as initial policy
2. **Environment Interaction** → LIBERO simulation generates trajectories
3. **Reward Calculation** → Simple 0/1 rule-based rewards from task completion
4. **PPO Training** → Updates policy using collected experiences
5. **Evaluation** → Tests on LIBERO benchmark tasks

### Key Features
- **Hybrid Engine**: Combines FSDP for training and vLLM for inference
- **Multi-node Support**: Configurable via `NUM_NODES` and `NUM_GPUS`
- **Action Chunking**: Processes sequences of 8 action tokens (7 dimensions each)
- **Outcome-level Rewards**: Uses task completion (0/1) rather than dense rewards

## Prerequisites
- veRL framework installed
- OpenVLA-OFT environment setup
- LIBERO simulation environment
- SFT model checkpoint (from HuggingFace collection or self-trained)
- WANDB API key for experiment tracking

## Configuration Notes
- **Memory Requirements**: Tested on A800 GPUs with 80GB memory
- **Batch Sizes**: Configured for multi-GPU setups (ppo_micro_batch_size based on NUM_GPUS)
- **Temperature**: Set to 1.6 for exploration during rollouts
- **Reward Coefficients**: Simple binary rewards scaled by verifier_reward_coef=5

## LoRA Training Support
- **LoRA Configuration**: Set `lora_rank > 0` to enable LoRA training
- **Target Modules**: Use `target_modules=all-linear` for automatic detection or specify manually
- **Memory Efficiency**: LoRA reduces memory usage by ~50% with minimal performance loss
- **Recommended Settings**: `lora_rank=32`, `lora_alpha=64`, learning rate 1e-5 to 5e-6