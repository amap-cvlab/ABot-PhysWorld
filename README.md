<div align="center">
<!-- <img src="assets/logo.png" alt="Logo" width="200"/> -->

<h1>🤖 ABot-PhysWorld: Interactive World Foundation Model for
Robotic Manipulation with Physics Alignment</h1>


<p align="center">
  <b>AMAP CV Lab</b>
</p>


<p align="center">
  <a href="https://github.com/amap-cvlab/ABot-PhysWorld"><img src="https://img.shields.io/static/v1?label=Paper&message=Technical_Report&color=red&logo=arxiv"></a>
  <a href="https://github.com/amap-cvlab/ABot-PhysWorld/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
    <a href="https://huggingface.co/acvlab"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange"></a>
</p>

</div>


> **ABot-PhysWorld** is a physically consistent, action-controllable video world model for robotic manipulation, built on a 14-billion-parameter Diffusion Transformer. It integrates physics-aware training, memory-efficient preference optimization, and precise spatial action injection to generate realistic and physically plausible robot-object interactions — even in zero-shot settings.


## Table of Contents
- [📚 Key Contributions](#-key-contributions)
- [📊 Evaluation](#-Evaluation)
- [📜 Citing](#-Citing)
- [🙏 Acknowledgement](#-acknowledgement)


## 📚 Key Contributions

### 1. **Industrial-Grade Embodied Data Pipeline**
- Aggregates and refines **~3M real-world manipulation videos** from five public datasets: `AgiBot`, `RoboCoin`, `RoboMind`, `Galaxea`, and `OXE`.
- Implements multi-stage filtering:
  - Motion consistency (optical flow)
  - Semantic coherence (CLIP embeddings)
  - Action annotation alignment
- Introduces **hierarchical dynamic sampling** to balance rare robot morphologies and long-tail tasks.

### 2. **Physics-Aware Post-Training via Decoupled DPO**
- Proposes a **decoupled VLM discriminator**: 
  - A *proposer* (Qwen3-VL-Thinking) generates task-specific physics checklists.
  - A *scorer* (Gemini 3 Pro) evaluates candidates using Chain-of-Thought reasoning.
- Uses **tournament-based sampling** to create high-margin preference triplets.
- Applies **LoRA-augmented Direct Preference Optimization (DPO)** on a 14B DiT model to preserve pre-trained physical priors.

### 3. **Parallel Context Block for Action Injection**
- Avoids catastrophic forgetting by **residually injecting spatial action maps** into cloned DiT blocks.
- Encodes end-effector poses, gripper states, and joint trajectories as image-like feature maps.
- Supports **cross-embodiment control** without modifying core physical knowledge.

### 4. **EZSbench: The First Zero-Shot Benchmark for Embodied Video Generation**
- Fully independent of training data.
- Covers unseen combinations of:
  - Robot arms
  - Scenes & materials
  - Tasks (from pick-and-place to complex assembly)
- Combines synthetic and real-world initial observations.
- Uses a **dual-model evaluation paradigm** to prevent self-evaluation bias.

---

## 📊 Evaluation

We evaluate ABot-PhysWorld on three key aspects:  
1. **Physical Consistency** (via **PBench** and **EZSbench**)  
2. **Zero-Shot Generalization** (via **EZSbench**)  
3. **Action-Conditioned Controllability** (via custom A2V benchmark)

### 📈 Summary of Advancements 🎉🎉

| Capability | Benchmark | Ours | Best Baseline | Gain |
|----------|-----------|------|---------------|------|
| Physical Fidelity | PBench (Domain Score) | **0.9306** | 0.8391 (Wan2.1) | +10.8% |
| Zero-Shot Generalization | EZSbench (Domain Score) | **0.8366** | 0.7951 (WoW) | +5.2% |
| Action Control | Trajectory Consistency | **0.8522** | 0.8157 (Enerverse) | +4.5% |

✅ ABot-PhysWorld establishes a new standard for **physically grounded**, **controllable**, and **generalizable** world models in robotic manipulation.



## 🛠️ Usage

> _Coming soon: Public release of model weights, inference code, and EZSbench toolkit._





## 📜 Citing

If you find **ABot-PhysWorld** is useful in your research or applications, please consider giving us a **star** 🌟 and **citing** it by the following BibTeX entry:

```
@article{
  title={ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment},
  year={2026}
}
```

---


## 🙏 Acknowledgement
This project builds upon [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VACE](https://github.com/ali-vilab/VACE), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We thank these teams for their open-source contributions.


---


