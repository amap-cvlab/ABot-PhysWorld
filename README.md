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

1. **Industrial-Grade Data Pipeline**  
   Curated ~3M real-world manipulation clips from five datasets (`AgiBot`, `RoboCoin`, `RoboMind`, `Galaxea`, `OXE`) with motion, semantic, and action consistency filtering, plus hierarchical sampling for balanced generalization.

2. **Physics-Aware DPO Training**  
   Introduces a decoupled VLM-based discriminator: Qwen3-VL generates task-specific physics checklists, Gemini 3 Pro scores videos via Chain-of-Thought; combined with LoRA-augmented DPO on a 14B DiT to enforce physical plausibility.

3. **Parallel Context Blocks for Action Control**  
   Enables precise action-conditioned generation by residually injecting spatial action maps into cloned DiT blocks, preserving physical priors while supporting cross-embodiment control.

4. **EZSbench – First True Zero-Shot Benchmark**  
   Fully training-independent evaluation covering unseen robot, scene, and task combinations, with dual-model scoring to eliminate self-evaluation bias.

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



## 🖼️ Case Studies: Qualitative Results

Selected representative zero-shot generation results demonstrating ABot-PhysWorld's strong generalization and physical plausibility.

---

### 🎯 Zero-Shot Capabilities

#### 🔧 Case 1: Deformable Object – Dual-Arm Towel Folding  
<div align="center">
  <table>
    <tr>
      <td><img src="examples/case1/case1-1.gif" width="300"></td>
      <td><img src="examples/case1/case1-2.gif" width="300"></td>
    </tr>
    <tr>
      <td><img src="examples/case1/case1-3.gif" width="300"></td>
      <td><img src="examples/case1/case1-4.gif" width="300"></td>
    </tr>
  </table>
</div>

- **Task**: Fold a towel using dual robotic arms  
- **Challenge**: Complex cloth dynamics and bimanual coordination  
- **Ours**:  
  ✅ Physically realistic deformation  
  ✅ Smooth, collision-free arm motion  
  ✅ Natural folding sequence with consistent contact

---

#### 🥤 Case 2: Fine Manipulation – Diverse Object Handling  
<div align="center">
  <table>
    <tr>
      <td><img src="examples/case2/case2-1.gif" width="300"></td>
      <td><img src="examples/case2/case2-2.gif" width="300"></td>
    </tr>
    <tr>
      <td><img src="examples/case2/case2-3.gif" width="300"></td>
      <td><img src="examples/case2/case2-4.gif" width="300"></td>
    </tr>
    <tr>
      <td><img src="examples/case2/case2-5.gif" width="300"></td>
      <td><img src="examples/case2/case2-6.gif" width="300"></td>
    </tr>
    <tr>
      <td><img src="examples/case2/case2-7.gif" width="300"></td>
      <td><img src="examples/case2/case2-8.gif" width="300"></td>
    </tr>
  </table>
</div>
- **Task**: Stack cups, build blocks, place a knife  
- **Challenge**: Varying shapes, weights, and friction  
- **Ours**:  
  ✅ Accurate grasp pose prediction  
  ✅ Adaptive gripper control  
  ✅ Stable pick-and-place without slippage or penetration

---

#### 🚪 Case 3: Articulated Object – Opening a Cabinet Door  
<div align="center">
  <table>
    <tr>
      <td><img src="examples/case3/case3-1.gif" width="300"></td>
      <td><img src="examples/case3/case3-2.gif" width="300"></td>
    </tr>
  </table>
</div>
- **Task**: Open a hinged cabinet or door  
- **Challenge**: Enforce rotational constraints and correct force direction  
- **Ours**:  
  ✅ Proper handle grasping  
  ✅ Realistic hinge rotation  
  ✅ Motion follows physical pivot axis

---

#### 🫗 Case 4: Fluid Interaction – Pouring Water  
<div align="center">
  <table>
    <tr>
      <td><img src="examples/case4/case4-1.gif" width="300"></td>
      <td><img src="examples/case4/case4-2.gif" width="300"></td>
    </tr>
  </table>
</div>
- **Task**: Pour water from a cup into a bowl using dual arms  
- **Challenge**: Bimanual coordination, tilt control, liquid dynamics  
- **Ours**:  
  ✅ Collision-free trajectory planning  
  ✅ Accurate pour timing and angle  
  ✅ Visual consistency in fluid transfer (simulated proxy)

---

#### 🧽 Case 5: Cleaning Task – Wiping a Stain  
<div align="center">
  <table>
    <tr>
      <td><img src="examples/case5/case5-1.gif" width="300"></td>
      <td><img src="examples/case5/case5-2.gif" width="300"></td>
    </tr>
    <tr>
      <td><img src="examples/case5/case5-3.gif" width="300"></td>
      <td><img src="examples/case5/case5-4.gif" width="300"></td>
    </tr>
  </table>
</div>

- **Task**: Wipe a stain off a table  
- **Challenge**: Maintain contact, uniform pressure, full coverage  
- **Ours**:  
  ✅ Continuous tool-surface contact  
  ✅ Systematic wiping motion  
  ✅ Gradual removal of the stain in video output

---

#### 🍓 Case 6: Multi-Scene Generalization – Fruit Sorting  
<div align="center">
  <table>
    <tr>
      <td><img src="examples/case6/case6-1.gif" width="300"></td>
      <td><img src="examples/case6/case6-2.gif" width="300"></td>
    </tr>
    <tr>
      <td><img src="examples/case6/case6-3.gif" width="300"></td>
      <td><img src="examples/case6/case6-4.gif" width="300"></td>
    </tr>
  </table>
</div>

- **Task**: Place fruits into a plate across diverse scenes  
- **Challenge**: Background, lighting, and fruit variation  
- **Ours**:  
  ✅ Robust object recognition under domain shifts  
  ✅ Consistent performance across unseen environments  
  ✅ Fast and stable manipulation regardless of setup

---

### 🔍 PBench Qualitative Comparison




We conduct systematic qualitative evaluations on the **PAI-Bench** test set. Below are key observations:

| Task | Baselines | **Ours** |
|------|-----------------------------|--------|
| Grasping | Frequent penetration, floatation | ✅ Firm contact, no violation |
| Long-horizon Planning | Inconsistent state transitions | ✅ Coherent multi-step reasoning |
| Rigid-body Dynamics | Unphysical deformations | ✅ Preserved geometry and mass behavior |
| Contact Modeling | Non-contact attraction | ✅ Realistic interaction onset |

> Our model consistently generates physically valid trajectories even in complex, unseen scenarios — proving its utility as a reliable simulator for embodied AI.

---


## 🙏 Acknowledgement
This project builds upon [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VACE](https://github.com/ali-vilab/VACE), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We thank these teams for their open-source contributions.


---


