![](https://xr-1-vla.github.io/static/images/xr1_teaser.png)

## We introduce XR-1, a versatile and scalable vision-language-action framework. XR-1 supports robust multi-task learning across diverse robot embodiments and environments

## Abstract

Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models. However, existing VLA models still face two fundamental challenges: (i) producing precise low-level actions from high-dimensional observations, (ii) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments. At its core, XR-1 introduces the Unified Vision-Motion Codes (UVMC), a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion. UVMC addresses these challenges by (i) serving as an intermediate representation between the observations and actions, and (ii) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a three-stage training paradigm: (i) self-supervised UVMC learning, (ii) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (iii) task-specific post-training. We validate XR-1 through extensive real-world experiments with more than 14,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\\pi\_{0.5}$, $\\pi\_0$, RDT, UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes.

## Overview

![](https://xr-1-vla.github.io/static/images/xr1_overview.png)

Overview of XR-1. In XR-1, we introduce the *Unified Vision-Motion Codes (UVMC)*, a discrete latent representation that jointly encodes visual dynamics and robotic motion. XR-1 adopts a three-stage training paradigm to enable precise low-level control across diverse robots and tasks.

## Experiment Setup

![](https://xr-1-vla.github.io/static/images/xr1_exp_setup.png)

Experimental Setup. We evaluate XR-1 across six robot embodiments(Tien Kung 1.0/2.0, Single-/Dual-Arm UR-5e, Dual-Arm Franka, and AgileX Cobot Magic 2.0), covering more than 120 manipulation tasks with over 14,000 rollouts.

## Comparison with Baselines

### Representative Tasks Comparison

We conducted evaluations on bimanual collaboration, dexterous manipulation, fluid/deformable object handling, contact-rich interactions, and dynamic environments. Our XR1 model was compared against baseline methods including RDT, π0.5, π0, GR00T-N1.5, and UniVLA. 

## Experiment Result

Main experiment: 6 different embodiments, with 20 tasks per embodiment, comparing results against RDT, π0.5, π0, GR00T-N1.5, and UniVLA baselines.

**Dual-Arm UR-5e\_results**

![](https://xr-1-vla.github.io/static/images/Dual-Arm%20UR-5e_results.png)

**Tien Kung 2.0\_results**

![](https://xr-1-vla.github.io/static/images/Tien%20Kung%202.0_results.png)

**Tienkung1\_results**

![](https://xr-1-vla.github.io/static/images/tienkung1_results.png)

**Dual-Arm Franka\_results**

![](https://xr-1-vla.github.io/static/images/Dual-Arm%20Franka_results.png)

**AgileX Cobot Magic V2.0\_results**

![](https://xr-1-vla.github.io/static/images/AgileX%20Cobot%20Magic%20V2.0_results.png)

**Single-Arm UR-5e\_results**

![](https://xr-1-vla.github.io/static/images/Single-Arm%20UR-5e_results.png)
