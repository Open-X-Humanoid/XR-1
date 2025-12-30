<div align="center">

# XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations 

A versatile and scalable vision-language-action framework: XR-1 supports robust multi-task learning across diverse robot embodiments and environments.

> Shichao Fan<sup>1,\*</sup>, Kun Wu<sup>1,\*</sup>, Zhengping Che<sup>1,\*,&dagger;</sup>, Xinhua Wang<sup>1</sup>, Di Wu<sup>1,4</sup>, Fei Liao<sup>1</sup>, Ning Liu<sup>1</sup>, Yixue Zhang<sup>1</sup>, Zhen Zhao<sup>1</sup>, Zhiyuan Xu<sup>1</sup>, Meng Li<sup>1</sup>, Qingjie Liu<sup>3</sup>, Shanghang Zhang<sup>4</sup>, Min Wan<sup>2</sup>, Jian Tang<sup>1,&#9993;</sup>

<sup>1</sup>Beijing Innovation Center of Humanoid Robotics,
<sup>2</sup>School of Mechanical Engineering and Automation, Beihang University,
<sup>3</sup>State Key Laboratory of Virtual Reality Technology and Systems, SCSE, Beihang University,
<sup>4</sup>State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University

<sup>*</sup>Co-first authors,
<sup>&dagger;</sup>Project leader,
<sup>&#9993;</sup>Corresponding author,

[![arXiv](https://img.shields.io/badge/arXiv-2511.02776-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.02776) [![Project Page](https://img.shields.io/badge/Project-Page-green.svg?logo=github&logoColor=white)](https://xr-1-vla.github.io/)

[\[📖 Document\]](#documents) [\[🚀 Quick Start\]](#-quick-start) [\[🤗 Models\]](#-models) [\[🤖 Deployment\]](#-XR-1-Deployment) [\[✅ Performance\]](#-performance-in-real-world) [\[🙋 FAQs\]](#-faqs)


![perform](assets/images/xr1_teaser.png)

</div>

## TODO List

- [x] Release pre-training / fine-tuning code for XR-1 series.
- [x] Release pre-trained model, and heterogeneous dataset sample of XR-1 on both HuggingFace and ModelScope.
- [ ] Release real world deloyment sample of XR-1.


## Documents
This repository is built upon a fork of [Lerobot](https://github.com/huggingface/lerobot). Please note that due to the rapid updates of Lerobot, our implementation is specifically aligned with **Lerobot dataset v2.1**. We have preserved the original directory structure to facilitate further development and integration for the community.


### 🚀 Quick Start
### 📑 Installation


Download our source code:
```bash
git clone https://github.com/Open-X-Humanoid/XR-1.git
cd XR-1
```
Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html), then install the dependencies:
```bash
conda create -y -n xr1 python=3.10
conda activate xr1
pip install -e ".[xr1]"
```

### 📑 Dataset Preparation 

1.  **Format Compatibility**: Since our environment relies on **LeRobot Dataset v2.1**, we recommend using [any4lerobot](https://github.com/Tavish9/any4lerobot/) to convert your data to this standard.
2.  **Sample Data**: We provide a heterogeneous dataset sample (including EGO4D and Robot data like TienKung2/UR/Franka) available at [X-Humanoid/XR-1-Dataset-Sample](https://huggingface.co/datasets/X-Humanoid/XR-1-Dataset-Sample). You can download it using the provided script: `scripts/hf_xr1_dataset_sample_download.sh` or `scripts/modelscope_xr1_dataset_sample_download.sh`.
3.  **Unified Dataloader**: We have designed a powerful dataloader that unifies heterogeneous data sources and embodiments, making **pre-training** extremely simple. You can find the implementation in `examples/xr1_cross_dataset_and_embodiment_dataloader.py`.

    **Key enhancements over the original LeRobot dataloader:**
    - **Unified Data Loading**: Seamlessly reads data from diverse sources and embodiments.
    - **Multi-Task Support**: Compatible with heterogeneous multi-task learning.
    - **Few-Shot Capabilities**: Supports training with small sample sizes.
    - **Extensibility**: Easily adaptable to new formats (e.g., non-LeRobot formats like Ego4D) with minimal development.

## 🤗 Models

To set up the model environment, first download the foundation models (e.g., SigLIP, PaliGemma) by running:
```bash
# Huggingface
bash scripts/hf_download.sh
```
Then, to obtain the **XR-1-Stage1-UVMC** and **XR-1-Stage2-Pretrain** models for fine-tuning, run:
```bash
# Huggingface
bash scripts/hf_xr1_pretrain_model_download.sh
# Or ModelScope
bash modelscope_xr1_pretrain_model_download.sh
```

## 📖 Training Recipe

We provide three training paths depending on your data and performance requirements:


### 📑 Fast Fine-tuning (For Quick Deployment)
If you need to quickly adapt the model to a new task or robot, you can fine-tune only the **Stage 3**. 
This is the fastest way to get a deployable model:

```bash
# Debug Mode (For testing configurations):
bash scripts/xr1_stage3_finetune.sh --debug
# Standard Training (Default):
bash scripts/xr1_stage3_finetune.sh

```


### 📑 Full Fine-tuning (Recommended for Best Performance)
For custom datasets where you aim for optimal performance, we strongly recommend fine-tuning **all three stages** (Stage 1, 2, and 3) sequentially to better align the representations with your specific data:
```bash
# Full Fine-tuning Stage1 & Stage2 & stage3
bash scripts/xr1_stage1_finetune.sh
bash scripts/xr1_stage2_finetune.sh
bash scripts/xr1_stage3_finetune.sh (optional)
```

### 📑 Pre-training from Scratch
Our framework fully supports pre-training if you have access to large-scale, heterogeneous datasets across diverse embodiments and environments:
```bash
# Pre-training Stage1 & Stage2
bash scripts/xr1_stage1_pretrain.sh
bash scripts/xr1_stage2_pretrain.sh
```

## 🤖 XR-1 Deployment

We provide a streamlined workflow to deploy and verify XR-1 on various robotic platforms, including Franka, UR, and Agilex. The following example demonstrates the process using a Franka dual-arm robot:
``` bash
# 1. Perform Fast Fine-tuning to train a specific Stage 3 model
bash scripts/xr1_stage3_finetune.sh --debug --dataset XR_1_DATASET_DUAL_ARM_FRANKA
# 2. Execute the deployment script
python deploy/real_robot/xr1_deploy.py
```

For deployment on **TienKung 2.0**, we recommend referring to the [x-humanoid-training-toolchain](https://github.com/Open-X-Humanoid/x-humanoid-training-toolchain/tree/main/deployment) for specialized instructions.


## ✅ Performance in Real-world
<table width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="assets/images/Dual-Arm%20UR-5e_results.png" alt="Dual-Arm UR-5e">
      <br><b>Dual-Arm UR-5e</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/images/Tien%20Kung%202.0_results.png" alt="Tien Kung 2.0">
      <br><b>Tien Kung 2.0</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/images/tienkung1_results.png" alt="Tien Kung 1.0">
      <br><b>Tien Kung 1.0</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/images/Dual-Arm%20Franka_results.png" alt="Dual-Arm Franka">
      <br><b>Dual-Arm Franka</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/images/AgileX%20Cobot%20Magic%20V2.0_results.png" alt="AgileX Cobot Magic V2.0">
      <br><b>AgileX Cobot Magic V2.0</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/images/Single-Arm%20UR-5e_results.png" alt="Single-Arm UR-5e">
      <br><b>Single-Arm UR-5e</b>
    </td>
  </tr>
</table>

## 🤗 FAQs
If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. We appreciate your feedback and contributions! 🚀

## License

This project is released under the [Apache License](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{fan2025xr,
  title={XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations},
  author={Fan, Shichao and Wu, Kun and Che, Zhengping and Wang, Xinhua and Wu, Di and Liao, Fei and Liu, Ning and Zhang, Yixue and Zhao, Zhen and Xu, Zhiyuan and others},
  journal={arXiv preprint arXiv:2511.02776},
  year={2025}
}
```

## Acknowledgement
XR-1 is built with reference to the code of the following projects: [Lerobot](https://github.com/huggingface/lerobot), [Moto](https://github.com/TencentARC/Moto), [QueST](https://github.com/pairlab/QueST) and [Pi0](https://github.com/Physical-Intelligence/openpi). Thanks for their awesome work!