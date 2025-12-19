<div align="center">

# XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations (Under Review)

a versatile and scalable vision-language-action framework. XR-1 supports robust multi-task learning across diverse robot embodiments and environments.

> Shichao Fan<sup>1,\*</sup>, Kun Wu<sup>1,\*</sup>, Zhengping Che<sup>1,\*,&dagger;</sup>, Xinhua Wang<sup>1</sup>, Di Wu<sup>1,4</sup>, Fei Liao<sup>1</sup>, Ning Liu<sup>1</sup>, Yixue Zhang<sup>1</sup>, Zhen Zhao<sup>1</sup>, Zhiyuan Xu<sup>1</sup>, Meng Li<sup>1</sup>, Qingjie Liu<sup>3</sup>, Shanghang Zhang<sup>4</sup>, Min Wan<sup>2</sup>, Jian Tang<sup>1,&#9993;</sup>

<sup>1</sup>Beijing Innovation Center of Humanoid Robotics,
<sup>2</sup>School of Mechanical Engineering and Automation, Beihang University,
<sup>3</sup>State Key Laboratory of Virtual Reality Technology and Systems, SCSE, Beihang University,
<sup>4</sup>State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University

<sup>*</sup>Co-first authors,
<sup>&dagger;</sup>Project leader,
<sup>&#9993;</sup>Corresponding author,

[\[📄Paper\]](https://arxiv.org/abs/2511.02776)  [\[🔥Project Page\]](https://xr-1-vla.github.io/) [\[📖 Document\]](#documents) [\[🚀 Quick Start\]](#-quick-start) [\[🤗 Model Zoo\]](#-model-zoo) [\[✅ Performance\]](#-performance-in-real-world) [\[🙋 FAQs\]](#-faqs)


[\[🔥Pre-train\]](#-pre-train-from-scratch) [\[🚀 Fine-tune\]](#-fine-tune-from-spatialvla) [\[🎄Custom Dataset\]](#-use-custom-datasets)

![perform](.assets/images/xr1_teaser.png)

</div>

## TODO List

- [ ] Release pre-training / fine-tuning code for XR-1 series.
- [ ] Release the code, model, and custom data of XR-1.
- [ ] Release real world deloyment sample of XR-1


## Documents
### 🚀 Quick Start
> Coming Soon...
### 🎄 Use Custom Datasets
> Coming Soon...

## 🤗 Model Zoo
> Coming Soon...


## ✅ Performance in Real-world
<table width="100%">
  <tr>
    <td align="center" width="50%">
      <img src=".assets/images/Dual-Arm%20UR-5e_results.png" alt="Dual-Arm UR-5e">
      <br><b>Dual-Arm UR-5e</b>
    </td>
    <td align="center" width="50%">
      <img src=".assets/images/Tien%20Kung%202.0_results.png" alt="Tien Kung 2.0">
      <br><b>Tien Kung 2.0</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src=".assets/images/tienkung1_results.png" alt="Tien Kung 1.0">
      <br><b>Tien Kung 1.0</b>
    </td>
    <td align="center" width="50%">
      <img src=".assets/images/Dual-Arm%20Franka_results.png" alt="Dual-Arm Franka">
      <br><b>Dual-Arm Franka</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src=".assets/images/AgileX%20Cobot%20Magic%20V2.0_results.png" alt="AgileX Cobot Magic V2.0">
      <br><b>AgileX Cobot Magic V2.0</b>
    </td>
    <td align="center" width="50%">
      <img src=".assets/images/Single-Arm%20UR-5e_results.png" alt="Single-Arm UR-5e">
      <br><b>Single-Arm UR-5e</b>
    </td>
  </tr>
</table>

## 🤗 FAQs
If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions. We appreciate your feedback and contributions! 🚀

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

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