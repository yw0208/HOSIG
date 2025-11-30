# HOSIG: Full-Body Human-Object-Scene Interaction Generation with Hierarchical Scene Perception
[![report](https://img.shields.io/badge/Project-Page-blue)](https://yw0208.github.io/hosig/)
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2506.01579)

Hello everyone, thanks for your patience. HOSIG has now been accepted by **AAAI 2026**. Related code and data will be released gradually over the next three months. Please stay tuned.

<video src="assets/demo-9.mp4"
       controls
       loop
       autoplay
       muted
       playsinline
       style="max-width: 100%; height: auto;">
</video>

## News

📢 **30/Nov/25** - Released inference code and processed TRUMANS for Scene-Guided Controllable Motion Generation.

## 📝 TODO List  
- [x] Release inference demo for Scene-Guided Controllable Motion Generation
- [ ] Release inference demo for Heuristic Navigation on 2D Obstacle-Aware Map
- [ ] Release inference demo for Scene-Aware Grasp Pose Generation 
- [ ] Release training for motion generator, including codes and data
- [ ] Release training for pose generator, including codes and data
- [ ] Release evaluation pipeline
- [ ] Release codes for processing data

## 📖 Getting Started

### Dependencies
- Python 3.8
- PyTorch 2.0.0+cu118

```
pip install -r requirements.txt
```
I'm not sure if requirements.txt contains all the necessary dependencies. If you're experiencing environment configuration issues, please feel free to contact me.

### Data
To run the demo, follow these steps:

1. Download the [TRUMANS](https://github.com/jnnan/trumans_utils/tree/main) dataset, especially Scene folder.
2. Download other necessary data [Baidu Pan](https://pan.baidu.com/s/1XL582kCx8NI7P02vbGa8qg?pwd=trcf) with passport **trcf**. Unzip the downloaded files and just place them in the `HOSIG` directory as shown in the following figure.

<img src="assets\path.png" width="400">

## 🚀 Quick Start

### Scene-Guided Controllable Motion Generation

#### Generate

To generate controllable motion, run the following command:

```
python demo\infer\generate_multi_samples.py
```
the output will be saved in `demo/data/key_frames`. Considering that generating all motions takes too long, you can go into the `generate_multi_samples.py` to modify scene, obj_name and sample_id.

#### Visualization

We provide visulization code for the generated motions. Run the following command to visualize the generated motions:
```
python demo\visualize\vis_smplx_params_all.py
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{yao2025hosig,
  title={HOSIG: Full-Body Human-Object-Scene Interaction Generation with Hierarchical Scene Perception},
  author={Yao, Wei and Sun, Yunlian and Zhang, Hongwen and Liu, Yebin and Tang, Jinhui},
  journal={arXiv preprint arXiv:2506.01579},
  year={2025}
}
```
