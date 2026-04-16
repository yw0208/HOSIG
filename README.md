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

📢 **2/Mar/26** - Released evaluation code.

📢 **1/Mar/26** - Released training code for Scene-Guided Controllable Motion Generation.

📢 **27/Feb/26** - Released inference and training code and processed TRUMANS for Scene-Aware Grasp Pose Generation.

📢 **6/Jan/26** - The preprocessed trumans data has been updated. If you encounter Size Mismatch issues during runtime, please refer to this [issue](https://github.com/yw0208/HOSIG/issues/5) and redownload the data again.

📢 **30/Nov/25** - Released inference code and processed TRUMANS for Scene-Guided Controllable Motion Generation.

## 📝 TODO List  
- [x] Release inference demo for Scene-Guided Controllable Motion Generation
- [x] Release inference demo for Heuristic Navigation on 2D Obstacle-Aware Map
- [x] Release inference demo for Scene-Aware Grasp Pose Generation 
- [x] Release training for motion generator, including codes and data
- [x] Release training for pose generator, including codes and data
- [x] Release evaluation pipeline
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
Visualization codes are based on [aitviewer](https://github.com/eth-ait/aitviewer), recommended to install on Windows. Some people may encounter strange bugs due to version updates when running my code. I have an older version of `smpl.py` in `assets`. Please replace `aitviewer\renderables\smpl.py` with the older version; this may solve the problem.

#### Train

First, you need to train the main branch of SCoMoGen. Please refer to [MDM_359](https://github.com/yw0208/MDM_359) to train the model, or directly use the latest updated `save/model000475000.pt` from [Baidu Pan](https://pan.baidu.com/s/1XL582kCx8NI7P02vbGa8qg?pwd=trcf).

Then, you can train the control branch of SCoMoGen for motion generation. Run the following command:

```
python -m train.train_mdm --save_dir save/scomogen --dataset trumans --num_steps 400000 --batch_size 64 --resume_checkpoint ./save/model000475000.pt --lr 1e-5
```

### Heuristic Navigation on 2D Obstacle-Aware Map

To run the demo, run the following command:
```
python demo\infer\object_A_star_3point.py
```

### Scene-Aware Grasp Pose Generation

First, go to [Baidu Pan](https://pan.baidu.com/s/1XL582kCx8NI7P02vbGa8qg?pwd=trcf), find the `TRUMAN/process4Gnet_3obj` folder and download it. Place it anywhere you like (note that you need to modify the file path in the code, Ctrl+Shift+F `process4Gnet_3obj`). 

To run the demo, run the following command:

```
cd sgap

python sgap\generate\multi_samples\get_batch.py

python sgap\generate\multi_samples\generate_one_sample.py
```

Note that SGAP cannot generate perfect results every time, so we choose to generate 10 times for each sample and then manually select the best result as the guide for SCoMoGen.

To train the model, run the following command:

```
cd sgap

python train\train.py
```

## 🛠️ Evaluation
To evaluate the performance of SCoMoGen, run the following command:
```
python evaluation\eval_interaction.py
```
Note that you should prepare evaluation meshes following `demo\visualize\vis_smplx_params_all.py`.

To evaluate the performance of SGAP, run the following command:
```
python evaluation\eval_grasp.py
```

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{yao2026hosig,
  title={Hosig: Full-body human-object-scene interaction generation with hierarchical scene perception},
  author={Yao, Wei and Sun, Yunlian and Zhang, Hongwen and Liu, Yebin and Tang, Jinhui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={14},
  pages={11901--11909},
  year={2026}
}
```
