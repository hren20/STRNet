# STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation

> Accepted at **CVPR 2026**

[Hao Ren](https://hren20.github.io)<sup>1,2</sup>,
[Zetong Bi](https://scholar.google.com/citations?hl=zh-CN&user=JJutrU4AAAAJ)<sup>1</sup>,
[Yiming Zeng](https://jzengym.github.io/JZENGYM/)<sup>1</sup>,
[Zhaoliang Wan](https://wan-zhaoliang.vercel.app/)<sup>2</sup>,
[Lu Qi](http://luqi.info/)<sup>2,3</sup>,
[Hui Cheng](https://cse.sysu.edu.cn/teacher/ChengHui)<sup>1</sup>

<sup>1</sup>Sun Yat-sen University,
<sup>2</sup>Insta360 Research,
<sup>3</sup>Wuhan University

---

## TLDR

STRNet provides a full pipeline for visual navigation with topological maps, covering dataset preparation, model training, and ROS-based deployment on mobile robots.

---

## Key Features
- Unified training and deployment structure for easy reproduction and transfer
- Topomap construction from trajectories for navigation
- End-to-end ROS scripts and configuration for deployment

---

## Directory Overview

```
strnet/
├── train/                         # Training, data processing, configs
│   ├── train.py                   # Training entry point
│   ├── process_*.py               # Data preprocessing scripts
│   ├── config/                    # Training configs
│   └── train_environment.yml      # Training environment
├── deployment/                    # Deployment and inference
│   ├── src/                       # Deployment scripts and runtime
│   ├── config/                    # Camera/robot/model configs
│   ├── model_weights/             # .pth weights and .yaml configs
│   └── deployment_environment.yaml
└── README.md
```

---

## Setup

### Environment (Training)

```bash
conda env create -f train/train_environment.yml
conda activate strnet
pip install -e train/
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

### Environment (Deployment)

```bash
conda env create -f deployment/deployment_environment.yaml
conda activate strnet
pip install -e train/
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

---

## Data Preparation

1. Download public datasets:
   - [RECON](https://sites.google.com/view/recon-robot/dataset)
   - [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/)
   - [GoStanford2](https://cvgl.stanford.edu/gonet/dataset/)
   - [SACSoN](https://sites.google.com/view/sacson-review/huron-dataset)


2. Run preprocessing:
   ```bash
   python train/process_recon.py  # or process_bags.py
   python train/data_split.py --dataset <your_dataset_path>
   ```

3. Expected data format:
```
dataset_name/
├── traj1/
│   ├── 0.jpg ... T_1.jpg
│   └── traj_data.pkl
└── ...
```

4. Data split output:
```
train/vint_train/data/data_splits/
└── <dataset_name>/
    ├── train/traj_names.txt
    └── test/traj_names.txt
```

---

## Model Training

```bash
cd train/
python train.py -c config/strnetnew.yaml
```

For custom configurations, start from files in [train/config](train/config).

---

## Deployment

### 1. Prepare Model Weights

Place the weight and config files:
```
deployment/model_weights/strnet/strnet.pth
deployment/model_weights/strnet/strnet.yaml
```

Update `deployment/config/models.yaml` so `config_path` and `ckpt_path` match your files.

### 2. Build a Topological Map (Topomap)

From `deployment/src/`:
```bash
./record_bag.sh <bag_name>
./create_topomap.sh <topomap_name> <bag_filename>
```

Topomap images are saved under:
```
deployment/topomaps/images/<topomap_name>/
```

### 3. Run Navigation

```bash
./navigate_new.sh "--model <model_name> --dir <topomap_dir>"
```

Where:
- `<model_name>` is the key in `deployment/config/models.yaml`
- `<topomap_dir>` is a folder name under `deployment/topomaps/images/`

---

## Hardware Notes

The deployment stack assumes a ROS-based mobile robot with RGB camera input. Adjust robot and camera topics in `deployment/config/robot.yaml` and `deployment/config/camera_*.yaml` to match your platform.

---

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgment

STRNet is inspired by the contributions of the following works to the open-source community: [NoMaD](https://github.com/robodhruv/visualnav-transformer), and [GreedyViG](https://github.com/SLDGroup/GreedyViG). We thank the authors for sharing their outstanding work.
