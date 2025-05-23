# MPLNet:  Multi-granularity Prompt Learning with Vision-Language Model for Remote Sensing Image Scene Classification

![model](https://github.com/user-attachments/assets/43caadd2-2aac-4d7a-aef9-6acd8fa80180)
## installation

To install mplnet please use the following commands:

```bash
conda create -n env_name python=3.8
conda activate env_name
pip install -U pip
pip install -e .
```

Special thanks to [GalLoP](https://github.com/MarcLafon/gallop) for their powerful module that improved our code efficiency.

## Training

We provide training scripts in the `scripts` folder. For instance, 1-shot training and testing on NWPU_RESISC45:

```bash
bash scripts/run_resisc45.sh 1
```

