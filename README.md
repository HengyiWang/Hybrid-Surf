# Hybrid-Surf
### [Paper](./thesis.pdf) | [Colab](https://colab.research.google.com/drive/1KlV5dZcD1l9_HPtf4Q_GvXDLv77MovzC?usp=sharing) 

<p align="center">
  <a href="">
    <img src="./media/hybrid-surf.gif" alt="Logo" width="60%">
  </a>
</p>

This repository contains the implementation of my MSc thesis on Neural Representations for 3D reconstructions, which uses a joint coordinate and sparse parametric encoding for RGB-D surface reconstruction. 

During that time, I drew inspiration from the pose optimization of [NeuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) and [GO-Surf](https://jingwenwang95.github.io/go_surf/), which involved sampling rays from all frames. This led me to extend my MSc thesis to a neural SLAM approach that eliminates the need for keyframe selection and optimizes on all previous keyframe rays (similar to [neuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) and [GO-Surf](https://jingwenwang95.github.io/go_surf/)) in the mapping process as described in our [Co-SLAM](https://hengyiwang.github.io/projects/CoSLAM.html) paper. 



## Installation

Please follow the instructions below to install the repo and dependencies.

```bash
git clone https://github.com/HengyiWang/Hybrid-Surf.git
cd Hybrid-Surf
```

### Install the environment

```bash
# Create conda environment
conda create -n hybridsurf python=3.7
conda activate hybridsurf

# Install the pytorch first (Please check the cuda version)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install all the dependencies via pip (Note here pytorch3d and tinycudann requires ~10min to build)
pip install -r requirements.txt

# Build extension (marching cubes from neuralRGBD)
cd external/NumpyMarchingCubes
python setup.py install
```



## Dataset
Download the NeuralRGBD dataset & mesh via:
```bash
 bash scripts/download_rgbd.sh
```



## Run
Run mesh culling for evaluation of the reconstruction via:

```bash
# GO-Surf culling strategy
python cull_mesh.py --scene morning_apartment
```



Run our code on NeuralRGBD dataset via:

```bash
python offline-surf.py --config configs/morning_apartment.txt   --tcnn_encoding --basedir './demo/' --geometric_init 300 --trainskip 1 --lrate 0.01 --lrate_decay 10
```



## Acknowlegement

This paper adapts the code from [neuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction), and evaluation script of [GO-Surf](https://jingwenwang95.github.io/go_surf/). Thanks for making the code available. I would also like to thank [HashNeRF-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch) and [torch-ngp](https://github.com/ashawkey/torch-ngp), which provides an excellent starting point for using [instant-ngp](https://github.com/NVlabs/instant-ngp). 

I would like to thank my supervisor, Prof. Lourdes Agapito, for her help and guidance throughout the project. Additionally, I would like to thank Mr. [Jingwen Wang](https://jingwenwang95.github.io/) for providing the [GO-Surf](https://github.com/JingwenWang95/go-surf) implementation for this project, as well as his help with the computational resources, evaluation strategy, and speed analysis during the extension of this project to neural SLAM. His contributions were important to the success of our [Co-SLAM](https://github.com/HengyiWang/Co-SLAM) paper.

I would also like to thank my fellow master students [Chen Liu](https://ryushinn.github.io/), [Chenghao Wu](https://theo-wu.github.io/), and [Weilue Luo](https://weilueluo.com/) for the helpful discussions during the extension of this project.

## Citation

if you find this project  is helpful to your research, please consider cite:
```
@MastersThesis{wang2022msc,
    author     =     {Wang, Hengyi},
    title     =     {Neural Reprensetations for 3D Reconstruction},
    school     =     {University College London (UCL)},
    year     =     {2022},
    month     =     {September}
}
```

