# FSP-AE
[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![pytorch](https://img.shields.io/badge/PyTorch_1.9-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

<div align="center">
    <img src="https://github.com/ikets/FSP-AE/blob/main/figure/spatial_upsampling.png" width=90%>
</div>

This repository contains the official implementation of **"Spatial Upsampling of Head-Related Transfer Function Using Neural Network Conditioned on Source Position and Frequency" [[PDF]]()** published in *IEEE Open Journal of Signal Processing*.<br>

If you use this code in your experiments, please cite [1] in your work.

## Requirements
We checked the code with the following computational environment.
- Ubuntu 20.04.2 LTS
- GeForce RTX 3090 (24GB VRAM)
- Python 3.8.10

See `requirements.txt` for the required Python libraries.
## Tutorial on Colab
You can test our pretrained model with [this short tutorial notebook](https://colab.research.google.com/github/ikets/FAP-AE/blob/main/notebook/tutorial.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ikets/FAP-AE/blob/main/notebook/tutorial.ipynb)
## Installation
1. Clone this repository.
2. Download [the HUTUBS HRTF database](http://dx.doi.org/10.14279/depositonce-8487) [2,3], and extract `HRIRs.zip` into `data/hutubs/HRIRs`.
3. Download [the RIEC HRTF Dataset](http://www.riec.tohoku.ac.jp/pub/hrtf/index.html) [4], and extract `RIEC_hrtf_all.zip` into `data/riec`.
4. Set up the Python environment. For example, if you use `pyenv-virtualenv`, run:
    ```sh
    pyenv install 3.8.10
    pyenv virtualenv 3.8.10 fsp_ae
    pyenv activate fsp_ae
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Testing
- To test our pretrained model, run:  
  ```sh
  python3 test.py --exp_dir exp/v1 --device cpu
  ```

- To test baseline methods, run:  
  ```sh
  python3 test_baseline.py --config_path config/baseline_v1.yaml --exp_dir exp_baseline
  ```

- To plot test result figures, run:  
  ```sh
  python3 plot_results.py --exp_dir_proposed exp/v1 --exp_dir_baseline exp_baseline
  ```


## Training
- To train a new model, run:
  ```sh
  python3 train.py --config_path config/<your_config>.yaml --device cuda
  ```

## Note
- `sampling/grids/grid_t*d*.mat` contains 3D Cartesian coordinates of points in a spherical *t*-design [5], obtained from [6].

## Cite
```bibtex
@article{Ito:OJSP2025,
  author    = {Yuki Ito and Tomohiko Nakamura and Shoichi Koyama and Shuichi Sakamoto and Hiroshi Saruwatari},
  title     = {Spatial Upsampling of Head-Related Transfer Function Using Neural Network Conditioned on Source Position and Frequency},
  journal   = {IEEE Open Journal of Signal Processing},
  year      = {2025},
  volume    = {},
  pages     = {},
  publisher = {IEEE},
  doi       = {}
}
```

## License
[CC-BY-4.0](https://github.com/ikets/FSP-AE/blob/main/LICENSE)

## References
[1] Yuki Ito, Tomohiko Nakamura, Shoichi Koyama, Shuichi Sakamoto, and Hiroshi Saruwatari, <strong>“Spatial Upsampling of Head-Related Transfer Function Using Neural Network Conditioned on Source Position and Frequency,”</strong> <em>IEEE Open J. Signal Process.</em>, vol. XX, pp. xxxx-xxxx, 2025. [[PDF]]() <br>

[2] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Peter Grosche,  Daniel Voss, and Stefan Weinzierl, “A cross-evaluated database of measured and simulated HRTFs including 3D head meshes, anthropometric features, and headphone impulse responses”, <em>J. Audio Eng. Soc.</em>, vol. 67, no. 9, pp. 705–718, 2019.<br>

[3] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Jan Joschka Wohlgemuth, Fabian Seipel, Daniel Voss, Peter Grosche, and Stefan Weinzierl, “The HUTUBS head-related transfer function (HRTF) database,” 2019, url: http://dx.doi.org/10.14279/depositonce-8487 (accessed May 6, 2022).<br>

[4] Kanji Watanabe, Yukio Iwaya, Yôiti Suzuki, Shouichi Takane, and Sojun Sato, “Dataset of head-related transfer functions measured with a circular loudspeaker array,” <em>Acoust. Sci. Tech.</em>, vol. 35, no. 3, pp. 159–165, 2014.<br>

[5] X. Chen and R. S. Womersley, “Existence of solutions to
systems of underdetermined equations and spherical designs,”
<em>SIAM J. Numer. Anal.,</em> vol. 44, no. 6, pp. 2326–2341, 2006.

[6] https://www.polyu.edu.hk/ama/staff/xjchen/sphdesigns.html (accessed Sep. 18, 2022)

