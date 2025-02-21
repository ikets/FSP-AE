# FSP-AE
[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3921/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.6-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

<div align="center">
    <img src="https://github.com/ikets/FSP-AE/blob/main/figure/spatial_upsampling.png" width=90%>
</div>

This repository contains the official implementation of **"Spatial Upsampling of Head-Related Transfer Function Using Neural Network Conditioned on Source Position and Frequency" [[PDF]]()** published in *IEEE Open Journal of Signal Processing*.<br>

If you use this code in your experiments, please cite [1] in your work.

## System Requirements
Tested on the following environment:
- OS: Ubuntu 22.04, 24.04
- GPU: NVIDIA RTX A5000 (24GB VRAM)
- CUDA: cuda 12.4
- Docker: version 27.5.1

If not using Docker, use Python 3.9. Required libraries are listed in [`requirements.txt`](https://github.com/ikets/FSP-AE/blob/main/requirements.txt).

## Installation (with Docker)
1. Install `Docker`.
2. Clone the repository and navigate to the top directory:
    ```
    $ git clone git@github.com:ikets/FSP-AE.git
    $ cd FSP-AE
    ```
3. Build Docker image:
    ```
    $ bash sh/docker_build.sh
    ```
4. Run Docker container:
    ```bash
    $ bash sh/docker_run.sh
    ```
5. Download and extract files from [the HUTUBS HRTF database](http://dx.doi.org/10.14279/depositonce-8487) [2,3] and [the RIEC HRTF Dataset](http://www.riec.tohoku.ac.jp/pub/hrtf/index.html) [4].
    ```
    root@foobar:/app# bash sh/prepare_data.sh
    ```
## Installation (with pyenv-virtualenv)
1. Install `pyenv` and `pyenv-virtualenv`.
2. Clone the repository and navigate to the top directory:
    ```
    $ git clone git@github.com:ikets/FSP-AE.git
    $ cd FSP-AE
    ```
3. Setup the virtual environment:
    ```
    $ pyenv install 3.9.21
    $ pyenv virtualenv 3.9.21 fsp_ae
    $ pyenv activate fsp_ae
    $ pip install --upgrade pip
    $ pip install -r requirements.txt
    ```
4. Download and extract files from [the HUTUBS HRTF database](http://dx.doi.org/10.14279/depositonce-8487) [2,3] and [the RIEC HRTF Dataset](http://www.riec.tohoku.ac.jp/pub/hrtf/index.html) [4].
    ```
    $ bash sh/prepare_data.sh
    ```
## Testing
- Test our pretrained model:  
  ```
  $ python3 test.py --exp_dir exp/v1 --device cpu
  ```

- Test the baseline methods:  
  ```
  $ python3 test_baseline.py --config_path config/baseline_v1.yaml --exp_dir exp_baseline
  ```

- Plot test result figures:  
  ```
  $ python3 plot_results.py --exp_dir_proposed exp/v1 --exp_dir_baseline exp_baseline --out_dir figure
  ```
or
- Sequentially run the above commands:
  ```
  $ bash reproduce_results.sh
  ```

## Training
- Train a new model:
  ```
  root@foobar:/app# python3 train.py --config_path config/<your_config>.yaml --device cuda
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

