#!/bin/bash

python3 test.py --exp_dir exp/v1 --device cpu
python3 test_baseline.py --config_path config/baseline_v1.yaml --exp_dir exp_baseline
python3 plot_results.py --exp_dir_proposed exp/v1 --exp_dir_baseline exp_baseline --out_dir figure

python3 test.py --exp_dir exp/v1_linear --device cpu
python3 test.py --exp_dir exp/v1_wo-freq-cond --device cpu
python3 plot_results_ablation_only.py