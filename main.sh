# python3 train.py -c config/v1.yaml -d cuda
python3 test.py --e exp/v1 -d cpu
python3 test_baseline.py -c config/baseline_v1.yaml -e exp_baseline
python3 plot_results.py --exp_dir_proposed exp/v1 --exp_dir_baseline exp_baseline