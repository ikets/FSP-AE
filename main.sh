# python3 train.py -c config/v1.yaml -d "cuda"
python3 test.py -c config/v1.yaml -d "cpu"
python3 test_baseline.py -c config/v1_baseline.yaml -e exp_baseline
python3 plot_results.py