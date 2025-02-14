import argparse
import os
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HRTFDataset
from model import FreqSrcPosCondAutoEncoder
from sampling import sample_uniform, sample_plane, sample_plane_parallel
from utils import load_yaml, get_hrir_with_itd


def load_state(path, model):
    state_dicts = th.load(path)
    model.load_state_dict(state_dicts["model"])
    model.stats = state_dicts["stats"]
    print(f"Loaded checkpoint {path}.")
    return model


def infer_and_save(model, hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device, suffix, exp_dir, config):
    with th.no_grad():
        hrtf_mag_pred, itd_pred = model(hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device)
        hrir_pred = get_hrir_with_itd(hrtf_mag_pred, itd_pred, input_type="hrtf_mag", fs=config.data.max_freq * 2, fs_up=config.data.fs_up)

    th.save(hrtf_mag_pred, f"{exp_dir}/hrtf_mag/hrtf_mag_{suffix}.pt")
    th.save(itd_pred, f"{exp_dir}/itd/itd_{suffix}.pt")
    th.save(hrir_pred, f"{exp_dir}/hrir/hrir_{suffix}.pt")
    print(f"Saved in {exp_dir}/<data_type>/<data_type>_{suffix}.pt. (`data_type`: hrtf_mag, itd, hrir)")


def test(args):
    config = load_yaml(args.config_path)
    device = args.device
    exp_dir = f"exp/{os.path.basename(args.config_path).split('.')[0]}"
    os.makedirs(f"{exp_dir}/hrtf_mag", exist_ok=True)
    os.makedirs(f"{exp_dir}/itd", exist_ok=True)
    os.makedirs(f"{exp_dir}/hrir", exist_ok=True)

    test_dataset = HRTFDataset(config.data, ["all"], (config.data.hutubs.sub_id.test, config.data.riec.sub_id.test))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = FreqSrcPosCondAutoEncoder(config.architecture)
    model = load_state(f"{exp_dir}/checkpoint_best_manual.pt", model)
    model.eval()

    uniform_num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    plane_axes_list = [(2,), (1, 2), (0, 1, 2)]
    plane_parallel_axis_list = [2]

    for data, _ in tqdm(test_loader):
        hrtf_mag, itd, freq, pos_cart_tar, radius, dataset_name, sofa_path = data["hrtf_mag"], data["itd"], data["frequency"], data["srcpos_cart"], data["srcpos_sph"][0, 0], data["dataset_name"][0], data["sofa_path"][0]

        if dataset_name == "hutubs":
            sub_id = os.path.basename(sofa_path).split("_")[0].replace("pp", "")
        elif dataset_name == "riec":
            sub_id = os.path.basename(sofa_path).split("_")[-1].replace(".sofa", "")
        sub_id = str(int(sub_id))

        for uniform_num_mes_pos in uniform_num_mes_pos_list:
            suffix = f"{dataset_name}_sub-{sub_id}_uniform-{uniform_num_mes_pos}"
            pos_cart_mes, idx_mes = sample_uniform(pos_cart_tar[0], uniform_num_mes_pos, radius, dataset_name)

            pos_cart_mes, hrtf_mag_mes, itd_mes = pos_cart_mes.unsqueeze(0), hrtf_mag[:, idx_mes, :, :], itd[:, idx_mes]
            infer_and_save(model, hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device, suffix, exp_dir, config)

        for plane_axes in plane_axes_list:
            plane_axes_str = ""
            for axis in plane_axes:
                plane_axes_str += f"{['x', 'y', 'z'][axis]}"
            suffix = f"{dataset_name}_sub-{sub_id}_plane-{plane_axes_str}"
            pos_cart_mes, idx_mes = sample_plane(pos_cart_tar[0], plane_axes)

            pos_cart_mes, hrtf_mag_mes, itd_mes = pos_cart_mes.unsqueeze(0), hrtf_mag[:, idx_mes, :, :], itd[:, idx_mes]
            infer_and_save(model, hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device, suffix, exp_dir, config)

        for plane_parallel_axis in plane_parallel_axis_list:
            suffix = f"{dataset_name}_sub-{sub_id}_plane-parallel-{['x', 'y', 'z'][plane_parallel_axis]}"
            pos_cart_mes, idx_mes = sample_plane_parallel(pos_cart_tar[0], plane_parallel_axis, radius * th.tensor([-0.5, 0.0, 0.5]))

            pos_cart_mes, hrtf_mag_mes, itd_mes = pos_cart_mes.unsqueeze(0), hrtf_mag[:, idx_mes, :, :], itd[:, idx_mes]
            infer_and_save(model, hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device, suffix, exp_dir, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", default="./config/v1.yaml")
    parser.add_argument("-d", "--device", default="cuda")

    args = parser.parse_args()
    test(args)
