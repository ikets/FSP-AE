import argparse
import os
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HRTFDataset
from baseline import sph_wavefunc_expansion, real_sph_harm_expansion, spatial_principal_component_analysis, woodworth, calculate_spc_mat_and_mean_vec
from sampling import sample_uniform, sample_plane, sample_plane_parallel
from utils import load_yaml, get_hrir_with_itd


def prepare_directories(par_dir):
    dir_list = ["swfe/hrtf",
                "rshe/hrtf_mag",
                "rshe/itd",
                "spca/hrtf_mag",
                "woodworth/itd"]
    for dir in dir_list:
        os.makedirs(f"{par_dir}/{dir}", exist_ok=True)


def load_spca_dict():
    spca_dicts = {}
    for database_name in ["hutubs", "riec"]:
        path = f"exp_baseline/spca/hrtf_mag/{database_name}.pt"
        spca_dicts[database_name] = th.load(path)
        print(f"Loaded dict for SPCA from {path}.")
    return spca_dicts


def calc_and_save_spca_dict(train_dataset):
    print("Calculate and save dicts for SPCA.")
    database_names = ["hutubs", "riec"]
    hrtf_mags = {}
    pos_cart = {}
    for database_name in database_names:
        hrtf_mags[database_name] = []

    for sofa_path in train_dataset.sofa_paths:
        item = train_dataset.get_data(sofa_path)
        database_name = item["dataset_name"]
        hrtf_mags[database_name].append(item["hrtf_mag"].unsqueeze(0))
        if database_name not in pos_cart:
            pos_cart[database_name] = item["srcpos_cart"]  # (B_t, 3)

    spca_dicts = {}
    for database_name in database_names:
        spca_dicts[database_name] = {}

    for database_name in database_names:
        hrtf_mags_cat = th.cat(hrtf_mags[database_name], dim=0)
        hrtf_mags_cat = hrtf_mags_cat.permute(1, 2, 3, 0)  # (B_t, 2, L, S)
        spc_mat, mean_vec = calculate_spc_mat_and_mean_vec(hrtf_mags_cat, pos_cart[database_name])
        spca_dict = {"spc_mat": spc_mat, "mean_vec": mean_vec}
        spca_dicts[database_name] = spca_dict

        # save
        save_path = f"exp_baseline/spca/hrtf_mag/{database_name}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        th.save(spca_dict, save_path)
        print(f"Saved in {save_path}.")

    return spca_dicts


def infer_and_save(hrtf_mag, hrtf, itd, idx_mes, freq, pos_cart_tar, pos_sph_tar, spca_dicts, database_name, suffix, exp_dir, config):
    pos_sph_mes, pos_cart_mes = pos_sph_tar[idx_mes, :], pos_cart_tar[idx_mes, :]
    hrtf_mes, hrtf_mag_mes, itd_mes = hrtf[idx_mes, :, :, :], hrtf_mag[idx_mes, :, :, :], itd[idx_mes, :]
    hrtf_swfe = sph_wavefunc_expansion(hrtf_mes, freq, pos_sph_mes, pos_sph_tar, reg_param=config.swfe.hrtf.reg_param, coeff=config.swfe.hrtf.coeff, sound_speed=config.swfe.hrtf.sound_speed, r_shoulder=config.swfe.hrtf.r_shoulder)
    hrtf_mag_rshe = real_sph_harm_expansion(hrtf_mag_mes, pos_sph_mes, pos_sph_tar, "hrtf_mag", reg_param=config.rshe.hrtf_mag.reg_param, coeff=config.rshe.hrtf_mag.coeff)
    itd_rshe = real_sph_harm_expansion(itd_mes, pos_sph_mes, pos_sph_tar, "itd", reg_param=config.rshe.itd.reg_param, coeff=config.rshe.itd.coeff)
    hrtf_mag_spca = spatial_principal_component_analysis(hrtf_mag_mes, pos_cart_tar, idx_mes, spc_mat=spca_dicts[database_name]["spc_mat"], mean_vec=spca_dicts[database_name]["mean_vec"], reg_param=config.spca.hrtf_mag.reg_param, coeff=config.spca.hrtf_mag.coeff)
    itd_wdwt = woodworth(itd_mes, pos_cart_mes, pos_sph_mes, pos_cart_tar, pos_sph_tar)

    sub_id_range = config.data[database_name]["sub_id"]["test"]
    for s, sub_id in enumerate(range(sub_id_range[0] + 1, sub_id_range[1] + 1)):
        suffix_sub = f"{database_name}_sub-{sub_id}_{suffix}.pt"
        th.save(hrtf_swfe[:, :, :, s], f"{exp_dir}/swfe/hrtf/hrtf_{suffix_sub}")
        th.save(hrtf_mag_rshe[:, :, :, s], f"{exp_dir}/rshe/hrtf_mag/hrtf_mag_{suffix_sub}")
        th.save(itd_rshe[:, s], f"{exp_dir}/rshe/itd/itd_{suffix_sub}")
        th.save(hrtf_mag_spca[:, :, :, s], f"{exp_dir}/spca/hrtf_mag/hrtf_mag_{suffix_sub}")
        th.save(itd_wdwt[:, s], f"{exp_dir}/woodworth/itd/itd_{suffix_sub}")

        print(f"Saved in {exp_dir}/<method>/<data_kind>/<data_kind>_{suffix_sub}.")


def test(args):
    config = load_yaml(args.config_path)
    exp_dir = args.exp_dir

    prepare_directories(exp_dir)

    test_dataset = HRTFDataset(config.data, ["all"], (config.data.hutubs.sub_id.test, config.data.riec.sub_id.test))

    if args.load_spca_dict:
        spca_dicts = load_spca_dict()
    else:
        train_dataset = HRTFDataset(config.data, config.training.num_mes_pos_train, (config.data.hutubs.sub_id.train, config.data.riec.sub_id.train))
        spca_dicts = calc_and_save_spca_dict(train_dataset)

    uniform_num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    plane_axes_list = [(2,), (1, 2), (0, 1, 2)]
    plane_parallel_axis_list = [2]

    database_names = ["hutubs", "riec"]
    items = {}
    items_cat = {}
    for database_name in database_names:
        items[database_name] = {"hrtf_mag": [], "hrtf": [], "itd": []}
        items_cat[database_name] = {}
    for sofa_path in test_dataset.sofa_paths:
        item = test_dataset.get_data(sofa_path)
        database_name = item["dataset_name"]
        for data_kind in ["hrtf_mag", "hrtf", "itd"]:
            items[database_name][data_kind].append(item[data_kind].unsqueeze(-1))
        for key in ["frequency", "srcpos_sph", "srcpos_cart"]:
            if key not in items_cat[database_name]:
                items_cat[database_name][key] = item[key]

    for database_name in database_names:
        for data_kind in ["hrtf_mag", "hrtf", "itd"]:
            item_cat = th.cat(items[database_name][data_kind], dim=-1)  # (B_t, 2, L, S) or (B_t, S)
            items_cat[database_name][data_kind] = item_cat

    for database_name in database_names:
        hrtf, hrtf_mag, itd = items_cat[database_name]["hrtf"], items_cat[database_name]["hrtf_mag"], items_cat[database_name]["itd"]
        freq = items_cat[database_name]["frequency"]
        pos_sph_tar, pos_cart_tar = items_cat[database_name]["srcpos_sph"], items_cat[database_name]["srcpos_cart"]
        radius = pos_sph_tar[0, 0]

        for uniform_num_mes_pos in uniform_num_mes_pos_list:
            suffix = f"uniform-{uniform_num_mes_pos}"
            _, idx_mes = sample_uniform(pos_cart_tar, uniform_num_mes_pos, radius, database_name)
            infer_and_save(hrtf_mag, hrtf, itd, idx_mes, freq, pos_cart_tar, pos_sph_tar, spca_dicts, database_name, suffix, exp_dir, config)

        for plane_axes in plane_axes_list:
            plane_axes_str = ""
            for axis in plane_axes:
                plane_axes_str += f"{['x', 'y', 'z'][axis]}"
            suffix = f"plane-{plane_axes_str}"
            _, idx_mes = sample_plane(pos_cart_tar, plane_axes)
            infer_and_save(hrtf_mag, hrtf, itd, idx_mes, freq, pos_cart_tar, pos_sph_tar, spca_dicts, database_name, suffix, exp_dir, config)

        for plane_parallel_axis in plane_parallel_axis_list:
            suffix = f"plane-parallel-{['x', 'y', 'z'][plane_parallel_axis]}"
            _, idx_mes = sample_plane_parallel(pos_cart_tar, plane_parallel_axis, radius * th.tensor([-0.5, 0.0, 0.5]))
            infer_and_save(hrtf_mag, hrtf, itd, idx_mes, freq, pos_cart_tar, pos_sph_tar, spca_dicts, database_name, suffix, exp_dir, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", default="./config/baseline_v1.yaml")
    parser.add_argument("-e", "--exp_dir", default="exp_baseline")
    parser.add_argument("-l", "--load_spca_dict", action="store_true")

    args = parser.parse_args()
    test(args)
