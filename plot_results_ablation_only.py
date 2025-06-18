import argparse
import matplotlib.pyplot as plt
import os
import torch as th
from torch.utils.data import DataLoader
import yaml

from dataset import HRTFDataset
from sampling import sample_uniform
from utils import AttrDict


def get_suffix_list():
    uniform_num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    plane_axes_list = [(2,), (1, 2), (0, 1, 2)]
    plane_parallel_axis_list = [2]

    suffix_list = []
    for uniform_num_mes_pos in uniform_num_mes_pos_list:
        suffix_list.append(f"uniform-{uniform_num_mes_pos}")
    for plane_axes in plane_axes_list:
        plane_axes_str = ""
        for axis in plane_axes:
            plane_axes_str += f"{['x', 'y', 'z'][axis]}"
        suffix_list.append(f"plane-{plane_axes_str}")
    for plane_parallel_axis in plane_parallel_axis_list:
        suffix_list.append(f"plane-parallel-{['x', 'y', 'z'][plane_parallel_axis]}")
    return suffix_list


def concat_pt_files(pt_list):
    ret = []
    for pt in pt_list:
        ret.append(th.load(pt).unsqueeze(0))
    ret = th.cat(ret, dim=0)
    return ret


def get_pt_list(prefix, suffix, database_name, sub_id_range):
    pt_list = [f"{prefix}_{database_name}_sub-{sub_id}_{suffix}.pt" for sub_id in range(sub_id_range[0] + 1, sub_id_range[1] + 1)]
    return pt_list


def get_hrtf_mag_dict(suffix_list, config, dirs, labels):
    hrtf_mag_dict = {}
    for database_name in ["hutubs", "riec"]:
        hrtf_mag_dict[database_name] = {}
        for suffix in suffix_list:
            hrtf_mag_dict[database_name][suffix] = {}
            sub_id_range = config.data[database_name]["sub_id"]["test"]
            # proposed
            for i, (dir, label) in enumerate(zip(dirs, labels)):
                pt_list = get_pt_list(f"{dir}/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
                hrtf_mag_dict[database_name][suffix][label] = concat_pt_files(pt_list)
            i += 1
            # swfe
            pt_list = get_pt_list(f"{dirs[i]}/swfe/hrtf/hrtf", suffix, database_name, sub_id_range)
            hrtf_mags = 20 * th.log10(th.abs(concat_pt_files(pt_list)))
            hrtf_mag_dict[database_name][suffix]["SWFE"] = hrtf_mags
            # rshe
            pt_list = get_pt_list(f"{dirs[i]}/rshe/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["RSHE"] = concat_pt_files(pt_list)
            # spca
            pt_list = get_pt_list(f"{dirs[i]}/spca/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["SPCA"] = concat_pt_files(pt_list)
    return hrtf_mag_dict


def plot_lsd_uniform(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path, labels):
    plt.style.use("seaborn-v0_8-colorblind")
    plt.figure(figsize=(8, 5))
    fs = 16
    plt.rcParams["font.size"] = fs
    num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    num_mes_pos_list_actual = []
    for num_mes_pos in num_mes_pos_list:
        _, idx_mes = sample_uniform(database_info[database_name]["srcpos_cart"], num_mes_pos, radius=database_info[database_name]["srcpos_sph"][0, 0], dataset_name=database_name)
        num_mes_pos_list_actual.append(len(idx_mes))

    hrtf_mag_gt = hrtf_mag_dict_gt[database_name]
    values = {}
    format = {"Proposed": "-o", "SWFE": "-s", "RSHE": "-P", "SPCA": "-D"}
    for l, label in enumerate(labels[1:]):
        format[label] = ["--p", "--h"][l]
    methods = format.keys()

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for m, method in enumerate(methods):
        if m in [1, 2, 3]:
            pass
        else:
            values[method] = {"mean": [], "std": []}
            for num_mes_pos in num_mes_pos_list:
                suffix = f"uniform-{num_mes_pos}"
                hrtf_mag_pred = hrtf_mag_dict[database_name][suffix][method]
                lsd = th.sqrt(th.mean((hrtf_mag_gt - hrtf_mag_pred)**2, dim=3))
                values[method]["mean"].append(th.mean(lsd))
                values[method]["std"].append(th.std(lsd))
            plt.errorbar(num_mes_pos_list_actual, values[method]["mean"], yerr=values[method]["std"], capsize=5, fmt=format[method], label=method.replace("-", "\u2212"), mec="black", ms=10, color=cycle[m])

    plt.ylabel("LSD (dB)", fontsize=fs + 2)
    plt.xlabel('Number of measurement positions ' + r'$B^{(\mathrm{m})}$', fontsize=fs + 2)
    plt.legend(bbox_to_anchor=(-0.1, 1.15), loc="upper left", borderaxespad=0, ncol=3, fancybox=False, frameon=False, fontsize=fs)
    plt.grid()
    plt.xscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")
    plt.close()


def plot_lsd(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, dirs, labels):
    for database_name in database_info:
        save_path = f"{dirs[-1]}/fig/lsd_{database_name}.pdf"
        plot_lsd_uniform(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path, labels)


def main(args):
    with open(f"{args.exp_dirs_proposed[0]}/config.yaml") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)

    test_dataset = HRTFDataset(config.data, ["all"], (config.data.hutubs.sub_id.test, config.data.riec.sub_id.test))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    database_info = {}
    hrtf_mag_dict_gt = {}
    itd_dict_gt = {}
    for dataset_name in ["hutubs", "riec"]:
        database_info[dataset_name] = {}
        hrtf_mag_dict_gt[dataset_name] = []
        itd_dict_gt[dataset_name] = []
    for data, _ in test_loader:
        hrtf_mag, itd, dataset_name = data["hrtf_mag"], data["itd"], data["dataset_name"][0]
        for key in ["frequency", "srcpos_sph", "srcpos_cart"]:
            if key not in database_info[dataset_name]:
                database_info[dataset_name][key] = data[key][0]
        hrtf_mag_dict_gt[dataset_name].append(hrtf_mag)
        itd_dict_gt[dataset_name].append(itd)
    for k, v in hrtf_mag_dict_gt.items():
        hrtf_mag_dict_gt[k] = th.cat(v, dim=0)
    for k, v in itd_dict_gt.items():
        itd_dict_gt[k] = th.cat(v, dim=0)

    dirs = args.exp_dirs_proposed + [args.exp_dir_baseline, args.out_dir]
    labels = args.labels_proposed
    suffix_list = get_suffix_list()
    hrtf_mag_dict_pred = get_hrtf_mag_dict(suffix_list, config, dirs, labels)

    plot_lsd(database_info, hrtf_mag_dict_pred, hrtf_mag_dict_gt, dirs, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs_proposed", default=["exp/v1", "exp/v1_linear", "exp/v1_wo-freq-cond"])
    parser.add_argument("--labels_proposed", default=["Proposed", "Proposed-NL", "Proposed-FC"])
    parser.add_argument("--exp_dir_baseline", default="exp_baseline")
    parser.add_argument("--out_dir", default="figure_ablation_only")

    args = parser.parse_args()
    main(args)
