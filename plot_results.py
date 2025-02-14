from attrdict import AttrDict
import matplotlib.pyplot as plt
import numpy as np
import os
import torch as th
from torch.utils.data import DataLoader
import yaml

from dataset import HRTFDataset
from sampling import sample_uniform


def concat_pt_files(pt_list):
    ret = []
    for pt in pt_list:
        ret.append(th.load(pt).unsqueeze(0))
    ret = th.cat(ret, dim=0)
    return ret


def get_pt_list(prefix, suffix, database_name, sub_id_range):
    pt_list = [f"{prefix}_{database_name}_sub-{sub_id}_{suffix}.pt" for sub_id in range(sub_id_range[0] + 1, sub_id_range[1] + 1)]
    return pt_list


def get_hrtf_mag_dict(suffix_list, config):
    hrtf_mag_dict = {}
    for database_name in ["hutubs", "riec"]:
        hrtf_mag_dict[database_name] = {}
        for suffix in suffix_list:
            hrtf_mag_dict[database_name][suffix] = {}
            # proposed
            sub_id_range = config.data[database_name]["sub_id"]["test"]
            pt_list = get_pt_list("exp/v1/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["Proposed"] = concat_pt_files(pt_list)
            # swfe
            pt_list = get_pt_list("exp_baseline/swfe/hrtf/hrtf", suffix, database_name, sub_id_range)
            hrtf_mags = 20 * th.log10(th.abs(concat_pt_files(pt_list)))
            hrtf_mag_dict[database_name][suffix]["SWFE"] = hrtf_mags
            # rshe
            pt_list = get_pt_list("exp_baseline/rshe/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["RSHE"] = concat_pt_files(pt_list)
            # spca
            pt_list = get_pt_list("exp_baseline/spca/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["SPCA"] = concat_pt_files(pt_list)
    return hrtf_mag_dict


def plot_lsd_uniform(database_data, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path):
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.size"] = 20
    num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    num_mes_pos_list_actual = []
    for num_mes_pos in num_mes_pos_list:
        _, idx_mes = sample_uniform(database_data[database_name]["srcpos_cart"], num_mes_pos, radius=database_data[database_name]["srcpos_sph"][0, 0], dataset_name=database_name)
        num_mes_pos_list_actual.append(len(idx_mes))

    hrtf_mag_gt = hrtf_mag_dict_gt[database_name]
    values = {}
    format = {"Proposed": "-o", "SWFE": "-s", "RSHE": "-P", "SPCA": "-D"}
    for method in ["Proposed", "SWFE", "RSHE", "SPCA"]:
        values[method] = {"mean": [], "std": []}
        for num_mes_pos in num_mes_pos_list:
            suffix = f"uniform-{num_mes_pos}"
            hrtf_mag_pred = hrtf_mag_dict[database_name][suffix][method]
            lsd = th.sqrt(th.mean((hrtf_mag_gt - hrtf_mag_pred)**2, dim=3))
            values[method]["mean"].append(th.mean(lsd))
            values[method]["std"].append(th.std(lsd))
        plt.errorbar(num_mes_pos_list_actual, values[method]["mean"], yerr=values[method]["std"], capsize=5, fmt=format[method], label=method, mec="black", ms=10)

    plt.ylabel("LSD (dB)")
    plt.xlabel('Number of measurement positions ' + r'$B^{(\mathrm{m})}$')
    plt.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False)
    plt.grid()
    plt.xscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")


def plot_ae_ild_uniform(database_data, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path):
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.size"] = 20
    num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    num_mes_pos_list_actual = []
    for num_mes_pos in num_mes_pos_list:
        _, idx_mes = sample_uniform(database_data[database_name]["srcpos_cart"], num_mes_pos, radius=database_data[database_name]["srcpos_sph"][0, 0], dataset_name=database_name)
        num_mes_pos_list_actual.append(len(idx_mes))

    hrtf_mag_gt = hrtf_mag_dict_gt[database_name]
    values = {}
    format = {"Proposed": "-o", "SWFE": "-s", "RSHE": "-P", "SPCA": "-D"}
    for method in ["Proposed", "SWFE", "RSHE", "SPCA"]:
        values[method] = {"mean": [], "std": []}
        for num_mes_pos in num_mes_pos_list:
            suffix = f"uniform-{num_mes_pos}"
            hrtf_mag_pred = hrtf_mag_dict[database_name][suffix][method]

            hrtf_mag_pred_ = th.cat((hrtf_mag_pred, th.flip(hrtf_mag_pred[:, :, :, :-2], dims=(3,))), dim=3)
            hrtf_mag_gt_ = th.cat((hrtf_mag_gt, th.flip(hrtf_mag_gt[:, :, :, :-2], dims=(3,))), dim=3)
            ild_pred = 10 * th.log10(th.sum((10**(hrtf_mag_pred_[:, :, 0, :] / 20))**2, dim=2) / th.sum((10**(hrtf_mag_pred_[:, :, 1, :] / 20))**2, dim=2))
            ild_gt = 10 * th.log10(th.sum((10**(hrtf_mag_gt_[:, :, 0, :] / 20))**2, dim=2) / th.sum((10**(hrtf_mag_gt_[:, :, 1, :] / 20))**2, dim=2))
            ild_ae = th.abs(ild_pred - ild_gt)

            values[method]["mean"].append(th.mean(ild_ae))
            values[method]["std"].append(th.std(ild_ae))
        plt.errorbar(num_mes_pos_list_actual, values[method]["mean"], yerr=values[method]["std"], capsize=5, fmt=format[method], label=method, mec="black", ms=10)

    plt.ylabel(r"$\mathrm{AE}^{(\mathrm{ILD})}$" + " (dB)")
    plt.xlabel('Number of measurement positions ' + r'$B^{(\mathrm{m})}$')
    plt.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False)
    plt.grid()
    plt.xscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")


def plot_fig6(database_data, hrtf_mag_dict, hrtf_mag_dict_gt):
    for database_name in database_data:
        save_path = f"figure/fig6/lsd_{database_name}.pdf"
        plot_lsd_uniform(database_data, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path)
        save_path = f"figure/fig6/ae_ild_{database_name}.pdf"
        plot_ae_ild_uniform(database_data, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path)


def plot_fig8(database_data, hrtf_mag_dict_gt):
    lr = 0
    s = 0
    database_name = "hutubs"
    basename = "_hutubs_sub-89_uniform-9.pt"

    pos_cart_tar = database_data[database_name]["srcpos_cart"]
    zeni = database_data[database_name]["srcpos_sph"][:, 2]

    b_list_front = th.arange(pos_cart_tar.shape[0])[th.abs(pos_cart_tar[:, 1]) < 0.02 * (pos_cart_tar[:, 0] > -1e-3)].tolist()
    b_list_back = th.flip(th.arange(pos_cart_tar.shape[0])[th.abs(pos_cart_tar[:, 1]) < 0.02 * (pos_cart_tar[:, 0] < 0)], dims=(0,)).tolist()
    b_list_back = b_list_back + [b_list_back[-1]]
    b_list = b_list_back + b_list_front
    ylist = np.arange(2, len(b_list), 6).tolist()

    for method in ["ground_truth", "proposed", "swfe", "rshe", "spca"]:
        if method == "ground_truth":
            hrtf_mag_pred = hrtf_mag_dict_gt[database_name][s, :, :, :]
        elif method == "proposed":
            hrtf_mag_pred = th.load(f"exp/v1/hrtf_mag/hrtf_mag{basename}")
        elif method == "swfe":
            hrtf_pred = th.load(f"exp_baseline/swfe/hrtf/hrtf{basename}")
            hrtf_mag_pred = 20 * th.log10(th.abs(hrtf_pred))
        else:
            hrtf_mag_pred = th.load(f"exp_baseline/{method}/hrtf_mag/hrtf_mag{basename}")

        hrtf_mag_plot = hrtf_mag_pred[b_list, lr, :]

        plt.rcParams["font.size"] = 20
        figsize = (10.5, 6)

        plt.figure(figsize=figsize)
        plt.imshow(hrtf_mag_plot, vmax=5, vmin=-25, aspect="auto", cmap="viridis")
        plt.colorbar(label="Magnitude (dB)")

        plt.xlim([-1, 127])
        plt.xticks(np.arange(0, 128 + 1, 16) - 1, (np.arange(0, 128 + 1, 16) * 125 / 1000).astype(int))
        plt.xlabel("Frequency (kHz)")
        plt.yticks(ylist, [f"{zeni[b].item()/np.pi*180:.0f}" for b in th.tensor(b_list)[ylist]])
        plt.ylabel("Zenith (deg)", labelpad=30)

        plt.text(-20, 10, "Back", rotation="vertical")
        plt.text(-20, 28, "Front", rotation="vertical")

        save_path = f"figure/fig8/{method}.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Saved to {save_path}.")


def plot_all():
    with open("./config/v1.yaml") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)

    test_dataset = HRTFDataset(config.data, ["all"], (config.data.hutubs.sub_id.test, config.data.riec.sub_id.test))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    database_data = {}
    hrtf_mag_dict_gt = {}
    itd_dict_gt = {}
    for dataset_name in ["hutubs", "riec"]:
        database_data[dataset_name] = {}
        hrtf_mag_dict_gt[dataset_name] = []
        itd_dict_gt[dataset_name] = []

    for data, _ in test_loader:
        hrtf_mag, itd, dataset_name = data["hrtf_mag"], data["itd"], data["dataset_name"][0]
        for key in ["frequency", "srcpos_sph", "srcpos_cart"]:
            if key not in database_data[dataset_name]:
                database_data[dataset_name][key] = data[key][0]
        hrtf_mag_dict_gt[dataset_name].append(hrtf_mag)
        itd_dict_gt[dataset_name].append(itd)

    for k, v in hrtf_mag_dict_gt.items():
        hrtf_mag_dict_gt[k] = th.cat(v, dim=0)
    for k, v in itd_dict_gt.items():
        itd_dict_gt[k] = th.cat(v, dim=0)

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
    hrtf_mag_dict = get_hrtf_mag_dict(suffix_list, config)

    plot_fig6(database_data, hrtf_mag_dict, hrtf_mag_dict_gt)

    plot_fig8(database_data, hrtf_mag_dict_gt)


if __name__ == '__main__':
    plot_all()
