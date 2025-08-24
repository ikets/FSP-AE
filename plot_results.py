import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch as th
from torch.utils.data import DataLoader
import yaml

from dataset import HRTFDataset
from sampling import sample_uniform
from utils import AttrDict, hrtf2hrir, hrir2itd


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


def get_hrtf_mag_dict(suffix_list, config, dirs):
    hrtf_mag_dict = {}
    for database_name in ["hutubs", "riec"]:
        hrtf_mag_dict[database_name] = {}
        for suffix in suffix_list:
            hrtf_mag_dict[database_name][suffix] = {}
            # proposed
            sub_id_range = config.data[database_name]["sub_id"]["test"]
            pt_list = get_pt_list(f"{dirs[0]}/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["Proposed"] = concat_pt_files(pt_list)
            # swfe
            pt_list = get_pt_list(f"{dirs[1]}/swfe/hrtf/hrtf", suffix, database_name, sub_id_range)
            hrtf_mags = 20 * th.log10(th.abs(concat_pt_files(pt_list)))
            hrtf_mag_dict[database_name][suffix]["SWFE"] = hrtf_mags
            # rshe
            pt_list = get_pt_list(f"{dirs[1]}/rshe/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["RSHE"] = concat_pt_files(pt_list)
            # spca
            pt_list = get_pt_list(f"{dirs[1]}/spca/hrtf_mag/hrtf_mag", suffix, database_name, sub_id_range)
            hrtf_mag_dict[database_name][suffix]["SPCA"] = concat_pt_files(pt_list)
    return hrtf_mag_dict


def get_itd_dict(suffix_list, config, dirs):
    itd_dict = {}
    for database_name in ["hutubs", "riec"]:
        itd_dict[database_name] = {}
        for suffix in suffix_list:
            itd_dict[database_name][suffix] = {}
            # proposed
            sub_id_range = config.data[database_name]["sub_id"]["test"]
            pt_list = get_pt_list(f"{dirs[0]}/itd/itd", suffix, database_name, sub_id_range)
            itd_dict[database_name][suffix]["Proposed"] = concat_pt_files(pt_list)
            # swfe
            pt_list = get_pt_list(f"{dirs[1]}/swfe/hrtf/hrtf", suffix, database_name, sub_id_range)
            itd_dict[database_name][suffix]["SWFE"] = hrir2itd(hrtf2hrir(concat_pt_files(pt_list)), fs=config.data.max_freq * 2, fs_up=config.data.fs_up)
            # rshe
            pt_list = get_pt_list(f"{dirs[1]}/rshe/itd/itd", suffix, database_name, sub_id_range)
            itd_dict[database_name][suffix]["RSHE"] = concat_pt_files(pt_list)
            # woodworth
            pt_list = get_pt_list(f"{dirs[1]}/woodworth/itd/itd", suffix, database_name, sub_id_range)
            itd_dict[database_name][suffix]["Woodworth"] = concat_pt_files(pt_list)
    return itd_dict


def plot_lsd_uniform(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path):
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
    for method in ["Proposed", "SWFE", "RSHE", "SPCA"]:
        values[method] = {"mean": [], "std": []}
        for num_mes_pos in num_mes_pos_list:
            suffix = f"uniform-{num_mes_pos}"
            hrtf_mag_pred = hrtf_mag_dict[database_name][suffix][method]
            lsd = th.sqrt(th.mean((hrtf_mag_gt - hrtf_mag_pred)**2, dim=3))
            values[method]["mean"].append(th.mean(lsd))
            values[method]["std"].append(th.std(lsd))
        plt.errorbar(num_mes_pos_list_actual, values[method]["mean"], yerr=values[method]["std"], capsize=5, fmt=format[method], label=method, mec="black", ms=10)

    plt.ylabel("LSD (dB)", fontsize=fs + 2)
    plt.xlabel('Number of measurement positions ' + r'$B^{(\mathrm{m})}$', fontsize=fs + 2)
    # plt.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False)
    plt.legend(bbox_to_anchor=(-0.1, 1.15), loc="upper left", borderaxespad=0, ncol=4, fancybox=False, frameon=False, fontsize=fs)
    plt.grid()
    plt.xscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")
    plt.close()


def plot_ae_ild_uniform(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path):
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

    plt.ylabel(r"$\mathrm{AE}^{(\mathrm{ILD})}$" + " (dB)", fontsize=fs + 2)
    plt.xlabel('Number of measurement positions ' + r'$B^{(\mathrm{m})}$', fontsize=fs + 2)
    # plt.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False)
    plt.legend(bbox_to_anchor=(-0.1, 1.15), loc="upper left", borderaxespad=0, ncol=4, fancybox=False, frameon=False, fontsize=fs)
    plt.grid()
    plt.xscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")
    plt.close()


def plot_ae_itd_uniform(database_info, itd_dict, itd_dict_gt, database_name, save_path):
    plt.style.use("seaborn-v0_8-colorblind")
    # plt.figure(figsize=(10, 6))
    # plt.rcParams["font.size"] = 20
    plt.figure(figsize=(8, 5))
    fs = 16
    plt.rcParams["font.size"] = fs
    num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 13)]
    num_mes_pos_list_actual = []
    for num_mes_pos in num_mes_pos_list:
        _, idx_mes = sample_uniform(database_info[database_name]["srcpos_cart"], num_mes_pos, radius=database_info[database_name]["srcpos_sph"][0, 0], dataset_name=database_name)
        num_mes_pos_list_actual.append(len(idx_mes))

    itd_gt = itd_dict_gt[database_name]
    values = {}
    format = {"Proposed": "-o", "SWFE": "-s", "RSHE": "-P", "Woodworth": "-D"}
    for method in ["Proposed", "SWFE", "RSHE", "Woodworth"]:
        values[method] = {"mean": [], "std": []}
        for num_mes_pos in num_mes_pos_list:
            suffix = f"uniform-{num_mes_pos}"
            itd_pred = itd_dict[database_name][suffix][method]
            itd_ae = th.abs(itd_pred - itd_gt)

            values[method]["mean"].append(th.mean(itd_ae))
            values[method]["std"].append(th.std(itd_ae))
        plt.errorbar(num_mes_pos_list_actual, np.array(values[method]["mean"]) * 1e6, yerr=np.array(values[method]["std"]) * 1e6, capsize=5, fmt=format[method], label=method, mec="black", ms=10)

    plt.ylabel(r"$\mathrm{AE}^{(\mathrm{ITD})}$" + " ($\mu$s)", fontsize=fs + 2)
    plt.xlabel('Number of measurement positions ' + r'$B^{(\mathrm{m})}$', fontsize=fs + 2)
    # plt.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False)
    # plt.legend(bbox_to_anchor=(-0.1, 1.15), loc="upper left", borderaxespad=0, ncol=4, fancybox=False, frameon=False, fontsize=fs)
    plt.legend(bbox_to_anchor=(-0.15, 1.12), loc='upper left', borderaxespad=0, fontsize=fs, fancybox=False, frameon=False, ncol=4, labelspacing=0.1)
    plt.grid()
    plt.xscale("log")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")
    plt.close()


def vhlines(ax, linestyle='-', color='gray', zorder=1, alpha=0.8, lw=0.75):
    ax.axhline(y=np.pi / 2, linestyle=linestyle, color=color, zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi / 2, linestyle=linestyle, color=color, zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi, linestyle=linestyle, color=color, zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi * 3 / 2, linestyle=linestyle, color=color, zorder=zorder, alpha=alpha, lw=lw)
    ax.text(np.pi / 2, np.pi + 0.05, "Left", ha='center')
    ax.text(np.pi * 3 / 2, np.pi + 0.05, "Right", ha='center')
    ax.text(np.pi, np.pi + 0.05, "Back", ha='center')


def plotazimzeni(pos, c, cblabel, cmap="viridis", figsize=(10.5, 4.5), emphasize_mes_pos=True, idx_mes_pos=None, vmin=None, vmax=None, save_path=None):
    '''
    args:
        pos: (B,*>3) tensor. (:,1):azimuth, (:,2):zenith
        c: (B) tensor.
        fname: str. filename
        title: str. title.
        cblabel: str. label of colorbar.
        cmap: colormap.
        figsie: (*,*) tuple.
        dpi: scalar.
    '''
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=figsize)
    vhlines(ax)
    if vmin is None:
        vmin = th.min(c)
    if vmax is None:
        vmax = th.max(c)
    mappable = ax.scatter(pos[:, 1], pos[:, 2], c=c, cmap=cmap, s=60, lw=0.3, ec="gray", zorder=2, vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=mappable, label=cblabel)
    if emphasize_mes_pos:
        ax.scatter(pos[idx_mes_pos, 1], pos[idx_mes_pos, 2], s=120, lw=0.5, c="None", marker="o", ec="k", zorder=1)
    ds = 0.1
    xlim = [0 - ds, 2 * np.pi]
    ylim = [0 - ds, ds + np.pi]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Zenith (deg)')
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12 + 1))
    ax.set_xticklabels([f'{int(azim)}' for azim in np.linspace(0, 2 * 180, 12 + 1)])
    ax.set_yticks(np.linspace(0, np.pi, 6 + 1))
    ax.set_yticklabels([f'{int(azim)}' for azim in np.linspace(0, 180, 6 + 1)])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")
    plt.close()


def plot_fig6(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, dirs):
    for database_name in database_info:
        save_path = f"{dirs[2]}/fig6/lsd_{database_name}.pdf"
        plot_lsd_uniform(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path)
        save_path = f"{dirs[2]}/fig6/ae_ild_{database_name}.pdf"
        plot_ae_ild_uniform(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path)


def plot_fig7(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, dirs):
    lr = 0
    database_name = "hutubs"
    suffix = "uniform-9"
    _, idx_mes = sample_uniform(database_info[database_name]["srcpos_cart"], 9, radius=database_info[database_name]["srcpos_sph"][0, 0], dataset_name=database_name)
    hrtf_mag_gt = hrtf_mag_dict_gt[database_name]
    for method in ["Proposed", "SWFE", "RSHE", "SPCA"]:
        hrtf_mag_pred = hrtf_mag_dict[database_name][suffix][method]
        lsd = th.sqrt(th.mean((hrtf_mag_gt - hrtf_mag_pred)**2, dim=-1))  # (S, B, 2)
        lsd = th.mean(lsd[:, :, lr], dim=0)  # (B)

        save_path = f"{dirs[2]}/fig7/{method.lower()}.pdf"
        plotazimzeni(pos=database_info[database_name]["srcpos_sph"], c=lsd, cblabel="LSD (dB)", idx_mes_pos=idx_mes, vmin=2, vmax=6, save_path=save_path)


def plot_fig8(database_info, hrtf_mag_dict_gt, dirs):
    lr = 0
    s = 0
    database_name = "hutubs"
    basename = "_hutubs_sub-89_uniform-9.pt"

    pos_cart_tar = database_info[database_name]["srcpos_cart"]
    zeni = database_info[database_name]["srcpos_sph"][:, 2]

    b_list_front = th.arange(pos_cart_tar.shape[0])[th.abs(pos_cart_tar[:, 1]) < 0.02 * (pos_cart_tar[:, 0] > -1e-3)].tolist()
    b_list_back = th.flip(th.arange(pos_cart_tar.shape[0])[th.abs(pos_cart_tar[:, 1]) < 0.02 * (pos_cart_tar[:, 0] < 0)], dims=(0,)).tolist()
    b_list_back = b_list_back + [b_list_back[-1]]
    b_list = b_list_back + b_list_front
    ylist = np.arange(2, len(b_list), 6).tolist()

    for method in ["ground_truth", "proposed", "swfe", "rshe", "spca"]:
        if method == "ground_truth":
            hrtf_mag_pred = hrtf_mag_dict_gt[database_name][s, :, :, :]
        elif method == "proposed":
            hrtf_mag_pred = th.load(f"{dirs[0]}/hrtf_mag/hrtf_mag{basename}")
        elif method == "swfe":
            hrtf_pred = th.load(f"{dirs[1]}/swfe/hrtf/hrtf{basename}")
            hrtf_mag_pred = 20 * th.log10(th.abs(hrtf_pred))
        else:
            hrtf_mag_pred = th.load(f"{dirs[1]}/{method}/hrtf_mag/hrtf_mag{basename}")

        hrtf_mag_plot = hrtf_mag_pred[b_list, lr, :]

        plt.rcParams["font.size"] = 20
        figsize = (10.5, 6)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(hrtf_mag_plot, vmax=5, vmin=-25, aspect="auto", cmap="viridis")
        fig.colorbar(im, label="Magnitude (dB)")

        ax.set_xlim([-1, 127])
        ax.set_xticks(np.arange(0, 128 + 1, 16) - 1, (np.arange(0, 128 + 1, 16) * 125 / 1000).astype(int))
        ax.set_xlabel("Frequency (kHz)", fontsize=24)
        ax.set_yticks(ylist, [f"{zeni[b].item()  /np.pi * 180:.0f}" for b in th.tensor(b_list)[ylist]])
        ax.set_ylabel("Zenith (deg)\n", fontsize=24)

        # plt.text(-20, 10, "Back", rotation="vertical")
        # plt.text(-20, 28, "Front", rotation="vertical")
        fig.text(
            0.03, 0.88,  # 位置 (x, y)
            # "Front                                    Back",
            r"Front $\leftarrow$                           $\rightarrow$ Back",
            fontsize=20,
            ha="left",
            va="top",
            rotation=90,
        )

        save_path = f"{dirs[2]}/fig8/{method}.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, transparent=True, bbox_inches="tight")
        print(f"Saved to {save_path}.")
        plt.close(fig)


def plot_lsd_plane(hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path):
    suffix_list = ["plane-z", "plane-yz", "plane-xyz", "plane-parallel-z"]
    xtick_list = ["plane-{z}", "plane-{y,z}", "plane-{x,y,z}", "plane-parallel-z"]
    hrtf_mag_gt = hrtf_mag_dict_gt[database_name]

    w = 1
    width = w * 0.6
    num_method = 4
    dh = width / num_method
    x_list = th.arange(0, len(suffix_list)).to(float) * w
    patterns = ["", ".", "/", "\\"]
    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(9, 4.5))
    plt.xticks(x_list, xtick_list)

    values = {}
    for m, method in enumerate(["Proposed", "SWFE", "RSHE", "SPCA"]):
        values[method] = {"mean": [], "std": []}
        for suffix in suffix_list:
            hrtf_mag_pred = hrtf_mag_dict[database_name][suffix][method]
            lsd = th.sqrt(th.mean((hrtf_mag_gt - hrtf_mag_pred)**2, dim=3))
            values[method]["mean"].append(th.mean(lsd))
            values[method]["std"].append(th.std(lsd))
        plt.bar(x_list + dh * (m - 3 / 2), values[method]["mean"], yerr=values[method]["std"], width=dh, label=method, zorder=2, hatch=patterns[m], ec="0.3", lw=1, capsize=6 / 0.2 * dh, ecolor="0")
    plt.grid(axis='y')
    # plt.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False)
    plt.legend(bbox_to_anchor=(0., 1.1), loc='upper left', borderaxespad=0, fancybox=False, frameon=False, ncol=4)
    plt.ylabel("LSD (dB)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved to {save_path}.")
    plt.close()


def plot_fig9(database_info, hrtf_mag_dict, hrtf_mag_dict_gt, dirs):
    for database_name in database_info:
        save_path = f"{dirs[2]}/fig9/{database_name}.pdf"
        plot_lsd_plane(hrtf_mag_dict, hrtf_mag_dict_gt, database_name, save_path)


def plot_fig10(database_info, itd_dict, itd_dict_gt, dirs):
    for database_name in database_info:
        save_path = f"{dirs[2]}/fig10/{database_name}.pdf"
        plot_ae_itd_uniform(database_info, itd_dict, itd_dict_gt, database_name, save_path)


def plot_fig11(itd_dict_gt, config, dirs):
    plt.style.use("seaborn-v0_8-colorblind")
    s = 0
    database_name = "hutubs"
    basename = "_hutubs_sub-89_uniform-9.pt"

    xlim = [-800e-6, 800e-6]
    ylim = [-800e-6, 800e-6]
    x = th.linspace(xlim[0], xlim[-1], 100)
    jnd = 32.5 * 1e-6 + 0.095 * th.abs(x)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for m, method in enumerate(["proposed", "swfe", "rshe", "woodworth"]):

        plt.figure(figsize=(5, 5))
        fs = 18
        plt.rcParams['font.size'] = fs
        plt.grid(zorder=0)
        plt.plot(x, x + jnd, color='k', linestyle='dashed')
        plt.plot(x, x - jnd, color='k', linestyle='dashed')
        plt.plot(x, x, color='k', linestyle='dotted')

        if method == "proposed":
            itd_pred = th.load(f"{dirs[0]}/itd/itd{basename}")
        elif method == "swfe":
            hrtf_pred = th.load(f"{dirs[1]}/swfe/hrtf/hrtf{basename}")
            itd_pred = hrir2itd(hrtf2hrir(hrtf_pred.unsqueeze(0)), fs=config.data.max_freq * 2, fs_up=config.data.fs_up)[0]
        else:
            itd_pred = th.load(f"{dirs[1]}/{method}/itd/itd{basename}")

        val = itd_pred
        val_gt = itd_dict_gt[database_name][s, :]
        plt.scatter(-val_gt, -val, c=colors[m], marker='.', s=100, label=method, zorder=2)

        plt.xlim(xlim)
        plt.ylim(ylim)

        tau_list = th.linspace(xlim[0], xlim[-1], 4 + 1)
        plt.xticks(tau_list, [f'{tau*1e6:.0f}' for tau in tau_list])
        # plt.xticklabels()

        tau_list = th.linspace(ylim[0], ylim[-1], 4 + 1)
        plt.yticks(tau_list, [f'{tau*1e6:.0f}' for tau in tau_list])

        plt.xlabel(r'True ITD  ($\mu$s)')
        plt.ylabel(r'Estimated ITD  ($\mu$s)')

        save_path = f"{dirs[2]}/fig11/{method.lower()}.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Saved to {save_path}.")
        plt.close()


def main(args):
    with open(f"{args.exp_dir_proposed}/config.yaml") as f:
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

    dirs = [args.exp_dir_proposed, args.exp_dir_baseline, args.out_dir]
    suffix_list = get_suffix_list()
    hrtf_mag_dict_pred = get_hrtf_mag_dict(suffix_list, config, dirs)
    itd_dict_pred = get_itd_dict(suffix_list, config, dirs)

    plot_fig6(database_info, hrtf_mag_dict_pred, hrtf_mag_dict_gt, dirs)
    plot_fig7(database_info, hrtf_mag_dict_pred, hrtf_mag_dict_gt, dirs)
    plot_fig8(database_info, hrtf_mag_dict_gt, dirs)
    plot_fig9(database_info, hrtf_mag_dict_pred, hrtf_mag_dict_gt, dirs)
    plot_fig10(database_info, itd_dict_pred, itd_dict_gt, dirs)
    plot_fig11(itd_dict_gt, config, dirs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir_proposed", default="exp/v1")
    parser.add_argument("--exp_dir_baseline", default="exp_baseline")
    parser.add_argument("--out_dir", default="figure")

    args = parser.parse_args()
    main(args)
