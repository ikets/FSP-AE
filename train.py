import argparse
import numpy as np
import os
import random
import torch as th
from torch.utils.data import DataLoader

from dataset import HRTFDataset
from loss import LSD
from model import FreqSrcPosCondAutoEncoder
from sampling import sample_all, sample_random
from utils import load_yaml


def load_state(path, model, optimizer=None, scheduler=None):
    state_dicts = th.load(path)
    model.load_state_dict(state_dicts["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state_dicts["scheduler"])
    epoch = state_dicts["epoch"]
    best_valid_loss = state_dicts["best_valid_loss"]
    print(f"Loaded checkpoint {path} from epoch {epoch}.")
    return model, optimizer, scheduler, epoch, best_valid_loss


def save_state(path, model, optimizer=None, scheduler=None, epoch=0, best_valid_loss=1e10):
    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_valid_loss": best_valid_loss
    }
    th.save(states, path)
    print(f"Saved in {path}.")


def calculate_and_set_stats(model, train_dataset):
    database_name = [db for db in ["hutubs", "riec"] if db not in model.stats]
    items = []

    for sofa_path in train_dataset.sofa_paths:
        item = train_dataset.get_data(sofa_path)
        dataset_name = item["dataset_name"]
        for data_type in ["hrtf_mag", "itd"]:
            items[dataset_name][data_type].append(item[data_type].unsqueeze(0))

    for dataset_name in database_name:
        for data_type in ["hrtf_mag", "itd"]:
            items_cat = th.cat(items[dataset_name][data_type], dim=0)
            mean, std = th.mean(items_cat), th.std(items_cat)
            model.set_stats(mean, std, dataset_name, data_type)


def valid(config, model, valid_loader, device):
    with th.no_grad:
        loss_lsd = LSD()
        loss_ae = th.nn.L1Loss()
        loss_dict = {"lsd": 0.0, "ae_itd": 0.0, "all": 0.0}
        for data, num_mes_pos_tup in valid_loader:
            num_mes_pos = num_mes_pos_tup[0]
            hrtf_mag, itd, freq, pos_cart_tar, dataset_name = data["hrtf_mag"], data["itd"], data["frequency"], data["srcpos_cart"], data["dataset_name"][0]

            # sample measuremet positions
            if num_mes_pos == "all":
                pos_cart_mes, idx_mes = sample_all(pos_cart_tar[0])
            else:
                pos_cart_mes, idx_mes = sample_random(pos_cart_tar[0], num_mes_pos.item())
            pos_cart_mes, hrtf_mag_mes, itd_mes = pos_cart_mes.unsqueeze(0), hrtf_mag[:, idx_mes, :, :], itd[:, idx_mes]

            hrtf_mag_pred, itd_pred = model(hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device)
            loss_dict["lsd"] += loss_lsd(hrtf_mag_pred, hrtf_mag, dim=3)
            loss_dict["ae_itd"] += loss_ae(itd_pred, itd)

        for k, v in loss_dict.items():
            v /= len(valid_loader)

        for k, v in loss_dict.items():
            loss_dict["all"] += config.training.loss_weight[k] * v

    return loss_dict["all"]


def train(args):
    config = load_yaml(args.config_path)
    device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    exp_dir = f"exp/{os.path.basename(args.config_path).split('.')[0]}"
    if os.path.isdir(exp_dir) and not args.forced:
        raise RuntimeError(f"{exp_dir} already exists! Use -f option to overwrite.")

    # prepare dataset
    train_dataset = HRTFDataset(config.data, config.training.num_mes_pos_train, (config.data.hutubs.sub_id.train, config.data.riec.sub_id.train))
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    valid_dataset = HRTFDataset(config.data, config.training.num_mes_pos_valid, (config.data.hutubs.sub_id.valid, config.data.riec.sub_id.valid))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    model = FreqSrcPosCondAutoEncoder(config.architecture)
    optimizer = th.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.training.lr_milestones, gamma=config.training.lr_gamma)
    if config.training.checkpoint_path is not None:
        model, optimizer, scheduler, epoch_offset, best_valid_loss = load_state(config.training.checkpoint_path, model, optimizer, scheduler)
    else:
        epoch_offset = 0
        best_valid_loss = 1e10
    calculate_and_set_stats(model, train_dataset)

    loss_lsd = LSD()
    loss_ae = th.nn.L1Loss()

    print("=======")
    print(model)
    print("=======")

    for epoch in range(epoch_offset, config.training.epochs):
        model.train()
        for data, num_mes_pos_tup in train_loader:
            num_mes_pos = num_mes_pos_tup[0]
            hrtf_mag, itd, freq, pos_cart_tar, dataset_name = data["hrtf_mag"], data["itd"], data["frequency"], data["srcpos_cart"], data["dataset_name"][0]

            # Sample measuremet positions
            if num_mes_pos == "all":
                pos_cart_mes, idx_mes = sample_all(pos_cart_tar[0])
            else:
                pos_cart_mes, idx_mes = sample_random(pos_cart_tar[0], num_mes_pos.item())
            pos_cart_mes, hrtf_mag_mes, itd_mes = pos_cart_mes.unsqueeze(0), hrtf_mag[:, idx_mes, :, :], itd[:, idx_mes]

            optimizer.zero_grad()
            hrtf_mag_pred, itd_pred = model(hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device)
            loss_dict = {
                "lsd": loss_lsd(hrtf_mag_pred, hrtf_mag, dim=3),
                "ae_itd": loss_ae(itd_pred, itd)
            }
            loss = 0
            for k, v in loss_dict.items():
                loss += config.training.loss_weight[k] * v
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        valid_loss = valid(config, model, valid_loader, device)
        if valid_loss < best_valid_loss:
            save_path_best = f"{exp_dir}/checkpoint_best.pt"
            save_state(save_path_best, model, optimizer, scheduler, epoch, best_valid_loss)
            print(f"Best valid loss: {valid_loss}! Saved in {save_path_best}.")
            best_valid_loss = valid_loss
        if epoch % config.training.save_interval == 0:
            save_path_log = f"{exp_dir}/checkpoint_log_{epoch}epoch.pt"
            save_state(save_path_log, model, optimizer, scheduler, epoch, best_valid_loss)
            print(f"Saved in {save_path_log}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", default="./config/v1.yaml")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-s", "--seed", default=0)
    parser.add_argument("-f", "--forced", default=False)

    args = parser.parse_args()
    train(args)
