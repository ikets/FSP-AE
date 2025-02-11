import argparse
import torch as th
from torch.utils.data import DataLoader

from dataset import HRTFDataset
from model import FreqSrcPosCondAutoEncoder
from sampling import sample_all, sample_random
from utils import load_yaml


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


def valid(model, valid_loader, device):
    pass


def train(args):
    config = load_yaml(args.config_path)
    device = args.device

    # prepare dataset
    train_dataset = HRTFDataset(config.data, config.training.num_mes_pos_train, (config.data.hutubs.sub_id.train, config.data.riec.sub_id.train))
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    valid_dataset = HRTFDataset(config.data, config.training.num_mes_pos_valid, (config.data.hutubs.sub_id.valid, config.data.riec.sub_id.valid))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    # init model
    model = FreqSrcPosCondAutoEncoder(config.architecture)
    print(model)
    # load state_dict

    # calculate stats
    calculate_and_set_stats(model, train_dataset)

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.training.lr_milestones, gamma=config.training.lr_gamma)

    for epoch in range(config.training.epochs):
        for data, num_mes_pos_tup in train_loader:
            num_mes_pos = num_mes_pos_tup[0]
            hrtf_mag, itd, freq, pos_cart_tar, dataset_name = data["hrtf_mag"], data["itd"], data["frequency"], data["srcpos_cart"], data["dataset_name"][0]

            # sample measuremet positions
            if num_mes_pos == "all":
                pos_cart_mes, idx_mes = sample_all(pos_cart_tar[0])
            else:
                pos_cart_mes, idx_mes = sample_random(pos_cart_tar[0], num_mes_pos.item())
            pos_cart_mes, hrtf_mag_mes, itd_mes = pos_cart_mes.unsqueeze(0), hrtf_mag[:, idx_mes, :, :], itd[:, idx_mes]

            hrtf_mag_pred, itd_pred = model(hrtf_mag_mes, itd_mes, freq, pos_cart_mes, pos_cart_tar, dataset_name, device)
            # calc loss
            # optimization

        scheduler.step()
        valid(model, valid_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", ".config/v1.yaml")
    parser.add_argument("-d", "--cevice", "cuda")

    args = parser.parse_args()
    train(args)
