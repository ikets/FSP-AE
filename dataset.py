import torch as th
import torch.nn.functional as F
import torchaudio as ta
import numpy as np
import sofa

from utils import sph2cart, hrir2itd


class HRTFDataset(th.utils.data.Dataset):
    def __init__(self, config, num_mes_pos_list, sub_ids):
        super().__init__()
        self.config = config
        self.mag2db = ta.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.sofa_paths = []
        self.data = {}  # chache
        self.num_mes_pos_list = num_mes_pos_list
        # HUTUBS
        for sub_id in range(sub_ids[0][0], sub_ids[0][1]):
            sofa_path = f"{config.hutubs.path}/HRIRs/pp{sub_id+1}_HRIRs_measured.sofa"
            self.sofa_paths.append(sofa_path)
        # RIEC
        for sub_id in range(sub_ids[1][0], sub_ids[1][1]):
            sofa_path = f"{config.riec.path}/RIEC_hrir_subject_{sub_id+1:03}.sofa"
            self.sofa_paths.append(sofa_path)

    def __getitem__(self, index):
        sofa_path = self.sofa_paths[index // len(self.num_mes_pos_list)]
        item = self.get_data(sofa_path)
        num_mes_pos = self.num_mes_pos_list[index % len(self.num_mes_pos_list)]

        return (item, num_mes_pos)

    def __len__(self):
        return len(self.sofa_paths) * len(self.num_mes_pos_list)

    def get_data(self, sofa_path):
        if sofa_path in self.data:
            item = self.data[sofa_path]
        else:
            if "hutubs" in sofa_path:
                dataset_name = "hutubs"
                front_id = self.config.hutubs.front_id
            elif "riec" in sofa_path:
                dataset_name = "riec"
                front_id = self.config.riec.front_id
            else:
                raise NotImplementedError
            item = self.sofa2data(sofa_path, False, dataset_name, front_id)
            self.data[sofa_path] = item
        return item

    def sofa2data(self, sofa_path, multiple_green_func=False, dataset_name="hutubs", front_id=202):
        sofa_data = sofa.Database.open(sofa_path)

        srcpos_ori = th.tensor(sofa_data.Source.Position.get_values())  # azimuth in [0,360),elevation in [-90,90], radius in {r}
        srcpos_sph = srcpos_ori.to(th.float32)
        srcpos_sph[:, 0] = srcpos_sph[:, 0] % 360  # azimuth in [0,360)
        srcpos_sph[:, 1] = 90 - srcpos_sph[:, 1]  # elevation in [-90,90] -> zenith in [180,0]
        srcpos_sph[:, :2] = srcpos_sph[:, :2] / 180 * np.pi  # azimuth in [0,2*pi), zenith in [0,pi]
        if dataset_name == "riec":
            assert th.all(th.abs(srcpos_sph[:, 2] - 1.5) < 1e-3)
            # srcpos_sph[:, 2] = 1.5
        srcpos_sph = th.cat((srcpos_sph[:, 2].unsqueeze(1), srcpos_sph[:, :2]), dim=1)  # radius, azimuth in [0,2*pi), zenith in [0,pi]
        srcpos_cart = sph2cart(srcpos_sph[:, 1], srcpos_sph[:, 2], srcpos_sph[:, 0])

        sr_ori = sofa_data.Data.SamplingRate.get_values()[0]  # 44100
        hrir_ori = th.tensor(sofa_data.Data.IR.get_values())  # 440 x 2 x 256

        # downsampling
        downsampler = ta.transforms.Resample(sr_ori, 2 * self.config.max_freq, dtype=th.float32)
        hrir_tar = downsampler(hrir_ori.to(th.float32))

        # time alignment
        max_idx_front = round(th.mean(th.argmax(hrir_tar[front_id, :, :], dim=-1).to(th.float32)).item())
        if max_idx_front > 2 * self.config.max_freq * 1e-3:
            hrir_tar = hrir_tar[:, :, round(max_idx_front - 2 * self.config.max_freq * 1e-3):]
        else:
            hrir_tar = F.pad(input=hrir_tar, pad=(round(2 * self.config.max_freq * 1e-3 - max_idx_front), 0))
        if 2 * self.config.num_freq_bin > hrir_tar.shape[-1]:
            hrir_tar = F.pad(input=hrir_tar, pad=(0, 2 * self.config.num_freq_bin - hrir_tar.shape[-1]))
        else:
            hrir_tar = hrir_tar[:, :, :2 * self.config.num_freq_bin]

        # fft & conj
        hrtf_p_m = th.conj(th.fft.fft(hrir_tar, dim=-1))
        # extract positive frequency
        hrtf_p = hrtf_p_m[:, :, 1:self.config.num_freq_bin + 1]

        freq = th.arange(1, self.config.num_freq_bin + 1) * self.config.max_freq / self.config.num_freq_bin

        if multiple_green_func:
            r = srcpos_sph[0, 0]
            k = freq * 2 * np.pi / 343.18
            green = th.exp(1j * k * r) / (4 * np.pi * r)
            hrtf_p = hrtf_p * green[None, None, :]

        hrtf_mag_p = self.mag2db(th.abs(hrtf_p))

        itd = hrir2itd(hrir_tar.unsqueeze(0), fs=self.config.max_freq * 2, fs_up=self.config.fs_up).squeeze(0)  # (B, )

        returns = {
            "srcpos_sph": srcpos_sph,      # (B, 3)
            "srcpos_cart": srcpos_cart,    # (B, 3)
            "frequency": freq,             # (L)
            "hrtf": hrtf_p,                # (B, 2, L)
            "hrtf_mag": hrtf_mag_p,        # (B, 2, L)
            "hrir": hrir_tar,              # (B, 2, 2L)
            "itd": itd,                    # (B)
            "dataset_name": dataset_name,  # str
            "sofa_path": sofa_path         # str
        }
        return returns
