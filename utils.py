import logging
import os
import torch as th
import torchaudio as ta
import yaml


class AttrDict(dict):
    def __getattr__(self, name):
        value = None
        if name in self.keys():
            value = self[name]
        if isinstance(value, dict):
            value = AttrDict(value)
        return value


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return AttrDict(config)


def get_logger(dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    h = logging.FileHandler(f"{dir}/{filename}")
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def sph2cart(phi, theta, r):
    '''Conversion from spherical to Cartesian coordinates

    Args:
        phi, theta, r: Azimuth angle, zenith angle, distance
    Returns:
        x, y, z : Position in Cartesian coordinates
    '''
    x = r * th.sin(theta) * th.cos(phi)
    y = r * th.sin(theta) * th.sin(phi)
    z = r * th.cos(theta)
    return th.hstack((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)))


def hrtf2hrir(hrtf):
    '''
    Args:
        hrtf: (S, B, 2, L) complex tensor
    Returns:
        hrir: (S, B, 2, 2L)
    '''
    zeros = th.zeros([hrtf.shape[0], hrtf.shape[1], hrtf.shape[2], 1]).to(hrtf.device).to(hrtf.dtype)
    tf_fc = th.conj(th.flip(hrtf[:, :, :, :-1], dims=(-1,)))

    hrtf = th.cat((zeros, hrtf, tf_fc), dim=-1)
    hrir = th.fft.ifft(th.conj(hrtf), dim=-1)
    hrir = th.real(hrir)

    return hrir


def hrir2itd(hrir, thrsh_ms=1000, lowpass=True, upsample_via_cpu=True, conv_cpu=True, fs=32000.0, fs_up=384000.0):
    '''
    Args:
        hrir: (S, B, 2, L') tensor
        thrsh_ms: threshold [ms]. (computed ITD is forced to be in [-thrsh_ms, +thrsh_ms] )
        lowpass: bool. If True, Low-pass filter is filtered to hrir.
    Returns:
        ITD: (S, B) tensor, interaural time difference [s] (-:src@left, +:src@right)
    '''

    if lowpass:
        hrir = ta.functional.lowpass_biquad(waveform=hrir, sample_rate=fs, cutoff_freq=1600)
    else:
        pass
    if upsample_via_cpu:
        hrir = hrir.cpu()
    upsampler = ta.transforms.Resample(fs, fs_up)
    hrir_us = upsampler(hrir.to(th.float32).contiguous())
    if upsample_via_cpu and not conv_cpu:
        hrir_us = hrir_us.cuda()
    S, B, _, L = hrir_us.shape
    thrsh_idx = round(fs_up / thrsh_ms)

    hrir_l = hrir_us[:, :, 0, :]
    hrir_r = hrir_us[:, :, 1, :]
    hrir_l_pad = th.nn.functional.pad(hrir_l, (L, L))  # torch.Size([S, 440, 384])
    hrir_l_pad_in = hrir_l_pad.reshape(1, S * B, -1)
    hrir_r_wt = hrir_r.reshape(S * B, 1, -1)
    crs_cor = th.nn.functional.conv1d(hrir_l_pad_in, hrir_r_wt, groups=S * B)
    crs_cor = crs_cor.reshape(S, B, -1)
    idx_beg = L - thrsh_idx
    idx_end = L + thrsh_idx + 1
    idx_max = th.argmax(crs_cor[:, :, idx_beg:idx_end], dim=-1) - thrsh_idx
    itd = idx_max / fs_up

    return itd


def get_hrir_with_itd(input=None, itd=None, input_kind="hrtf_mag", fs=32000.0, fs_up=384000.0):
    '''
    Args:
        input: one of
            - hrir:     (S, B, 2, 2L)
            - hrtf_mag: (S, B, 2, L)
            - hrtf:     (S, B, 2, L)
        itd: (S, B)
    Returns:
        hrir with itd:  (S, B, 2, 2L)
    '''

    if input_kind == "hrir":
        hrir_ori = input
    elif input_kind == "hrtf_mag":
        hrtf_mag_lin = 10**(input / 20)
        _, hrir_ori = minphase_recon(hrtf_mag_lin.to(th.complex64), negative_freq=False)
    elif input_kind == "hrtf":
        _, hrir_ori = minphase_recon(input, negative_freq=False)
    itd_ori = hrir2itd(hrir=hrir_ori, fs=fs, fs_up=fs_up).to(input.device)
    hrir_out = assign_itd(hrir_ori=hrir_ori, itd_ori=itd_ori, itd_des=itd, fs=fs)

    return hrir_out


def minphase_recon(trans_func, negative_freq=False):
    '''
    Args:
        trans_func: (S, B, 2, L) or (S, B, 2, 2L) tensor (L: # of positive freq. bins)
        negative_freq: bool. If True, trans_func.shape == (S, B, 2, 2L)

    Returns:
        phase_min: (S, B, 2, 2L) tensor
        imp_resp_min:  (S, B, 2, 2L) tensor. Impulse response with minimum phase.
    '''
    if negative_freq:
        trans_func_pm = trans_func
    else:
        trans_func_nf = th.conj(th.flip(trans_func[:, :, :, :-1], dims=(-1,)))  # negatibe freq.
        trans_func_pm = th.cat((th.ones_like(trans_func)[:, :, :, 0:1], trans_func, trans_func_nf), dim=-1)  # [1,pos,neg]
    log_mag_pm = th.log(th.abs(trans_func_pm))
    phase_min = - hilbert_transform(log_mag_pm)
    imp_resp_min = th.real(th.fft.ifft(th.abs(trans_func_pm) * th.exp(1j * phase_min), axis=-1))

    return phase_min, imp_resp_min


def hilbert_transform(input, detach=False):
    # https://stackoverflow.com/questions/50902981/hilbert-transform-using-cuda
    assert input.dim() == 4
    n = input.shape[-1]
    # Allocates memory on GPU with size/dimensions of signal
    if detach:
        transforms = input.clone().detach()
    else:
        transforms = input.clone()
    transforms = th.fft.fft(transforms, axis=-1)
    transforms[:, :, :, 1:n // 2] *= -1j             # positive frequency
    transforms[:, :, :, (n + 2) // 2 + 1: n] *= +1j  # negative frequency
    transforms[:, :, :, 0] = 0  # DC signal
    if n % 2 == 0:
        transforms[:, :, :, n // 2] = 0  # the (-1)**n term

    return th.fft.ifft(transforms, axis=-1)


def assign_itd(hrir_ori, itd_ori, itd_des, fs, shift_s=1e-3):
    '''
    Args:
        hrir_ori: (S, B, 2, L) tensor. (L: filter length)
        itd_ori: (S, B) tensor. ITD [s] of hrir_ori
        itd_des: (S, B) tensor. desired ITD [s]
        fs: scaler [Hz]. Sampling Frequency.
        shift_lr: scaler [s]. offset when ITD==0.
    Returns:
        ir_itd_des:  (S, B, 2, L) tensor. Impulse response with desired ITD.
    '''
    S, B = itd_ori.shape
    L = hrir_ori.shape[-1]
    shift_idx = shift_s * fs
    ITD_idx_fs_half = (itd_des - itd_ori) * fs / 2
    offset = th.ones(S, B, 2).to(ITD_idx_fs_half.device) * shift_idx
    offset[:, :, 0] += ITD_idx_fs_half  # left
    offset[:, :, 1] -= ITD_idx_fs_half  # right
    offset = th.round(offset).to(int)

    arange = th.arange(L).reshape(1, 1, 1, L).tile(S, B, 2, 1).to(ITD_idx_fs_half.device)
    arange = (arange - offset[:, :, :, None]) % L

    # square window to remove pre-echo
    window_length = int(L - shift_idx)
    window_sq = th.cat((th.ones(window_length), th.zeros(L - window_length))).to(hrir_ori.device)
    hrir_ori_w = hrir_ori * window_sq[None, None, None, :]
    ir_itd_des = th.gather(hrir_ori_w, -1, arange)

    return ir_itd_des
