import torch as th
import numpy as np
import scipy.special as special
from scipy.spatial import KDTree


def reg_matrix(matrix_type="duraiswami", n_vec=None, order=None):
    if matrix_type == "duraiswami":
        assert n_vec is not None
        return th.diag(1 + th.tensor(n_vec) * (th.tensor(n_vec) + 1))
    elif matrix_type == "tikhonov":
        assert order is not None
        return th.eye(order)
    else:
        raise ValueError("matrix_type must be 'duraiswami' or 'tikhonov'.")


def sph_harm(m, n, phi, theta):
    """Spherical harmonic function
    m, n: degrees and orders
    phi (in [0, 2*pi]): azimuth angle
    theta (in [0, pi]): zenith angle
    """
    return special.sph_harm(m, n, phi, theta)


def sph_harm_nmvec(order, rep=None):
    """Vectors of spherical harmonic orders and degrees
    Returns (order+1)**2 size vectors of n and m
    n = [0, 1, 1, 1, ..., order, ..., order]^T
    m = [0, -1, 0, 1, ..., -order, ..., order]^T

    Parameters
    ------
    order: Maximum order
    rep: Same vectors are copied as [n, .., n] and [m, ..., m]

    Returns
    ------
    n, m: Vectors of orders and degrees
    """
    n = np.array([0])
    m = np.array([0])
    for nn in np.arange(1, order + 1):
        nn_vec = np.tile([nn], 2 * nn + 1)
        n = np.append(n, nn_vec)
        mm = np.arange(-nn, nn + 1)
        m = np.append(m, mm)
    if rep is not None:
        n = np.tile(n[:, None], (1, rep))
        m = np.tile(m[:, None], (1, rep))
    return n, m


def spherical_hn(n, k, z):
    """nth-order sphericah Henkel function of kth kind
    Returns h_n^(k)(z)
    """
    if k == 1:
        return special.spherical_jn(n, z) + 1j * special.spherical_yn(n, z)
    elif k == 2:
        return special.spherical_jn(n, z) - 1j * special.spherical_yn(n, z)
    else:
        raise ValueError()


def sph_wavefunc_expansion(hrtf, freq, mes_pos_sph, tar_pos_sph, reg_param=1e-6, coeff=2.0, sound_speed=343.18, r_shoulder=0.45):
    '''
    Args:
        hrtf:        (B_m, 2, L, S)
        freq:        (L)
        mes_pos_sph: (B_m, 3)
        tar_pos_sph: (B_t, 3)

    Returns:
        hrtf_pred: (B_t, 2, L, S)
    '''
    k_list = 2 * np.pi * freq / sound_speed  # wave number
    N_o_max = th.ceil(th.sqrt(mes_pos_sph.shape[0] * coeff)) - 1
    N_o_l_list = th.minimum(th.ceil(np.e * k_list * r_shoulder / 2).to(th.int), th.tensor([N_o_max]))

    hrtf_pred = th.zeros(size=[mes_pos_sph.shape[0]] + hrtf.shape[1:], dtype=hrtf.dtype, device=hrtf.device)

    for l, (k_l, N_o_l) in enumerate(zip(k_list, N_o_l_list)):
        n_vec, m_vec = sph_harm_nmvec(N_o_l)

        sph_harm_mes = sph_harm(m_vec[None, :], n_vec[None, :], mes_pos_sph[:, 1, None], mes_pos_sph[:, 2, None])  # B_m, (N+1)^2
        sph_hankel_mes = spherical_hn(n_vec[None, :], 1, mes_pos_sph[:, 0, None] * k_l)  # B_m, (N+1)^2
        sph_wave_mes = (sph_harm_mes * sph_hankel_mes).to(th.complex64)  # B_m, (N+1)^2

        sph_harm_tar = sph_harm(m_vec[None, :], n_vec[None, :], tar_pos_sph[:, 1, None], tar_pos_sph[:, 2, None])  # B,_t (N+1)^2
        sph_hankel_tar = spherical_hn(n_vec[None, :], 1, tar_pos_sph[:, 0, None] * k_l)  # B_t, (N+1)^2
        sph_wave_tar = (sph_harm_tar * sph_hankel_tar).to(th.complex64)  # B_t, (N+1)^2

        phi_adj_phi = th.matmul(th.conj(sph_wave_mes.permute(1, 0)), sph_wave_mes)  # (N+1)^2, (N+1)^2
        gamma = reg_matrix("duraiswami", n_vec=n_vec)
        phi_adj_h = th.matmul(th.conj(sph_wave_mes.permute(1, 0))[None, None, :, :], hrtf[:, :, l, :].permute(1, 2, 0).unsqueeze(-1))  # 2, S, (N+1)^2, 1
        c = th.linalg.solve((phi_adj_phi + reg_param * gamma)[None, None, :, :], phi_adj_h)  # 2, S, (N+1)^2, 1
        hrtf_pred[:, :, l, :] = th.matmul(sph_wave_tar[None, None, :, :], c).reshape(c.shape[0], c.shape[1], -1).permute(2, 0, 1)  # 2 S B 1 -> 2 S B -> B 2 S

    return hrtf_pred


def real_sph_harm_expansion(input, mes_pos_sph, tar_pos_sph, input_type="hrtf_mag", reg_param=1e-3, coeff=2.0):
    '''
    Args:
        input: one of
            - hrtf_mag: (B_m, 2, L, S)
            - itd:      (B_m, S)
        mes_pos_sph: (B_m, 3)
        tar_pos_sph: (B_t, 3)

    Returns:
        output: one of
            - hrtf_mag_pred: (B_t, 2, L, S)
            - itd_pred:      (B_t, S)
    '''
    n_vec, m_vec = sph_harm_nmvec(th.ceil(th.sqrt(mes_pos_sph.shape[0] * coeff)) - 1)
    gamma = reg_matrix("duraiswami", n_vec=n_vec)

    sph_harm_mes = sph_harm(m_vec[None, :], n_vec[None, :], mes_pos_sph[:, 1, None], mes_pos_sph[:, 2, None])  # B_m, (N+1)^2
    sph_harm_mes = th.real(sph_harm_mes) * (m_vec >= 0)[None, :] - th.imag(sph_harm_mes) * (m_vec < 0)[None, :]
    phi_trans_phi = th.matmul(sph_harm_mes.permute(1, 0), sph_harm_mes)  # (N+1)^2, (N+1)^2

    sph_harm_tar = sph_harm(m_vec[None, :], n_vec[None, :], tar_pos_sph[:, 1, None], tar_pos_sph[:, 2, None])  # B_t, (N+1)^2
    sph_harm_tar = th.real(sph_harm_tar) * (m_vec >= 0)[None, :] - th.imag(sph_harm_tar) * (m_vec < 0)[None, :]

    if input_type == "hrtf_mag":
        phi_trans_h = th.matmul(sph_harm_mes.permute(1, 0)[None, None, None], input.permute(1, 2, 3, 0).unsqueeze(-1))  # (...,(N+1)^2, B_m) * (2, L, S, B_m, 1) -> (2, L, S, (N+1)^2, 1)
        c = th.linalg.solve((phi_trans_phi + reg_param * gamma)[None, None, None], phi_trans_h)  # (2, L, S, (N+1)^2, 1)
        output = th.matmul(sph_harm_tar[None, None, None], c).reshape(c.shape[0], c.shape[1], c.shape[2], -1).permute(3, 0, 1, 2)
    elif input_type == "itd":
        phi_trans_h = th.matmul(sph_harm_mes.permute(1, 0)[None], input.permute(1, 0).unsqueeze(-1))  # (...,(N+1)^2, B_m) * (S, B_m, 1) -> (S, (N+1)^2, 1)
        c = th.linalg.solve((phi_trans_phi + reg_param * gamma)[None], phi_trans_h)  # (S, (N+1)^2, 1)
        output = th.matmul(sph_harm_tar[None], c).reshape(c.shape[0], -1).permute(1, 0)
    else:
        raise NotImplementedError

    return output


def spatial_principal_component_analysis(input_cen, tar_pos_cart, mes_idx, spc_mat, mean_vec, input_type="hrtf_mag", reg_param=1e-3, coeff=2.0):
    '''
    Args:
        input: one of
            - hrtf_mag: (B_m, 2, L, S)
            - itd:      (B_m, S)
        mes_pos_cart: (B_m, 3)
        tar_pos_cart: (B_t, 3)
        mes_idx:      (B_m)
        spc_mat:     (B_t, B_t)
        mean_vec:    (B_t)

    Returns:
        output: one of
            - hrtf_mag_pred: (B_t, 2, L, S)
            - itd_pred:      (B_t, S)
    '''
    N = th.pow(th.ceil(th.sqrt(mes_idx.shape[0] * coeff)), 2)  # num of SPCs
    gamma = reg_matrix("tikhonov", order=N)  # (N, N)
    spc_mat_trunc_tar = spc_mat[:, :N]  # (B_t, N)
    spc_mat_trunc_mes = spc_mat_trunc_tar[mes_idx, :]  # (B_m, N)
    phi_trans_phi = th.matmul(spc_mat_trunc_mes.permute(1, 0), spc_mat_trunc_mes)  # (B_m, B_m)

    if input_type == "hrtf_mag":
        # left
        input_l_cen = input_cen[:, 0, :, :] - mean_vec[mes_idx, None, None]  # (B_m, L, S)
        phi_trans_h = th.matmul(spc_mat_trunc_mes.permute(1, 0)[None, None], input_l_cen.permute(1, 2, 0).unsqueeze(-1))  # (..., N, B_m) * (L, S, B_m,1) -> (L, S, N, 1)
        c = th.linalg.solve((phi_trans_phi + reg_param * gamma)[None, None], phi_trans_h)  # (L, S, N, 1)
        output_l = th.matmul(spc_mat_trunc_tar[None, None], c).reshape(c.shape[0], c.shape[1], -1).permute(2, 0, 1) + mean_vec[:, None, None]  # (L, S, B_t, 1) -> (L, S, B_t) -> (B_t, L, S)

        # right
        tar_pos_cart_flip = tar_pos_cart * th.tensor([1.0, -1.0, 1.0])[None, :]
        kdt = KDTree(tar_pos_cart_flip.cpu())
        _, mes_idx_flip = kdt.query(tar_pos_cart[mes_idx])
        _, tar_idx_flip = kdt.query(tar_pos_cart)
        spc_mat_trunc_mes_flip = spc_mat_trunc_tar[mes_idx_flip, :]  # (B_m, N)
        spc_mat_trunc_tar_flip = spc_mat_trunc_tar[tar_idx_flip, :]  # (B_t, N)

        input_r_cen = input_cen[:, 1, :, :] - mean_vec[mes_idx_flip, None, None]  # (B_m, L, S)

        phi_trans_h = th.matmul(spc_mat_trunc_mes_flip.permute(1, 0)[None, None], input_r_cen.permute(1, 2, 0).unsqueeze(-1))  # (..., N, B_m) * (L, S, B_m,1) -> (L, S, N, 1)
        c = th.linalg.solve((phi_trans_phi + reg_param * gamma)[None, None], phi_trans_h)  # (L, S, N, 1)
        output_r = th.matmul(spc_mat_trunc_tar_flip[None, None], c).reshape(c.shape[0], c.shape[1], -1).permute(2, 0, 1) + mean_vec[tar_idx_flip, None, None]  # (L, S, B_t, 1) -> (L, S, B_t) -> (B_t, L, S)

        output = th.cat((output_l.unsqueeze(1), output_r.unsqueeze(1)), dim=1)  # (B_t, 2, L, S)
    elif input_type == "itd":
        input_cen = input - mean_vec[mes_idx, None]  # (B_m, S)
        phi_trans_h = th.matmul(spc_mat_trunc_mes.permute(1, 0)[None], input_cen.permute(1, 0).unsqueeze(-1))  # (..., N, B_m) * (S, B_m, 1) -> (S, N, 1)
        c = th.linalg.solve((phi_trans_phi + reg_param * gamma)[None], phi_trans_h)  # (S, N, 1)
        output = th.matmul(spc_mat_trunc_tar[None], c).reshape(c.shape[0], -1).permute(1, 0) + mean_vec[:, None]  # (S, B_t, 1) -> (S, B_t) -> (B_t, S)
    else:
        raise NotImplementedError

    return output


def woodworth(itd, mes_pos_cart, mes_pos_sph, tar_pos_cart, tar_pos_sph):
    y_on_r_mes = mes_pos_cart[:, 1, :] / mes_pos_sph[:, 0, :]
    alpha_mes = y_on_r_mes + th.asin(y_on_r_mes)

    y_on_r_tar = tar_pos_cart[:, 1, :] / tar_pos_sph[:, 0, :]
    alpha_tar = y_on_r_tar + th.asin(y_on_r_tar)

    rh_on_ss = th.matmul(itd.permute(1, 0).unsqueeze(1), alpha_mes.permute(1, 0).unsqueeze(2)).reshape(itd.shape[-1]) / th.norm(alpha_mes, p=2, dim=0)**2  # (S, 1, B) @ (S, B, 1) -> (S, 1, 1) -> (S,)
    itd_pred = alpha_tar * rh_on_ss[None, :]

    return itd_pred


def calculate_spc_mat_and_mean_vec(input, pos_cart, input_type="hrtf_mag"):
    '''
    Args:
        input: one of
            - hrtf_mag: (B_t, 2, L, S)
            - itd:      (B_t, S)
        pos_cart: (B_t, 3)

    Returns:
        spc_mat: (B_t, B_t)
        mean_vec: (B_t)
    '''
    if input_type == "hrtf_mag":
        pos_cart_flip = pos_cart * th.tensor([1.0, -1.0, 1.0])[None, :, None]
        kdt = KDTree(pos_cart_flip.cpu())
        _, idx_flip = kdt.query(pos_cart)

        input_lrcat = th.cat((input[:, 0, :, :], input[idx_flip, 1, :, :]), dim=-1)  # (B_t, L, 2S)
        mean_vec = th.mean(input_lrcat, dim=(1, 2))
        input_cen = input_lrcat - mean_vec[:, None, None]
        input_cen = th.reshape(input_cen, (input_cen.shape[0], -1)).permute(1, 0)  # (L*2S, B)
    elif input_type == "itd":
        mean_vec = th.mean(input, dim=-1)  # (B_t)
        input_cen = input - mean_vec[:, None]  # (B_t, S)
        input_cen = input_cen.permute(1, 0)  # (S, B_t)
    else:
        raise NotImplementedError

    var_mat = th.matmul(input_cen.permute(1, 0), input_cen)  # (B_t, B_t)
    _, spc_mat = th.linalg.eig(var_mat)  # (B_t, B_t)

    return spc_mat, mean_vec
