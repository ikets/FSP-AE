import math
import scipy.io
import scipy.spatial
import torch as th


def sample_all(pos_cart):
    return pos_cart, list(range(0, pos_cart.shape[0]))


def sample_random(pos_cart, num_mes_pos):
    '''
    Args:
        pos_cart: (B, 3) tensor
        num_mes_pos: int

    Returns:
        pos_cart_sampled: subset of pos_cart (B_mes, 3).
        idx: list of sampled index, length: (B_mes)
    '''
    idx = th.randperm(pos_cart.shape[0])[:num_mes_pos]
    idx = sorted([i.item() for i in idx])
    return pos_cart[idx, :], idx


def sample_uniform(pos_cart, num_mes_pos, radius=1.47, dataset_name="hutubs"):
    '''
    Args:
        pos_cart: (B, 3) tensor
        num_mes_pos: int

    Returns:
        pos_cart_sampled: subset of pos_cart (B_mes, 3).
        idx: list of sampled index, length: (B_mes)
    '''
    valid_num_mes_pos_list = [4, 6] + [(t + 1) ** 2 for t in range(2, 19)]
    assert num_mes_pos in valid_num_mes_pos_list, f"`num_mes_pos` must be one of {valid_num_mes_pos_list}."
    if num_mes_pos in [4, 6]:
        grid_dic = scipy.io.loadmat(f'sampling/grids/grid_rp{num_mes_pos}.mat')
        grid = th.tensor(grid_dic["Y"]).T  # (num_mes_pos, 3)
        if dataset_name == "riec":
            row = grid[:, 2] >= -0.5
            grid = grid[row]
        grid = grid * float(radius)

        kdt = scipy.spatial.KDTree(pos_cart.cpu())
        dist, idx = kdt.query(grid)  # dist, index
        idx = list(set(sorted(idx)))  # remove duplication
    else:
        t = int(math.sqrt(num_mes_pos) - 1)
        grid_dic = scipy.io.loadmat(f'sampling/grids/grid_t{t}d{num_mes_pos}.mat')
        grid = th.tensor(grid_dic["Y"]).T  # (num_mes_pos, 3)
        if dataset_name == "riec":
            row = grid[:, 2] >= -0.5
            grid = grid[row]
        grid = grid * radius

        kdt = scipy.spatial.KDTree(pos_cart.cpu())
        dist, idx = kdt.query(grid)  # dist, index
        idx = list(set(sorted(idx)))  # remove duplication

    return pos_cart[idx, :], idx


def sample_plane(pos_cart, axes=(0, 1, 2), threshold=0.01):
    '''
    Args:
        pos_cart: (B, 3) tensor
        axes: tuple of int in {0, 1, 2}

    Returns:
        subset of pos_cart (B_mes, 3).
        list of sampled index, length: (B_mes)
    '''
    idx = th.zeros(0).to(pos_cart.device)
    idx_all = th.arange(0, pos_cart.shape[0]).to(pos_cart.device)
    for ax in axes:
        idx = th.cat((idx, idx_all[(th.abs(pos_cart[:, ax]) < threshold)]), dim=0)
    idx = idx.to(th.int).tolist()
    idx = list(set(sorted(idx)))  # remove duplication

    return pos_cart[idx, :], idx


def sample_plane_parallel(pos_cart, axis=2, values=(-0.7, 0.0, 0.7), threshold=0.01):
    '''
    Args:
        pos_cart: (B, 3) tensor
        axis: int in {0, 1, 2}
        values: tuple of float

    Returns:
        subset of pos_cart (B_mes, 3).
        list of sampled index, length: (B_mes)
    '''
    idx = th.zeros(0).to(pos_cart.device)
    idx_all = th.arange(0, pos_cart.shape[0]).to(pos_cart.device)
    for v in values:
        idx = th.cat((idx, idx_all[(th.abs(pos_cart[:, axis] - v) < threshold)]), dim=0)
    idx = idx.to(th.int).tolist()
    idx = list(set(sorted(idx)))  # remove duplication

    return pos_cart[idx, :], idx
