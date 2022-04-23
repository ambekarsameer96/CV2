import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity

def multires_sampling(points, RMS_li, **args):
    amount = len(points) // 100
    if len(RMS_li) > 1:
        ratio = (RMS_li[0] - RMS_li[-1]) / RMS_li[0]
        if ratio > 0.4:
            amount = len(points) // (len(points) // 16)
        if ratio > 0.6:
            amount = len(points) // (len(points) // 8)
        if ratio > 0.70:
            amount = len(points) // (len(points) // 4)

    return points[::amount]


def step_sampling(points):
    amount = 50
    return points[::amount]


def uniform_sampling(points, ratio, **args):
    idx = np.random.choice(len(points), len(points) // ratio)
    return points[idx]


def no_sampling(points, **args):
    return points


def density_sampling(points, **args):
    #sub-sampling from points by density
    #find nearest neighbors
    kde = KernelDensity(leaf_size=20, bandwidth=0.2).fit(points)
    density_est = kde.score_samples(points)
    idx = np.arange(len(points))[density_est > density_est.mean()]
    #give list of indices with good amount of neigbors, such that the density is high.
    return points[idx]


def min_dist(source, target):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    idx = []
    for sample in source:
        dist = np.linalg.norm(target - sample, axis=1)
        idx.append(np.argmin(dist))

    return target[idx], source


#2.2 kd-tree
#obtain points from Source and Target, try using KD tree, pair with matching.
def kd_method(source, target):
    tree = KDTree(target)
    nearest_ind = tree.query(source, k=1)[1].squeeze(1)
    result = target[nearest_ind]
    return result, source


def min_dist_buffer(source, target):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    # idx = []
    idx = {}
    # xy_source = np.delete(source, 2, 1)
    for sample in source:
        dist = np.linalg.norm((target - sample[:2]) ** 2, axis=1)
        present_sample = idx.get(np.argmin(dist), None)

        if present_sample is not None:

            if present_sample[2] < sample[2]:
                idx[np.argmin(dist)] = sample

        else:
            idx[np.argmin(dist)] = sample

    return idx


def z_buffer(source_tr, target, H=64, W=100):
    # Union of A1, A2
    sour_tar_union = np.vstack((source_tr, target))
    x_dim = sour_tar_union[:, 0]
    y_dim = sour_tar_union[:, 1]

    # Minimum enclosing box
    min_x, max_x = min(x_dim), max(x_dim)
    min_y, max_y = min(y_dim), max(y_dim)

    # Initialize the bounding box
    bound_box = [[0] * W for i in range(H)]

    # Dimensions of the box
    max_hor = abs(max_x - min_x)
    max_ver = abs(max_y - min_y)

    # Step for each dimenssion
    hor_const = max_hor / (W - 1)
    ver_const = max_ver / (H - 1)

    # Fill in the bounding box
    count = 0
    for i in range(H):
        for j in range(W):
            if count % W == 0:
                x_cor = min_x
                y_cor = max_y - i * ver_const
                bound_box [i][j] = (x_cor, y_cor)
            else:
                x_cor = min_x + j*hor_const
                bound_box [i][j] = (x_cor, y_cor)
            count += 1

    # Source and target xy-planes
    flattened_bound_box = np.asarray(bound_box).reshape(H * W, 2)

    # Initialize source, target buffers
    source_buffer = np.matrix(np.ones((H * W, 3)) * np.inf)
    target_buffer = np.matrix(np.ones((H * W, 3)) * np.inf)

    # List with pointers to the minimum distance elements
    sour_idx_list = min_dist_buffer(source_tr, flattened_bound_box)
    tar_idx_list = min_dist_buffer(target, flattened_bound_box)

    # Fill in the source buffer
    for i, idx in enumerate(sour_idx_list):
        source_buffer[idx] = source_tr[i]

    # Fill in the target buffer
    for i, idx in enumerate(tar_idx_list):
        target_buffer[idx] = target[i]

    source_buffer_3d = np.reshape(np.array(source_buffer), (H, W, 3))
    target_buffer_3d = np.reshape(np.array(target_buffer), (H, W, 3))

    window = 10

    matches = []
    source_used = []
    for i, row in enumerate(source_buffer_3d):
        win_y_top = np.minimum(target_buffer_3d.shape[0], i + window // 2)
        win_y_bot = np.maximum(0, i - window // 2)
        for j, point_xyz in enumerate(row):

            if np.any(point_xyz == np.inf):
                continue

            win_x_l = np.maximum(0, j - window // 2)
            win_x_r = np.minimum(target_buffer_3d.shape[1], j + window // 2)
            window_content = target_buffer_3d[win_y_bot: win_y_top, win_x_l: win_x_r].reshape(-1, 3)
            if np.all(window_content == np.inf):
                continue

            matches.append(min_dist([point_xyz], window_content)[0])
            source_used.append(point_xyz)

    return np.array(matches).squeeze(1), np.array(source_used)
