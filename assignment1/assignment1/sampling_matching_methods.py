import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity
from copy import deepcopy

def multires_sampling(points, amount=10):
    return points[::amount]

def unif_sampling(points, ratio=4):
    idx = np.random.choice(len(points), len(points) // ratio)
    return points[idx]

def no_sampling(points):
    return points

# def density_sampling(source):
#     #sub-sampling from source by density
#     #find nearest neighbors
#     tree = KDTree(source)
#     nearest_dist, nearest_ind = tree.query(source, k=10)
#     idx = np.arange(len(source))[np.linalg.norm(nearest_dist, axis=1) < 0.0025]
#     print(idx.shape, source.shape)
#     #give list of indices with good amount of neigbors, such that the density is high.
#     return source[idx]

def density_sampling(points):
    #sub-sampling from points by density
    #find nearest neighbors
    kde = KernelDensity(leaf_size=20, bandwidth=0.2).fit(points)
    density_est = kde.score_samples(points)
    idx = np.arange(len(points))[density_est > density_est.mean()]
    #give list of indices with good amount of neigbors, such that the density is high.
    return points[idx]


def min_dist(source, target, pr=False):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    idx = []
    for sample in source:
        dist = np.linalg.norm(target - sample, axis=1)
        idx.append(np.argmin(dist))

        if pr:
            pass

    if pr:
        idx = np.unravel_index(idx, target.shape[:2])
        print(idx)

    result = target[idx]
    return result


#2.2 kd-tree
#obtain points from Source and Target, try using KD tree, pair with matching.
def kd_method(source, target):
    tree = KDTree(target)
    nearest_ind = tree.query(source, k=1)[1].squeeze(1)
    result = target[nearest_ind]
    return result

def min_dist_buffer(source, target):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    idx = []
    temp_target = deepcopy(target)
    for sample in source:
        dist = np.linalg.norm((temp_target - sample) ** 2, axis=1)
        idx.append(np.argmin(dist))
        temp_target[np.argmin(dist)] = np.inf
    return idx

def z_buffer(source_tr, target, H=128, W=100):
    # Union of A1, A2
    sour_tar_union = np.vstack((source_tr, target))
    flattened_a = sour_tar_union.ravel()
    # x,y-plane
    x_dim = flattened_a[:12800]
    y_dim = flattened_a[12800:2*12800]
    # Minimum enclosing box
    min_x = min(x_dim)
    max_x = max(x_dim)
    min_y = min(y_dim)
    max_y = max(y_dim)
    # Corner points of the box
    bottom_left = min_x, min_y
    top_left = min_x, max_y
    top_right = max_x, max_y
    bottom_right = max_x, min_y
    # Initialize the bounding box
    bound_box = [[0]*W for i in range(H)]
    # Dimensions of the box
    max_hor = abs(max_x - min_x)
    max_ver = abs(max_y - min_y)
    # Step for each dimenssion
    hor_const = max_hor/(W-1)
    ver_const = max_ver/(H-1)
    # Fill in the bounding box
    count = 0
    for i in range(H):
        for j in range(W):
            if count % W == 0:
                x_cor = min_x
                y_cor = max_y - i*ver_const
                bound_box [i][j] = (x_cor, y_cor)
            else:
                x_cor = min_x + j*hor_const
                bound_box [i][j] = (x_cor, y_cor)
            count += 1

    # Source and target xy-planes
    source_xy = np.delete(source_tr, 2, 1)
    target_xy = np.delete(target, 2, 1)
    flattened_bound_box = np.asarray(bound_box).reshape(12800,2)
    # Initialize source, target buffers
    source_buffer = np.matrix(np.ones((12800, 3)) * np.inf)
    target_buffer = np.matrix(np.ones((12800, 3)) * np.inf)
    # List with pointers to the minimum distance elements
    sour_idx_list = min_dist_buffer(source_xy, flattened_bound_box)
    tar_idx_list = min_dist_buffer(target_xy, flattened_bound_box)
    # Fill in the source buffer
    for i, idx in enumerate(sour_idx_list):
        original_x = source_tr[i][0]
        original_y = source_tr[i][1]
        original_z = source_tr[i][2]
        source_buffer[idx, 0] = original_x
        source_buffer[idx, 1] = original_y
        source_buffer[idx, 2] = original_z
    source_buffer_3d = np.reshape(np.array(source_buffer), (128, 100, 3))
    # Fill in the target buffer
    for i, idx in enumerate(tar_idx_list):
        original_x = target[i][0]
        original_y = target[i][1]
        original_z = target[i][2]
        target_buffer[idx, 0] = original_x
        target_buffer[idx, 1] = original_y
        target_buffer[idx, 2] = original_z
    target_buffer_3d = np.reshape(np.array(target_buffer), (128, 100, 3))

    window = 10

    matches = []

    for i, y in enumerate(source_buffer_3d):
        win_y_top = np.minimum(target_buffer_3d.shape[0], i + window // 2)
        win_y_bot = np.maximum(0, i - window // 2)
        for j, x in enumerate(y):

            if np.all(x == np.inf):
                continue

            win_x_l = np.maximum(0, j - window // 2)
            win_x_r = np.minimum(target_buffer_3d.shape[1], j + window // 2)
            window_content = target_buffer_3d[win_y_bot: win_y_top, win_x_l: win_x_r]
            matches.append(min_dist([x], window_content, pr=True))

    return matches
