import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from copy import deepcopy

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.

# == Load data ==
def vis_open3d(data):
    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([vis_pcd])


def open3d(path, vis=False):
    pcd = o3d.io.read_point_cloud(path)

    # convert into ndarray
    pcd_arr = np.asarray(pcd.points)

    # clean the point cloud using a threshold
    # distances to origin (0,0,0)
    distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]

    if vis:
        vis_open3d(pcd_arr_cleaned)

    return pcd_arr_cleaned


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy')).T
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy')).T
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy')).T
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy')).T
    return source, target


############################
#     ICP                  #
############################
def min_dist(source, target, informative=False):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    idx = []
    # nearest_dist = []
    for sample in source:
        dist = np.linalg.norm((target - sample) ** 2, axis=1)
        # for k in range(10):
        #     cand = np.argpartition(dist, k)
        #     if cand[k] not in idx:
        #         idx.append(cand[k])
        #         break

        # nearest_dist.append(np.min(dist))
        idx.append(np.argmin(dist))

    result = target[idx]

    if informative:
        result = find_informative_regions(nearest_dist, idx)

    return result

def calc_R_t(source_trans, target):
    """
    compute R and t for a given translated source and target point cloud.
    """
    target_BF = min_dist(source_trans, target)
    # SVD: mean and centereing
    source_trans_mean = np.mean(source_trans, axis=0)
    target_mean = np.mean(target_BF, axis=0)

    centered_source_trans = source_trans - source_trans_mean
    centered_target = target_BF - target_mean

    # SVD: covariance matrix
    W = np.eye(len(source_trans))
    cov = centered_source_trans @ W @ centered_target.T

    # SVD: computing svd, R and t
    u, s, vh = np.linalg.svd(cov)

    mid = np.eye(3)
    mid[-1][-1] = np.linalg.det(vh.T @ u.T)

    R = vh.T @ mid @ u
    t = target_mean - R @ source_trans_mean

    return R, t

def multiply_RT_source(source, R, t):
    return source @ R + t

def ICP(source, target, th=0.9):
    """
    Perform ICP algorithm and return rotation matrix and translation vector.
    """
    # initialize R and t
    R_best = np.eye(3)
    t_best = np.zeros((3))

    rms_prev = 0.1
    rms = 0.2

    rms_best = np.inf
    # source_copy = deepcopy(source)
    source_trans = multiply_RT_source(source, R_best, t_best)
    for _ in range(50):

        R, t = calc_R_t(source_trans, target)

        # source_trans_copy = deepcopy(source_trans)
        source_trans = multiply_RT_source(source_trans, R, t)
        rms = np.sqrt(np.mean(np.linalg.norm((source - source_trans) ** 2, axis=1)))

        # if rms < rms_best:
        #     R_best, t_best = R, t
        #     rms_best = rms
        # else:
        #     R, t = R_best, t_best
        #     rms = rms_best
        # if np.abs(rms_prev - rms) <= th:
        #     break
        # else:
        #     rms_prev = rms

    return R, t

def ICP_iterative(source, target, th=0.9):
    """
    Perform ICP algorithm and return rotation matrix and translation vector.
    """
    # initialize R and t
    R_best = np.eye(3)
    t_best = np.zeros((3))

    R_agg = np.eye(3)
    t_agg = np.ones((3,1))


    rms_prev = 0.1
    rms = 0.2

    rms_best = np.inf
    source_prev = multiply_RT_source(source, R_best, t_best)
    while True:

        R, t = calc_R_t(source_prev, target)
        source_trans = multiply_RT_source(source_prev, R, t)

        source_tr = source_trans @ R + t
        final = np.append(target, source_tr, axis=0)
        vis_open3d(final)

        rms = np.sqrt(np.mean(np.linalg.norm((source_prev - source_trans) ** 2, axis=1)))
        source_prev = source_trans
        R_agg = R_agg.T @ R
        t_agg = t_agg.T @ t
        print(rms)
        if np.abs(rms_prev - rms) <= th:
            break
        else:
            rms_prev = rms

    return R_agg, t_agga


def multi_res_sampling(points):
    return points[::10]

def find_informative_regions(nearest_dist, nearest_ind):
    #find informative regions
    informative_regions = []
    for i in range(len(nearest_dist)):
        if nearest_dist[i] < 0.01:
            informative_regions.append(nearest_ind[i])
    return informative_regions

source, target = open_wave_data()
# source, target = open_bunny_data()

source_sub = multi_res_sampling(source)
target_sub = multi_res_sampling(target)

R, t = ICP(source_sub, target_sub, 1e-4)


# visualize final matching
source_tr = source @ R + t
final = np.append(source, source_tr, axis=0)
vis_open3d(final)
