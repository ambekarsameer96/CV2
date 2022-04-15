import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import sampling_methods as sm

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

def min_dist(source, target):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    idx = []

    for sample in source:
        dist = np.linalg.norm(target - sample, axis=1)
        idx.append(np.argmin(dist))

    result = target[idx]
    return result

def calc_R_t(source, target):
    """
    compute R and t for a given translated source and target point cloud.
    """
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = (source - source_mean).T
    target_centered = (target - target_mean).T

    cov = source_centered @ target_centered.T
    U, S, VT = np.linalg.svd(cov)
    mid = np.eye(3)
    mid[-1][-1] = np.linalg.det(VT.T @ U.T)

    R = VT.T @ mid @ U.T
    t = target_mean - R @ source_mean

    return R, t

def ICP(source, target, th=0.9, iterative=False):
    """
    Perform ICP algorithm and return rotation matrix and translation vector.
    """
    R = np.eye(3)
    t = np.zeros(3)

    RMS_old = 100
    source_old = source

    while True:

        source_trans = source_old @ R.T + t
        target_BF = min_dist(source_trans, target)
        RMS = np.sqrt(np.mean(np.linalg.norm(target_BF - source_trans, axis=1)))

        if RMS < th or np.abs(RMS - RMS_old) < 1e-2:
            break

        R, t = calc_R_t(source_trans, target_BF)
        RMS_old = RMS
        if iterative:
            source_old = source_trans

    return R, t

def run_ICP(source, target):
    R, t = ICP(source, target, 0.3)
    source_tr = source @ R + t

    if o3dvis:
        final = np.append(source, source_tr, axis=0)
        vis_open3d(final)

    if matplotvis:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(source[:,0], source[:,1], source[:,2], s=1)
        ax.scatter3D(source_tr[:,0], source_tr[:,1], source_tr[:,2], s=1)
        plt.show()

source, target = open_wave_data()
# source, target = open_bunny_data()


# Using all points
run_ICP(source, target)

# Uniform sub-sampling
source_unif = sm.unif_sampling(source)
target_unif = sm.unif_sampling(target)
run_ICP(source_unif, target_unif)

# Random sub-sampling


# Multi-resolution sub-sampling
ratio = 5
source_multires = sm.multires_sampling(source)
target_multires = sm.multires_sampling(target)
run_ICP(source_multires, target_multires)

# Informative region sub-sampling

# window = 9
# for i, x in enumerate(source):
#     win_y_top = np.minimum(target.shape[0], i + window // 2)
#     win_y_bot = np.maximum(0, i - window // 2)
#     for j, y in enumerate(x):
#         win_x_l = np.maximum(0, j - window // 2)
#         win_x_r = np.minimum(target.shape[1], j + window // 2)
#
#         window_content = target[win_y_bot: win_y_top, win_x_l: win_x_r]
#         min_dist(source, target)
