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

def ICP(source, target, th=0.9):
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
        print(RMS)

        if RMS < th or np.abs(RMS - RMS_old) < 1e-3:
            break

        source_old = source_trans
        R, t = calc_R_t(source_old, target_BF)
    return R, t

source, target = open_wave_data()
# source, target = open_bunny_data()

ratio = 5
source = sm.multi_resolution(source, ratio)
target = sm.multi_resolution(target, ratio)

R, t = ICP(source, target, 0.3)

source_tr = source @ R + t

o3dvis = True
matplotvis = False

if o3dvis:
    final = np.append(source, source_tr, axis=0)
    vis_open3d(final)

if matplotvis:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(source[:,0], source[:,1], source[:,2], s=1)
    ax.scatter3D(source_tr[:,0], source_tr[:,1], source_tr[:,2], s=1)
    plt.show()
