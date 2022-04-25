from audioop import add
import numpy as np
import open3d as o3d
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import sampling_methods as sm
from scipy.stats import multivariate_normal


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

def ICP(source, target, th=0.9, iterative=True):
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

    print(RMS)
    return R, t

def run_ICP(source, target, o3dvis=False, matplotvis=False):
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

    return R, t

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

def z_buffer(source_tr, target, H, W):
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

source, target = open_wave_data()
# source, target = open_bunny_data()

# Using all points
R, t = run_ICP(source, target)

# Uniform sub-sampling
# source_unif = sm.unif_sampling(source)
# target_unif = sm.unif_sampling(target)
# R, t = run_ICP(source_unif, target_unif)

# Random sub-sampling
# TO ADD

# Multi-resolution sub-sampling
# ratio = 5
# source_multires = sm.multires_sampling(source, ratio)
# target_multires = sm.multires_sampling(target, ratio)
# R, t = run_ICP(source_multires, target_multires)

# Informative region sub-sampling
# TO ADD

# KDTree
# TO ADD

# z-buffer
# TO ADD


# source_tr = source @ R.T + t
# z_buffer(source_tr, target, 128, 100)
############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.
#  The first frame is used as the reference frame.
#  The second frame is used as the target frame.
#  The camera poses are estimated using the ICP algorithm.
#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

def file_picker(file_path):
    files1 = os.listdir(file_path)
    l1 = []
    for file1 in files1:
        if '.pcd' in str(file1):
            if not 'normal' in str(file1):
                #print('\n')
                #print(file1)
                file1 = os.path.join(file_path, file1)
                l1.append(file1) 
    return l1

file_path = './Data/data/'
file_list = file_picker(file_path)

def read_pcd_sec3(filename1):
    cur_pcd = o3d.io.read_point_cloud(filename1)
    pcd_arr = np.asarray(cur_pcd.points)
    distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]
    
    return pcd_arr_cleaned

def add_noise(pcd):
    mean = np.mean(pcd)
    std = np.std(pcd)
    noise = multivariate_normal.pdf(pcd[0], mean=mean, cov=std)
    #print(noise)
    #return noise
    return noise

def add_noise_D(pcd):
    noise_level = 5
    D = 3
    noise = np.random.multivariate_normal(
            mean=np.zeros(D), cov=np.diag([noise_level for _ in range(D)]),
        )
    source_noisy = noise.reshape(D, 1) + pcd
    return source_noisy

#pick 2,4,10th files and send to main_ICP 
source = read_pcd_sec3(file_list[0])
target = read_pcd_sec3(file_list[2])
source = add_noise_D(source)
        
for i, file1 in enumerate(file_list):
    
    if (i+1)%2:
        #call the ICP function and update the R,T
        #do some preprocessing , first add noise 

        # if target is str:
        #     target = read_pcd_sec3(file1[i+1])
        #print('Source', file1)
        print('Target',file1 )
        print(source.shape)
        print('-----')
        print(target.shape)
        #noise adder

        target = add_noise_D(target)
        #print('Noise', target2.shape)

        R,T = ICP(source, target, th=0.9, iterative=True)
        source = R
        target = read_pcd_sec3(file1)
        #read files 

        






#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################
