import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.


######                                                           ######
##      notice: This is just some example, feel free to adapt        ##
######                                                           ######


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
def min_dist(source, dest):
    """
    Get point with minimal distance to source point for each point in source.
    """
    idx = []

    for sample in source:
        dist = np.linalg.norm(sample - dest, axis=1)
        idx.append(np.argmin(dist))

    result = dest[idx]
    return result


def calc_R_t(source_trans, target):
    """
    compute R and t for a given translated source and target point cloud.
    """
    # compute closest points from A2 to A1
    target_BF = min_dist(source_trans, target)

    # SVD: mean and centereing
    source_trans_mean = np.mean(source_trans, axis=0)
    target_mean = np.mean(target_BF, axis=0)
    centered_source_trans = source_trans - source_trans_mean
    centered_target = target_BF - target_mean

    # SVD: covariance matrix
    W = np.eye(len(source_trans))
    cov = centered_source_trans.T @ W @ centered_target

    # SVD: computing svd, R and t
    u, s, vh = np.linalg.svd(cov)

    mid = np.eye(3)
    mid[-1][-1] = np.linalg.det(vh.T @ u.T)

    R = vh.T @ mid @ u
    t = target_mean - R @ source_trans_mean

    return R, t


def ICP(source, target, th=0.9):
    """
    Perform ICP algorithm and return rotation matrix and translation vector.
    """
    # initialize R and t
    R = np.eye(3)
    t = np.zeros((3))

    rms = np.inf
    source_trans = source @ R + t

    while rms > th:
        R, t = calc_R_t(source_trans, target)

        source_trans = source @ R + t
        rms = np.sqrt(np.mean(np.linalg.norm(source - source_trans, axis=1)))
        print(f"RMS: {rms}")

    return R, t

source, target = open_wave_data()
vis_open3d(source)
vis_open3d(target)


R, t = ICP(source, target, 0.89)

source_tr = source @ R + t

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


#a = np.append(source, source_tr, axis=0)
#vis_open3d(a)

###### 0. (adding noise)

###### 1. initialize R= I , t= 0

###### go to 2. unless RMS is unchanged(<= epsilon)

###### 2. using different sampling methods

###### 3. transform point cloud with R and t

###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach

###### 5. Calculate RMS

###### 6. Refine R and t using SVD

# plt.scatter(A1[:,0], A1[:, 1], label='A1', alpha=0.5)
# plt.scatter(A2[:, 0], A2[:, 1], label='A2', alpha=0.5)
# plt.scatter(di[:, 0], di[:, 1], label='min', alpha=0.5, s=3)
# plt.show()
############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################