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
def min_dist(source, target):
    """
    Get point with minimal distance to source from target for each point in source.
    """
    idx = []

    for sample in source:
        dist = np.linalg.norm((target - sample) ** 2, axis=1)
        idx.append(np.argmin(dist))

    result = target[idx]
    return result


def calc_R_t(source_trans, target):
    """
    compute R and t for a given translated source and target point cloud.
    """
    # compute closest points from A2 to A1 using euclidean distance
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

    rms_old = 0.1
    rms_new = 0.2
    source_trans = source @ R + t

    while np.abs(rms_old - rms_new) > th:
        R, t = calc_R_t(source_trans, target)

        source_trans = source @ R + t

        rms_old = rms_new
        rms_new = np.sqrt(np.mean(np.linalg.norm((source - source_trans) ** 2, axis=1)))
        print(f"RMS: {rms_new}")

    return R, t

source, target = open_wave_data()
# source, target = open_bunny_data()
# vis_open3d(source)
# vis_open3d(target)
# R, t = ICP(source, target, 0.05)
#
# source_tr = source @ R + t
#
# a = np.append(source, source_tr, axis=0)
# vis_open3d(a)

np.random.seed(42)
a = np.random.rand(10,2)
b = np.random.rand(20,2)
t = min_dist(a, b)
plt.scatter(a[:,0], a[:,1], s=50, label='a', alpha=0.6)
plt.scatter(b[:,0], b[:,1], s=50, label='b', alpha=0.6)

for p1, p2 in zip(a, t):
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    plt.plot(x, y)
plt.legend()
plt.show()
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

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(source_trans[:,0], source_trans[:,1], source_trans[:,2])
# ax.scatter3D(target_BF[:,0], target_BF[:,1], target_BF[:,2])
# for (i,j) in zip(source_trans, target_BF):
#     x = [i[0], j[0]]
#     y = [i[1], j[1]]
#     z = [i[2], j[2]]
#
#     ax.plot3D(x, y, z)
# plt.show()

# def plot3d(points, mean):
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.scatter3D(source[:,0], source[:,1], source[:,2])
#     ax.scatter3D(mean[0], mean[1], mean[2], s= 100)
#     plt.show()

############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################
