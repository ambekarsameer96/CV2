import numpy as np
# import open3d as o3d
# import os
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

    # ***  you need to clean the point cloud using a threshold ***
    distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]

    if vis:
        vis_open3d(pcd_arr_cleaned)

    return pcd_arr_cleaned


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy'))
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    return source, target


############################
#     ICP                  #
############################
def min_dist(source, dest):
    idx = []

    for sample in source:
        dist = np.linalg.norm(sample - dest, axis=1)
        idx.append(np.argmin(dist))

    result = dest[idx]
    return result

# pcd_arr_cleaned = open3d("Data/data/0000000000.pcd", True)
# A1 = pcd_arr_cleaned @ R + t


# initialize R and t
R = np.eye(3)
t = np.zeros((3,1))

# compute translation
np.random.seed(42)
A1 = np.random.rand(4,3)
A2 = np.random.rand(4,3)

# compute closest points from A2 to A1
di = min_dist(A1, A2)

# SVD: mean and centereing
A1_mean = np.mean(A1, axis=0)
A2_mean = np.mean(di, axis=0)
centered_A1 = A1 - A1_mean
centered_A2 = di - A2_mean

# SVD: covariance matrix
W = np.eye(len(di))
cov = centered_A1.T @ W @ centered_A2
u, s, vh = np.linalg.svd(cov)

# SVD: computing svd, R and t
mid = np.eye(3)
mid[-1][-1] = np.linalg.det(vh.T @ u.T)

r = vh.T @ mid @ u
t = np.mean(A1, axis=0) - r @ centered_A1

print(r)
print(t)
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
