import numpy as np
import open3d as o3d
import os

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.


######                                                           ######
##      notice: This is just some example, feel free to adapt        ##
######                                                           ######


# == Load data ==
def open3d_example():
    pcd = o3d.io.read_point_cloud("Data/data/0000000000.pcd")
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    pcd_arr_cleaned = pcd_arr[pcd_arr[:, 2] > 0.1]



    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
    o3d.visualization.draw_geometries([vis_pcd])


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy'))
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    return source, target

open3d_example()
open_wave_data()
open_bunny_data()
############################
#     ICP                  #
############################

#start here 
#take all pcd files except normal  
import os 
import numpy as np
path1 = './Data/data/'
l1 = os.listdir(path1)
l2 = []

for file1 in l1:
    if not 'normal' in str(file1):
        l2.append(file1)

#new list is l2 



###### 0. (adding noise)
# add noise to the data
def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


#follow the PDF
def brute_force_matcher_using_distance_source_target(source,target):
    # brute force matching using distance between source and target
    source_target_matches = []
    for i in range(len(source)):
        for j in range(len(target)):
            if np.linalg.norm(source[i] - target[j]) < 0.01:
                source_target_matches.append([i, j])
    return source_target_matches



###### 1. initialize R= I , t= 0
def init_RT():
    R = np.eye(3)
    t = np.zeros(3)
    return R, t
#call init function 
R,t = init_RT()

#multiply R and t with soruce 
def multiply_RT_source(source, R, t):
    source_new = np.zeros(source.shape)
    for i in range(len(source)):
        source_new[i] = R.dot(source[i]) + t
    return source_new


#update R and t values using SVD 
def SVD_update(source, target, R, t):
    # SVD
    A = np.zeros((len(source), 3))
    for i in range(len(source)):
        A[i] = np.cross(source[i], target[i])
    U, S, V = np.linalg.svd(A)
    R_new = U.dot(V)
    t_new = np.mean(target - R.dot(source), axis=0)
    return R_new, t_new

R_new, t_new = SVD_update(source, target, R, t)

#check for values of RMSE
val = RMSE_checker(source, target)
if (val) < 0.5:
    cont_var = False










###### go to 2. unless RMS is unchanged(<= epsilon)
from sklearn.metrics import mean_squared_error
def RMSE_checker(source,target):
    rms = mean_squared_error(source,target, squared=False)
    if rms <= 0.01:
        return 0
    else:
        return 1

###### 2. using different sampling methods
def sampling(source, target, method):
    return
#methods for sampling are 


###### 3. transform point cloud with R and t

###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach

###### 5. Calculate RMS

###### 6. Refine R and t using SVD


############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################
