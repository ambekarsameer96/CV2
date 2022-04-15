import numpy as np
import open3d as o3d
import os
from scipy.stats import multivariate_normal

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.


######                                                           ######
##      notice: This is just some example, feel free to adapt        ##
######                                                           ######


# == Load data ==
def open3d_example():
    pcd = o3d.io.read_point_cloud("Data/data/0000000000_normal.pcd", remove_nan_points=True)
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]
    print(pcd_arr_cleaned)


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

'''def create_dataset(data_dir):
    pcd = []
    for data in os.listdir(data_dir):
        extend = data.split(".")[1]
        name = data.split(".")[0]
        if extend == "pcd":
            if "_" not in name:
                cur_pcd = o3d.io.read_point_cloud(data_dir+data)
                pcd_arr = np.asarray(cur_pcd.points)
                distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
                pcd_arr_cleaned = pcd_arr[distances < 2]
                pcd.append(pcd_arr_cleaned)
    return pcd'''

def read_pcd(data_dir):
    cur_pcd = o3d.io.read_point_cloud(data_dir)
    pcd_arr = np.asarray(cur_pcd.points)
    distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]
    
    return pcd_arr_cleaned

############################
#     ICP                  #
############################

###### 0. (adding noise)
def add_noise(data_dir):
    for data in os.listdir(data_dir):
        extend = data.split(".")[1]
        name = data.split(".")[0]
        if extend == "pcd":
            if "_" not in name:
                print(name)
                pcd = read_pcd(data_dir+data)
                mean = np.mean(pcd)
                std = np.std(pcd)
                noise = multivariate_normal.pdf(pcd[0], mean=mean, cov=std)
                print(noise)
    #return noise
    
###### 1. initialize R= I , t= 0

###### go to 2. unless RMS is unchanged(<= epsilon)

###### 2. using different sampling methods

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


if __name__ == "__main__":
    noise = add_noise("Data/data/")
    #print(noise)