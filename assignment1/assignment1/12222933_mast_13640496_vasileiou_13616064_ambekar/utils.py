import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

DATA_DIR = 'Data' # This depends on where this file is located. Change for your needs.

def vis_open3d(data):
    """
    Visualize pointcloud using open3d.
    """
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([vis_pcd])

def vis_matplotlib(source, target, title):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(target[:,0], target[:,1], target[:,2], s=1)
    ax.scatter3D(source[:,0], source[:,1], source[:,2], s=1)
    plt.show()
    if title:
        plt.savefig(f"media/{title}")


def open_pointcloud_data(path, origin=[[0,0,0]]):
    """
    Open and clean cloud data of points further than 2m away from origin.
    In this implementation, the origin is assumed to be at (0,0,0).
    """
    pcd = o3d.io.read_point_cloud(path)

    # convert into ndarray
    pcd_arr = np.asarray(pcd.points)

    # clean the point cloud using a threshold
    distances = np.sqrt(np.sum((pcd_arr - origin) ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]

    return pcd_arr_cleaned


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy')).T
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy')).T
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy')).T
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy')).T
    return source, target


def file_picker(file_path):
    """
    Obtains pcd files from a given directory.
    """
    files1 = os.listdir(file_path)
    files1 = sorted(files1)
    l1 = []
    for file1 in files1:
        if '.pcd' in str(file1):
            if not 'normal' in str(file1):
                file1 = os.path.join(file_path, file1)
                l1.append(file1)
    return l1

def add_noise(points, noise_variance, noise_amount):
    """
    Adds noise to pointcloud
    """
    mean = np.mean(points, axis=0)
    cov_vec = [noise_variance] * points.shape[1]
    cov = np.cov(points, rowvar=False) * noise_variance
    noise = np.random.multivariate_normal(mean=mean, cov=cov, size=noise_amount)
    return np.vstack((points, noise))
