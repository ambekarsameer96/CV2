import argparse
import enum
from multiprocessing.connection import wait
import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import open3d as o3d
from scipy.spatial import procrustes
import argparse

#read the point view matrix
def read_point_view_matrix(file_name):

    print('using file name ', file_name)
    pvm = np.loadtxt(file_name)
    print('Size of PVM matrix',pvm.shape)
    pvm = pvm - np.mean(pvm, axis=1).reshape(-1, 1)
    point_view_matrix = pvm

    return point_view_matrix

def structure_from_motion(args):
    #open the point view matrix file
    point_view_matrix = read_point_view_matrix(args.filename)
    print('using file name ', args.filename)
    pvm = np.loadtxt(args.filename)
    print('Size of PVM matrix',pvm.shape)
    pvm = pvm - np.mean(pvm, axis=1).reshape(-1, 1)
    point_view_matrix = pvm

    print('Global Point view matrix shape', point_view_matrix.shape)
    all_S = []

    #for every dense block calculate S and M and append to all_s
    for i in range(0, point_view_matrix.shape[0], args.consecutive_frames):
        b = []
        block11 = point_view_matrix[i]

        try:
            block12 = point_view_matrix[i+1]
            block21  = point_view_matrix[i+5]
            block22 = point_view_matrix[i+6]
            block31  = point_view_matrix[i+11]
            block32 = point_view_matrix[i+12]
        except:
            print('')
        b.append(block11)
        b.append(block12)

        b.append(block21)
        b.append(block22)

        b.append(block31)
        b.append(block32)

        b = np.array(b)
        # b is the block block now, now pass it to dense block filter
        b = dense_block_nan_remover(b)
        S = get_motion_structure(b)
        all_S.append(S)

    if args.single_dense_block:
        print('Generating using Single Dense Block')
        S = get_motion_structure(point_view_matrix)
        for i in range(0, point_view_matrix.shape[0], args.consecutive_frames):
            all_S.append(S)
        print('Size of all S', len(all_S))
        #now compute matrices and stitch
        stitch_matrix  = []
        stitch_matrix = np.array(stitch_matrix)
        m1, m2, m3=procrustes(all_S[0], all_S[1])
        stitch_matrix = all_S[0]
        for i in range(2, len(all_S)):

            frame1 = all_S[i]
            try:
                frame2 = all_S[i+1]
            except:
                print("")

            m1, m2, m3=procrustes(frame1, frame2)
            stitch_matrix = np.append(stitch_matrix, m2, axis=1)

            stitch.append(stitch_matrix)

        print('Shape of matrix', stitch_matrix.shape)
        visualise(stitch_matrix.T)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(stitch_matrix[0], stitch_matrix[1], stitch_matrix[2])
        plt.show()

    else:
        print('Generating using Multiple Dense Blocks')
        print('length of all S files', len(all_S))
        stitch_matrix = []
        #create a numpy array
        stitch_matrix = np.array(stitch_matrix)
        m1, m2, m3=procrustes(all_S[0], all_S[1])
        stitch_matrix = all_S[0]
        for i in range(2, len(all_S)):

            frame1 = all_S[i]
            try:
                frame2 = all_S[i+1]
            except:
                print("")

            m1, m2, m3=procrustes(frame1, frame2)
            stitch_matrix = np.append(stitch_matrix, m2, axis=1)

        print('Shape of matrix', stitch_matrix.shape)
        visualise(stitch_matrix.T)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(stitch_matrix[0], stitch_matrix[1], stitch_matrix[2])
        plt.show()


def get_motion_structure(pvm):
    U, W, Vtranspose = np.linalg.svd(pvm)
    W3r = np.diag(W[:3])
    V3r = Vtranspose[:3, :]
    S = np.sqrt(W3r).dot(V3r)
    return S


def dense_block_nan_remover(point_view_matrix):
    for i in range(point_view_matrix.shape[0]):

        for j in range(point_view_matrix.shape[1]):

            #if the value is nan then replace it with 0
            if np.isnan(point_view_matrix[i,j]):
                point_view_matrix[i,j] = 0

    return point_view_matrix



def visualise(S):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(S)
    o3d.io.write_point_cloud("1.ply", pcd)
    pcd_load = o3d.io.read_point_cloud("1.ply")
    o3d.visualization.draw_geometries([pcd_load])


if __name__ == '__main__':

    #add args
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='PointViewMatrix.txt', help='file name')
    #add argument for consecutive frames
    parser.add_argument('--consecutive_frames', type=int, default=3, help='number of consecutive frames')
    parser.add_argument('--single_dense_block', type=bool, default=False, help='single dense block')
    #parser variable for filename
    args = parser.parse_args()
    structure_from_motion(args)
