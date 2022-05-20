from statistics import variance
from tkinter import N
import h5py
import numpy as np
from math import radians, cos, sin, tan
import matplotlib.pyplot as plt
import cv2
from supplemental_code.supplemental_code import *

def load_weights(bfm, model, dim, M):
    id = np.asarray(bfm[model+"/model/mean"] , dtype=np.float32)
    id = np.reshape(id, (-1, 3))
    basis =  np.asarray(bfm[model+"/model/pcaBasis"] , dtype=np.float32)
    basis = np.reshape(basis, (-1, 3, dim))[:, :, :M]
    variance = np.asarray(bfm[model+"/model/pcaVariance"] , dtype=np.float32)[:M]
    
    return id, basis, variance

def morphable_model():
    bfm = h5py.File("model2017-1_face12_nomouth.h5", "r")
    # Select facial identity from BFM
    mu_id, basis_id, variance_id = load_weights(bfm, "shape", 199, 30)
    # Select expression from BFM
    mu_exp, basis_exp, variance_exp = load_weights(bfm, "expression", 100, 20)
    # Uniform distribution
    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    # G calculation
    id_var = mu_id + np.matmul(basis_id, alpha * np.sqrt(variance_id))
    exp_var = mu_exp + np.matmul(basis_exp, delta * np.sqrt(variance_exp))
    G = id_var + exp_var
    # Triangle topology
    triangle = np.asarray(bfm["shape/representer/cells"] , dtype=np.float32)
    # color 
    color = np.asarray(bfm["color/model/mean"] , dtype=np.float32)
    color = np.reshape(color, (-1, 3))
    
    return G, triangle, color

def trig(angle):
  r = radians(angle)
  return cos(r), sin(r)

def matrixT(rotation, translation):
    xC, xS = trig(rotation[0])
    yC, yS = trig(rotation[1])
    zC, zS = trig(rotation[2])
    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]

    Translate_matrix = np.array([[1, 0, 0, dX],
                               [0, 1, 0, dY],
                               [0, 0, 1, dZ],
                               [0, 0, 0, 1]])
    Rotate_X_matrix = np.array([[1, 0, 0, 0],
                              [0, xC, -xS, 0],
                              [0, xS, xC, 0],
                              [0, 0, 0, 1]])
    Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                              [0, 1, 0, 0],
                              [-yS, 0, yC, 0],
                              [0, 0, 0, 1]])
    Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                              [zS, zC, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

    return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))

def pinhole_camera_model(rotation, translation, G):
    # set new coordinates
    n = 12
    f = 108
    fov = 0.5
    aspect_ratio = 1
    t = tan(fov/2)*n
    b = -t
    r = t*aspect_ratio
    l = b
    # calculate transformation matrix T
    matT = matrixT(rotation, translation)
     # calculate viewpoint matrix
    Vp = np.array([[(r-l)/2, 0, 0, (r+l)/2],
                    [0, (t-b)/2, 0, (t+b)/2],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1]])
    # calculate projection matrix
    P = np.array([[(2*n)/(r-l), 0, (r+l)/(r-l), 0],
                    [0, (2*n)/(t-b), (t+b)/(t-b), 0],
                    [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                    [0, 0, -1, 0]])
    # add ones to 3D points
    G_4D = np.c_[G, np.ones(G.shape[0])]
    Trans_G = np.matmul(matT, G_4D.T)
    PI = np.matmul(Vp, P)
    image_2d = np.matmul(PI, Trans_G)
    return Trans_G, image_2d

def load_landmark(image_2d):
    f = open('supplemental_code/Landmarks68_model2017-1_face12_nomouth.anl','r')
    lines = f.readlines()
    # put indices to a list
    lines = [int(line) for line in lines]
    # find the mappings
    pred = image_2d.T[:, :2][lines]
    # return the prediction and the homogeneous coordinate
    return pred, image_2d.T[lines][:, 3]

def visualize_landmark(pred):
    plt.scatter(pred[:,0], pred[:,1], s = 12)
    plt.grid(True)
    plt.savefig('landmarks.png')

def main():
    #--------------------------------------------------------------------#
    # part 4.2.1
    G, triangle, color = morphable_model()
    save_obj('3D.obj', G, color, triangle.T)
    #--------------------------------------------------------------------#
    # part 4.2.2.a
    rotation = [0, 10, 0]
    translation = [0, 0, -500]
    Trans_G, image_2d = pinhole_camera_model(rotation, translation, G)
    save_obj('3D_right_rot.obj', Trans_G.T[:, :3], color, triangle.T)
    #--------------------------------------------------------------------#
    # part 4.2.2.b
    uv, homog = load_landmark(image_2d)
    # U,V projection (corresponding 2D pixel coordinate of each 3D point)
    uv_2d = (uv.T / homog).T
    visualize_landmark(uv_2d)
    #--------------------------------------------------------------------#
    # part 4.2.3.a
    img = cv2.imread('neutral_image.jpg')
    gt_landmark = detect_landmark(img)
    for (x, y) in gt_landmark:
        cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
        cv2.imwrite('landmark_gt.jpg', img)


main()
