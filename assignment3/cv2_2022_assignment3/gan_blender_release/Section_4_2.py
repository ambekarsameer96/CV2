from statistics import variance
import h5py
import numpy as np
from math import radians, cos, sin
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

G, triangle, color = morphable_model()
save_obj('3D.obj', G, color, triangle.T)

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

def pinhole_camera_model(rotation, translation):
    matT = matrixT(rotation, translation)
    P = np.array([[(2*n)/(r-l), 0, (r+l)/(r-l), 0],
                    [0, (2*n)/(t-b), (t+b)/(t-b), 0],
                    [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                    [0, 0, -1, 0]])

    Vp = np.array([[(r-l)/2, 0, 0, (r+l)/2],
                    [0, (t-b)/2, 0, (t+b)/2],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1]])