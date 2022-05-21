from statistics import variance
from tkinter import N
from winreg import DeleteValue
import h5py
import numpy as np
from math import radians, cos, sin, tan
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from supplemental_code.supplemental_code import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LatentNet(nn.Module):
    def __init__(self, rotation, translation, alpha, delta):
        super(LatentNet, self).__init__()
        self.rotation = torch.tensor(rotation, dtype=torch.float32)
        self.rotation = nn.Parameter(self.rotation, requires_grad = True)
        self.translation = torch.tensor(translation, dtype=torch.float32)
        self.translation = nn.Parameter(self.translation, requires_grad = True)
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = nn.Parameter(self.alpha, requires_grad = True)
        self.delta = torch.tensor(delta, dtype=torch.float32)
        self.delta = nn.Parameter(self.delta, requires_grad = True)

    def forward(self, bfm):
        G, _, _ = morphable_model(bfm, self.alpha.detach().numpy(), self.delta.detach().numpy())
        _, image_2d = pinhole_camera_model(self.rotation.detach().numpy(), self.translation.detach().numpy(), G)
        uv, homog = load_landmark(image_2d)
        uv_2d = (uv.T / homog).T
        uv_2d = torch.tensor(uv_2d, dtype=torch.float32)
        return uv_2d

def load_weights(bfm, model, dim, M):
    id = np.asarray(bfm[model+"/model/mean"] , dtype=np.float32)
    id = np.reshape(id, (-1, 3))
    basis =  np.asarray(bfm[model+"/model/pcaBasis"] , dtype=np.float32)
    basis = np.reshape(basis, (-1, 3, dim))[:, :, :M]
    variance = np.asarray(bfm[model+"/model/pcaVariance"] , dtype=np.float32)[:M]
    
    return id, basis, variance

def morphable_model(bfm, alpha, delta):
    # Select facial identity from BFM
    mu_id, basis_id, variance_id = load_weights(bfm, "shape", 199, 30)
    # Select expression from BFM
    mu_exp, basis_exp, variance_exp = load_weights(bfm, "expression", 100, 20)
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

def visualize_landmark(pred, name):
    plt.scatter(pred[:,0], pred[:,1], s = 12)
    plt.grid(True)
    plt.savefig(name)

def loss_estimation(uv, gt_landmark, la, ld, alpha, delta):
    gt_landmark = torch.tensor(gt_landmark, dtype=torch.float32)
    alpha = torch.tensor(alpha, dtype=torch.float32)
    delta = torch.tensor(delta, dtype=torch.float32)
    L_lan = torch.mean(torch.pow((uv - gt_landmark), 2))
    L_reg = la*torch.sum(torch.pow(alpha, 2)) + ld*torch.sum(torch.pow(delta, 2))
    L_fit = L_lan + L_reg
    return L_fit

def train(bfm, lr, epochs, seed, rotation, translation, alpha, delta, la, ld):
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LatentNet(rotation, translation, alpha, delta)
    optimizer = optim.SGD(model.parameters(), lr = lr)
    model.to(device)

    img =  cv2.imread('./cv2_2022_assignment3/input/000031.jpg')
    gt_landmark = detect_landmark(img)

    losses = []
    model.train()
    prev_loss = 1000000000000
    prev_model = model
    for epoch in range(epochs):
        uv = model(bfm)
        loss = loss_estimation(uv, gt_landmark, la, ld, model.alpha, model.delta)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = torch.mean(losses)
        if (train_loss > prev_loss + 0.1):
            print('Training stopped due to early stopping')
            best_model = prev_model
            break
        else:
            prev_loss = train_loss
            prev_model = model
            
        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format("train", epoch+1, epochs, train_loss))
        print("-------------------------------------------------------------------------------------")
    
    return losses, best_model

def main():
    #--------------------------------------------------------------------#
    # part 4.2.1
    bfm = h5py.File("model2017-1_face12_nomouth.h5", "r")
    # Uniform distribution
    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)

    G, triangle, color = morphable_model(bfm, alpha, delta)
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
    visualize_landmark(uv_2d, 'landmark.jpg')
    #--------------------------------------------------------------------#
    # part 4.2.3.a
    img = cv2.imread('neutral_image.jpg')
    gt_landmark = detect_landmark(img)
    for (x, y) in gt_landmark:
        cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
        cv2.imwrite('landmark_gt.jpg', img)
    #--------------------------------------------------------------------#
    # part 4.2.3.b
    lr = 0.01
    epochs = 200
    seed = 42
    la = 1
    ld = 1
    losses, best_model = train(bfm, lr, epochs, seed, rotation, translation, alpha, delta, la, ld)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    G, triangle, color = morphable_model(bfm, best_model.alpha, best_model.delta)
    save_obj('trained_3D.obj', G, color, triangle.T)
    Trans_G, image_2d = pinhole_camera_model(best_model.rotation, best_model.translation, G)
    save_obj('trained_3D_right_rot.obj', Trans_G.T[:, :3], color, triangle.T)
    uv, homog = load_landmark(image_2d)
    uv_2d = (uv.T / homog).T
    visualize_landmark(uv_2d, 'trained_landmark.jpg')
    #--------------------------------------------------------------------#

main()
