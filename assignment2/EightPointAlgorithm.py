import argparse
import cv2 as cv
import numpy as np


def load_image_gray(frame1, frame2):
    img1 = cv.imread('./Data/House/House/frame000000'+frame1+'.png')
    img2 = cv.imread('./Data/House/House/frame000000'+frame2+'.png')
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    return img1, img2

def calculate_keypoint_matching(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, matches

def matrix_T(x,y):
    mx = np.mean(x)
    my = np.mean(y)

    d = np.square(x - mx) + np.square(y - my)
    d = np.sqrt(d)
    d = np.mean(d)

    T = np.array([[(np.sqrt(2)/d), 0, (-mx*np.sqrt(2)/d)],
                [0, (np.sqrt(2)/d), (-my*np.sqrt(2)/d)],
                [0,0,1]])

    return T

def matrix_A(x1, y1, x2, y2):
    A = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, np.ones((len(x1)))]).T

    return A

def matrix_F(V):
    #the entries of F are the components of the column of V corresponding to the smallest singular value.
    F = V[-1].reshape(3, 3)
    #find the SVD of F 
    UF, DF, VF_T = np.linalg.svd(F)
    #set the smallest singular value in the diagonal matrix DF to zero
    DF[-1] = 0
    #recompute F
    DF = np.diag(DF)
    F = UF.dot(DF.dot(VF_T))

    return F


def eight_point(first_frame, second_frame, method):
    #load two cosequitive frames
    img1, img2 = load_image_gray(first_frame, second_frame)

    #calculate matching keypoints
    kp1, kp2, matches = calculate_keypoint_matching(img1, img2)
    
    templist = []
    for match in matches:
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt
        templist.append([x1, y1, x2, y2])

    x1 = np.array([row[0] for row in templist])
    y1 = np.array([row[1] for row in templist])
    x2 = np.array([row[2] for row in templist])
    y2 = np.array([row[3] for row in templist])

    #apply normalization in case of normal
    if method == 'normal':
        T1 = matrix_T(x1,y1)
        T2 = matrix_T(x2,y2)
        x1, y1 = T1.dot(x1), T1.dot(y1)
        x2, y2 = T2.dot(x2), T2.dot(y2)


    #correspondence = np.matrix(templist)

    A = matrix_A(x1, y1, x2, y2)
    #find the SVD of A
    U, D, V_T = np.linalg.svd(A)
    
    F = matrix_F(V_T)

    #apply de-normalization in case of normal
    if method == 'normal':
        F = (T2.T).dot(F.dot(T1))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--first_frame', default='01', type=str, help='first_frame')
    parser.add_argument('--second_frame', default='02', type=str, help='second_frame')
    parser.add_argument('--method', default='normal', type=str, help='choose method between simple/normal/ransac', choices=['simple', 'normal', 'ransac'])

    args = parser.parse_args()
    kwargs = vars(args)

    eight_point(**kwargs)
