import argparse
import enum
import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt

def load_image_gray(frame1, frame2):
    img1 = cv.imread('./Data/House/House/frame000000'+frame1+'.png')
    img2 = cv.imread('./Data/House/House/frame000000'+frame2+'.png')
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    return img1, img2

def calculate_keypoint_matching(img1, img2, num_points=8):
    sift = cv.SIFT_create()
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

def matrix_F(x1, y1, x2, y2, p1, p2, method):
    #apply normalization in case of normal
    if method == 'normal':
        T1 = matrix_T(x1,y1)
        T2 = matrix_T(x2,y2)
        x1, y1, _ = T1.dot(p1.T)
        x2, y2, _ = T2.dot(p2.T)

    A = matrix_A(x1, y1, x2, y2)
    #find the SVD of A
    U, D, V_T = np.linalg.svd(A)

    #the entries of F are the components of the column of V corresponding to the smallest singular value.
    F = V_T[-1].reshape(3, 3)
    #find the SVD of F
    UF, DF, VF_T = np.linalg.svd(F)
    #set the smallest singular value in the diagonal matrix DF to zero
    DF[-1] = 0
    #recompute F
    DF = np.diag(DF)
    F = UF.dot(DF.dot(VF_T))

    #apply de-normalization in case of normal
    if method == 'normal':
        F = (T2.T).dot(F.dot(T1))

    return F

def Sampson_distance(F, p1, p2):
    d = []
    for i in range(len(p1)):
        num = np.square(p2[i].T.dot(F.dot(p1[i])))
        denom = np.square(F.dot(p1[i]))[0] + np.square(F.dot(p1[i]))[1] + np.square(F.T.dot(p2[i]))[0] + np.square(F.T.dot(p2[i]))[1]
        d.append(num / denom)
    return d

def F_RANSAC(x1, x2, y1, y2, p1, p2, num_iter, thr):
    indices = list(range(len(p1)))
    max_d_thr = []

    for i in range(num_iter):
        idx = random.sample(indices, 8)
        x1_8, x2_8 = x1[idx], x2[idx]
        y1_8, y2_8 = y1[idx], y2[idx]
        p1_8, p2_8 = p1[idx], p2[idx]

        F = matrix_F(x1_8, y1_8, x2_8, y2_8, p1_8, p2_8, 'normal')
        d = Sampson_distance(F, p1, p2)
        #print(np.array(d).shape)

        d_thr = []
        ind_thr = []
        for j, dim in enumerate(d):

            if dim <= thr:
                d_thr.append(dim)
                ind_thr.append(j)

        if len(d_thr) > len(max_d_thr):
            max_d_thr = d_thr
            max_ind = ind_thr

    print(len(max_ind))
    x1_max, x2_max = x1[max_ind], x2[max_ind]
    y1_max, y2_max = y1[max_ind], y2[max_ind]
    p1_max, p2_max = p1[max_ind], p2[max_ind]
    F = matrix_F(x1_max, y1_max, x2_max, y2_max, p1_max, p2_max, 'normal')
    return F, max_ind

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        img1 = cv.circle(img1,(x1, y1),5,color,-1)
        img2 = cv.circle(img2,(x2, y2),5,color,-1)
    return img1,img2

def plotting(img1, img2, pts1, pts2, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    print(f"{pts2.shape=}")
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)

    idx = np.random.choice(np.arange(len(pts1)), 8)

    img5,img6 = drawlines(img1,img2,lines1[idx],pts1[idx],pts2[idx])

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2[idx],pts2[idx],pts1[idx])
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

def eight_point(first_frame, second_frame, method, num_iter, thr):
    #load two cosequitive frames
    random.seed(42)
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

    p1 = np.array(list(zip(x1, y1, np.ones((len(x1))))))
    p2 = np.array(list(zip(x2, y2, np.ones((len(x2))))))

    if method in ['simple', 'normal']:
        F = matrix_F(x1, y1, x2, y2, p1, p2, method)
    else:
        F, _ = F_RANSAC(x1, x2, y1, y2, p1, p2, num_iter, thr)

    p1 = p1[:,0:2]
    p2 = p2[:,0:2]

    plotting(img1, img2, p1, p2, F)


    return F, p1, p2


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--first_frame', default='01', type=str, help='first_frame')
    parser.add_argument('--second_frame', default='02', type=str, help='second_frame')
    parser.add_argument('--method', default='normal', type=str, help='choose method between simple/normal/ransac', choices=['simple', 'normal', 'ransac'])
    parser.add_argument('--num_iter', default=300, type=int, help='number of iterations for RANSAC method')
    parser.add_argument('--thr', default=0.3, type=int, help='threshold on Sampson disatances for RANSAC method')

    args = parser.parse_args()
    kwargs = vars(args)

    print(eight_point(**kwargs)[0])
