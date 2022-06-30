import numpy as np
import cv2 as cv
import EightPointAlgorithm as epa
import os
import matplotlib.pyplot as plt

def file_picker(file_path):
    """
    Obtains pcd files from a given directory.
    """
    files = os.listdir(file_path)
    files = sorted(files)
    l1 = [file[-6:-4] for file in files]
    return l1


def get_matches(f1, f2, filtering=False, filter_ratio=0.4):
    img1, img2 = epa.load_image_gray(f1, f2)
    kp1, kp2, matches = epa.calculate_keypoint_matching(img1, img2)
    points1 = []
    points2 = []

    for match in matches:
        p1 = np.array(kp1[match.queryIdx].pt)
        p2 = np.array(kp2[match.trainIdx].pt)

        # Outlier removal
        if not filtering or np.linalg.norm(p1 - p2) < filter_ratio:
            points1.append(p1)
            points2.append(p2)

    return points1, points2


def chaining(file_path):
    file_IDs = file_picker(file_path)

    # 2m rows for m images, one placeholder column
    m = len(file_IDs) * 2
    PVM = np.zeros((m, 1))

    # indices for selecting pairs in images and PVM, including pair 49-1
    idx = [i for i in range(len(file_IDs))]
    # idx.append(idx[0])

    for id in range(len(idx) - 1):
        id1 = idx[id]
        id2 = idx[id + 1]

        i = id1 * 2
        j = id2 * 2

        f1 = file_IDs[id1]
        f2 = file_IDs[id2]
        p1, p2 = get_matches(f1, f2)

        for p1_i, p2_j in zip(p1, p2):

            new_point = True
            x1, y1 = p1_i
            x2, y2 = p2_j

            # if x coordinate of p1 in row i
            match_x1 = np.nonzero(PVM[i] == x1)[0]
            if len(match_x1) > 0:

                # if y coordinate of p1 follows x coordinate
                if PVM[i + 1, match_x1[0]] == y1:
                    PVM[j:j+2, match_x1[0]] = p2_j
                    new_point = False

            match_x2 = np.nonzero(PVM[j] == x2)[0]
            if len(match_x2) > 0:
                if PVM[j + 1, match_x2[0]] == y2:
                    PVM[i:i+2, match_x2[0]] = p2_j
                    new_point = False

            if new_point:
                new_point_col = np.zeros((m, 1))
                new_point_col[i: i + 2] = np.expand_dims(p1_i, 0).T
                new_point_col[j: j + 2] = np.expand_dims(p2_j, 0).T
                PVM = np.hstack((PVM, new_point_col))

    # remove placeholder
    PVM = np.delete(PVM, 0, axis=1)
    return PVM

def dense_repr(matrix, ratio=2):
    cond = np.sum(matrix != 0, axis=0)
    th = np.mean(cond) * ratio
    dense = matrix[:, cond > th]
    return dense

if __name__ == '__main__':
    # pvm = np.loadtxt("PointViewMatrix.txt")
    pvm = chaining("Data/House/House")
    pvm = dense_repr(pvm, ratio=2)
    pvm = np.around(pvm, 2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spy(pvm, markersize=1, color='black')
    ax.set_aspect('auto')
    ax.plot()
    ax.set_xlabel("Surface points")
    ax.set_ylabel("Views (x row & y row)")
    plt.show()
    # np.savetxt("pvm6.txt", pvm, fmt='%2g')
