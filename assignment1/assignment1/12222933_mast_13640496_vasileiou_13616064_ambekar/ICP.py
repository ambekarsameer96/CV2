import sampling_matching_methods as sm
from audioop import add
import numpy as np
import utils

def calc_R_t(source, target):
    """
    compute R and t for a given translated source and target point cloud.
    """
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = (source - source_mean).T
    target_centered = (target - target_mean).T

    cov = source_centered @ target_centered.T
    U, S, VT = np.linalg.svd(cov)
    mid = np.eye(3)
    mid[-1][-1] = np.linalg.det(VT.T @ U.T)

    R = VT.T @ mid @ U.T
    t = target_mean - (R @ source_mean)
    return R, t

def rototranslation(Rt, R, t):
    """
    update rototranslation matrix Rt with R and t.
    """
    Rt_add = np.eye(4)
    Rt_add[:3, :3] = R
    Rt_add[:3, 3] = t
    return Rt @ Rt_add

def compute_RMS(source, target):
    # match points from target to points from source
    target_BF, source_sampled_intersection = sm.kd_method(source_sampled, target)

    # compute RMS
    RMS = np.sqrt(np.mean(np.linalg.norm(target_BF - source_sampled_intersection, axis=1)))
    return RMS

def compute_translation(points, Rt):
    """
    Compute translation of pointcloud using rototranslation matrix.
    """
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    points = (R @ points.T).T + t.T
    return points


def ICP(source, target, th=0.001,
        matching_fn=sm.min_dist, sampling_fn=None, iterative_sampling_fn=None,
        sampling_fn_ratio=4, iterative_sampling_fn_ratio=4):
    """
    Perform ICP algorithm and return rotation matrix and translation vector.
    """
    if sampling_fn is None:
        sampling_fn = sm.no_sampling

    R = np.eye(3)
    t = np.zeros(3)
    Rt = np.eye(4)

    RMS_li = []
    RMS_prev = 100
    iterations = 500

    source_sampled = sampling_fn(source, ratio=sampling_fn_ratio)
    target = sampling_fn(target, ratio=sampling_fn_ratio)

    for _ in range(iterations):
        if iterative_sampling_fn is not None:
            source_sampled = iterative_sampling_fn(source, RMS_li=RMS_li, ratio=iterative_sampling_fn_ratio)

        # match points from target to points from source
        target_BF, source_sampled_intersection = matching_fn(source_sampled, target)

        # compute RMS
        RMS = np.sqrt(np.mean(np.linalg.norm(target_BF - source_sampled_intersection, axis=1)))
        RMS_li.append(RMS)

        # convergence condition
        if np.abs(RMS - RMS_prev) < 1e-5:
            break

        # compute R and T and update rototranslation matrix
        R, t = calc_R_t(source_sampled_intersection, target_BF)
        Rt = rototranslation(Rt, R, t)

        # rotate and translate source pointcloud
        source_sampled = (R @ source_sampled.T).T + t.T

        # unsampled source needs to be translated for iterative sampling
        if iterative_sampling_fn is not None:
            source = (R @ source.T).T + t.T

        RMS_prev = RMS

    print(f"Mean RMS: {np.mean(RMS_li)}\nFinal RMS: {RMS}\n")
    return Rt, RMS_li


def run_ICP(source, target, args=dict(), o3dvis=False, matplotvis=False, title=None):
    if title:
        print(title)

    Rt, RMS_li = ICP(source, target, **args)
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    source_tr = compute_translation(source, Rt)

    if o3dvis:
        final = np.append(target, source_tr, axis=0)
        utils.vis_open3d(final)

    if matplotvis:
        utils.vis_matplotlib(source_tr, target, title)

    return RMS_li
