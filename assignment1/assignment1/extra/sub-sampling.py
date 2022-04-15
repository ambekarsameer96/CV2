from sklearn.neighbors import KDTree
#multi-resolution sub-sampling
#def multi_res_sub_sampling(source, target, R, t, epsilon):

# import numpy as np
# from sklearn.neighbors import KDTree
# np.random.seed(0)
# X = np.random.random((5, 2))  # 5 points in 2 dimensions
# tree = KDTree(X)
# nearest_dist, nearest_ind = tree.query(X, k=2)
# print(X)

#2.1d - multi resolution sub-sampling
def multi_res_sub_sampling(source, target, R, t, epsilon):
    #initialize R and t
    R, t = init_RT(source)
    #multiply R and t with source
    source_new = multiply_RT_source(source, R, t)
    #update R and t values using SVD
    R_new, t_new = SVD_update(source_new, target, R, t)
    #check for values of RMSE
    val = RMSE_checker(source_new, target)
    if (val) < 0.5:
        return R_new, t_new
    else:
        return R, t

#2.1 sub-sampling from informative regions.
def sub_sampler_informative_regions(source, target, R, t, epislon):
    #find informative regions
    informative_regions = find_informative_regions(nearest_dist, nearest_ind)
    #sub-sampling from informative regions
    source_new = sub_sampler_informative_regions(source, informative_regions)
    #update R and t values using SVD
    R_new, t_new = SVD_update(source_new, target, R, t)
    #check for values of RMSE
    val = RMSE_checker(source_new, target)
    if (val) < 0.5:
        return R_new, t_new
    else:
        return R, t


def find_informative_regions(nearest_dist, nearest_ind):
    #find informative regions
    informative_regions = []
    for i in range(len(nearest_dist)):
        if nearest_dist[i] < 0.01:
            informative_regions.append(nearest_ind[i])
    return informative_regions



#2.2 kd-tree
#obtain points from Source and Target, try using KD tree, pair with maching.
def kd_method():
    tree = KDTree(source)
    nearest_dist, nearest_ind = tree.query(target, k=1)
    nearest_dist = list(nearest_dist)
    nearest_ind = list(nearest_ind)
    return nearest_dist, nearest_ind
