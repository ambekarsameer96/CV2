import sampling_matching_methods as sm
import numpy as np
import pickle
import ICP
import utils

source, target = utils.open_wave_data()
# source, target = utils.open_bunny_data()

# Using all points
RMS_li_all = ICP.run_ICP(source, target, title='all')

# Uniform sub-sampling
# source_n = utils.add_noise(source, 0.5, 2000)
# target_n = utils.add_noise(target)

unif_RMS_li = {}
for matching in {sm.min_dist, sm.kd_method}:
    unif_args = {'sampling_fn': sm.uniform_sampling}
    unif_args['matching_fn'] = matching
    for ratio in [2, 4, 8, 16]:
        unif_args['sampling_fn_ratio'] = ratio
        unif_RMS_li[f"sampling_fn_ratio: {ratio}"] = ICP.run_ICP(source, target,
                    args=unif_args, title=f'unif sampling {ratio}')

    # with open("unif_RMS_li.pl", 'wb') as f:
    #     pickle.dump(unif_RMS_li, f)


# Random sub-sampling
rand_RMS_li = {}
rand_args = {'iterative_sampling_fn': sm.uniform_sampling}
for ratio in [2, 4, 8, 16]:
    rand_args['iterative_sampling_fn_ratio'] = ratio
    rand_RMS_li[f"iterative_sampling_fn_ratio: {ratio}"] = ICP.run_ICP(source, target,
                args=rand_args, title=f'random sampling {ratio}')

# with open("rand_RMS_li.pl", 'wb') as f:
#     pickle.dump(rand_RMS_li, f)


# Multi-resolution sub-sampling
multires_args = {'iterative_sampling_fn': sm.multires_sampling}
RMS_li_multi = ICP.run_ICP(source, target, args=multires_args, title='multi')

# Informative region sub-sampling
IR_args = {'sampling_fn': sm.density_sampling}
RMS_li_info = ICP.run_ICP(source, target, args=IR_args, title='informative')

# KDTree
KD_args = {'matching_fn': sm.kd_method}
RMS_li_Kd = ICP.run_ICP(source, target, args=KD_args, title='KDTree')

# z-buffer
ZB_args = {'matching_fn': sm.z_buffer}
RMS_li_zbuff = ICP.run_ICP(source, target, args=ZB_args, title='z-buffer')

# RMSes = {
# "RMS_li_all": RMS_li_all,
# # "RMS_li_unif": RMS_li_unif,
# # "RMS_li_rand": RMS_li_rand,
# "RMS_li_multi": RMS_li_multi,
# "RMS_li_info": RMS_li_info,
# "RMS_li_Kd": RMS_li_Kd,
# "RMS_li_zbuff": RMS_li_zbuff
# }
# with open("RMSes.pl", 'wb') as f:
#     pickle.dump(RMSes, f)

############################
#   Merge Scene            #
############################
file_path = './Data/data/'
file_list = utils.file_picker(file_path)

# 3.1
# pick 2,4,10th files and send to main_ICP
def exercise31(steps):
    Rts = []
    RMS_li = []
    source = utils.open_pointcloud_data(file_list[0])

    # Compute RT values for each pair of frames
    for i in range(steps, len(file_list), steps):
        print(file_list[i])
        target = utils.open_pointcloud_data(file_list[i])
        Rt, RMS = ICP.ICP(source, target, matching_fn=sm.kd_method, sampling_fn=sm.uniform_sampling)

        RMS_li.append(np.mean(RMS))
        Rts.append(Rt)
        source = target

    # Construct final point cloud at the end
    final = utils.open_pointcloud_data(file_list[0])
    for en, i in enumerate(range(steps, len(file_list), steps)):
        target = utils.open_pointcloud_data(file_list[i])
        final = ICP.compute_translation(final, Rts[en])
        final = np.vstack((final, target))

    final = np.delete(final, 0, axis=0)
    utils.vis_open3d(final)

    return RMS_li

RMS_31 = {}
for steps in [1,2,4,10]:
    RMS_31[f"step_size {steps}"] = exercise31(steps)

with open("RMS31.pl", 'wb') as f:
    pickle.dump(RMS_31, f)

# 3.2
# pick 2,4,10th files and send to main_ICP
# final = utils.open_pointcloud_data(file_list[0])
#
# # Compute RT values for each pair of frames
# for i in range(steps, len(file_list), steps):
#     print(file_list[i])
#     target = utils.open_pointcloud_data(file_list[i])
#     Rt, RMS = ICP.ICP(final, target, matching_fn=sm.kd_method, sampling_fn=sm.step_sampling)
#     final = ICP.compute_translation(final, Rt)
#     final = np.vstack((final, target))
#
# vis_open3d(final)
