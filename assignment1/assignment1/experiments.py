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
unif_RMS_li = {}
unif_args = {'sampling_fn': sm.uniform_sampling}
for ratio in [2, 4, 8, 16]:
    unif_args['sampling_fn_ratio'] = ratio
    unif_RMS_li[f"sampling_fn_ratio: {ratio}"] = ICP.run_ICP(source, target,
                args=unif_args, title=f'unif sampling {ratio}')

with open("unif_RMS_li.pl", 'wb') as f:
    pickle.dump(unif_RMS_li, f)

# Random sub-sampling
rand_RMS_li = {}
rand_args = {'iterative_sampling_fn': sm.uniform_sampling}
for ratio in [2, 4, 8, 16]:
    rand_args['iterative_sampling_fn_ratio'] = ratio
    rand_RMS_li[f"iterative_sampling_fn_ratio: {ratio}"] = ICP.run_ICP(source, target,
                args=rand_args, title=f'random sampling {ratio}')

with open("rand_RMS_li.pl", 'wb') as f:
    pickle.dump(rand_RMS_li, f)

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

RMSes = {
"RMS_li_all": RMS_li_all,
# "RMS_li_unif": RMS_li_unif,
# "RMS_li_rand": RMS_li_rand,
"RMS_li_multi": RMS_li_multi,
"RMS_li_info": RMS_li_info,
"RMS_li_Kd": RMS_li_Kd,
"RMS_li_zbuff": RMS_li_zbuff
}
with open("RMSes.pl", 'wb') as f:
    pickle.dump(RMSes, f)

# ############################
# #   Merge Scene            #
# ############################
# file_path = './Data/data/'
# file_list = file_picker(file_path)
#
# # 3.1
# #pick 2,4,10th files and send to main_ICP
# Rts = []
# steps = 2
# source = open_pointcloud_data(file_list[0])
#
# # Compute RT values for each pair of frames
# for i in range(steps, len(file_list), steps):
#     print(file_list[i])
#     target = open_pointcloud_data(file_list[i])
#     Rt, RMS = ICP(source, target, matching_fn=sm.kd_method, sampling_fn=sm.uniform_sampling)
#     Rts.append(Rt)
#     source = target
#
# # Construct final point cloud at the end
# final = open_pointcloud_data(file_list[0])
# for en, i in enumerate(range(steps, len(file_list), steps)):
#     print(f"{i=}")
#     target = open_pointcloud_data(file_list[i])
#     final = compute_translation(final, Rts[en])
#     final = np.vstack((final, target))
#
# final = np.delete(final, 0, axis=0)
# vis_open3d(final)
#
# # 3.2
# # pick 2,4,10th files and send to main_ICP
# # final = open_pointcloud_data(file_list[0])
# #
# # # Compute RT values for each pair of frames
# # for i in range(steps, len(file_list), steps):
# #     print(file_list[i])
# #     target = open_pointcloud_data(file_list[i])
# #     Rt, RMS = ICP(final, target, matching_fn=sm.kd_method, sampling_fn=sm.step_sampling)
# #     final = compute_translation(final, Rt)
# #     final = np.vstack((final, target))
# #
# # vis_open3d(final)
