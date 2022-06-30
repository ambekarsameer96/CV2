import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils


# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender_Laplacian'
data_root = '../dataset/data_set/data/'
train_list = '../dataset/data_set/train.str'
test_list = '../dataset/data_set/test.str'
batch_size = 8
batches_train = 8
nthreads = 0
max_epochs = 20
displayIter = 20
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

# done = u'\u2713'
done = 'yea'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
# Define your generators and Discriminators here
G = res_unet.MultiScaleResUNet(in_nc=7,out_nc=3).to(device)
D = discriminators_pix2pix.MultiscaleDiscriminator().to(device)
print(done)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
    # use the .to(device) function, where device is automatically detected
    # above.
# G, _, _ = utils.loadModels(G, '../dataset\Pretrained_model\checkpoint_G.pth')
# D, _, _ = utils.loadModels(D, '../dataset\Pretrained_model\checkpoint_D.pth')
G, _, _ = utils.loadModels(G, '../dataset/Pretrained_model/checkpoint_G.pth')
D, _, _ = utils.loadModels(D, '../dataset/Pretrained_model/checkpoint_D.pth')

G.to(device)
D.to(device)
print(done)

print('[I] STATUS: Initiate optimizer...', end='')
# Define your optimizers and the schedulers and connect the networks from
# before
optimizer_G = torch.optim.SGD(G.parameters(), lr=lr_gen, momentum=momentum, weight_decay=weightDecay)
optimizer_D = torch.optim.SGD(D.parameters(), lr=lr_dis, momentum=momentum, weight_decay=weightDecay)

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
print(done)

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
# perceptual loss
L_id = vgg_loss.VGGLoss()

# pixel loss
L_pi = nn.L1Loss()
L_di = gan_loss.GANLoss()
L_ga = gan_loss.GANLoss()
L_id.to(device)
L_pi.to(device)
L_di.to(device)
L_ga.to(device)

print(done)

print('[I] STATUS: Initiate Dataloaders...')
# Initialize your datasets here
trainData = SwappedDatasetLoader(train_list, data_root)
trainLoader = DataLoader(trainData, batch_size=batch_size)

testData = SwappedDatasetLoader(test_list, data_root)
testLoader = DataLoader(testData, batch_size=batch_size)
# testLoader.to(device)
print(done)

print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
print(done)


def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1 - mask)

# Step-2
# Find the Gaussian pyramid of the two images and the mask
def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower)
    return gaussian_pyr

# Step-3
# Then calculate the Laplacian pyramid
def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr

# Step-4
# Now blend the two images wrt. the mask
def blend(laplacian_A,laplacian_B,mask_pyr):
    LS = []
    for la,lb,mask in zip(laplacian_A,laplacian_B,mask_pyr):
        ls = lb * mask + la * (1 - mask)
        ls = np.asarray(ls, np.float64)
        LS.append(ls)
    return LS

# Step-5
# Reconstruct the original image
def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def laplacian_pyramid_blending(source_img, target_img, mask):
    num_levels = 7

    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(source_img, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(target_img, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, num_levels)
    mask_pyr_final.reverse()

    for i, lg_mask in enumerate(mask_pyr_final):
        if len(lg_mask.shape) < 3:
            mask_pyr_final[i] = np.expand_dims(lg_mask, -1)

    # Blend the images
    add_laplace = blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)
    # Reconstruct the images
    final = reconstruct(add_laplace)
    # Save the final image to the disk
    return final[num_levels]

def laplacian_blnder(source_tensor, target_tensor, mask_tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        # source_img = source_tensor[b].clone().cpu().numpy()
        # target_img = target_tensor[b].clone().cpu().numpy()
        source_img = img_utils.tensor2rgb(source_tensor[b]) / 255
        target_img = img_utils.tensor2rgb(target_tensor[b]) / 255
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        out_bgr = laplacian_pyramid_blending(source_img, target_img, mask)
        out_bgr[np.where(out_bgr < 0)] = 0
        out_bgr = np.round(out_bgr * 255).astype('uint8')
        out_tensors.append(img_utils.rgb2tensor(out_bgr))
    return torch.cat(out_tensors, dim=0)

def blend_imgs_bgr(source_img, target_img, mask):
    # Implement poisson blending here. You can us the built-in seamlessclone
    # function in opencv which is an implementation of Poisson Blending.
    a = np.where(mask != 0)

    # no mask
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img

    # small mask
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return target_img

    center = (np.min(a[1]) + np.max(a[1])) // 2, (np.min(a[0]) + np.max(a[0])) // 2
    output = cv2.seamlessClone(source_img, target_img, mask*255, center, cv2.NORMAL_CLONE)
    return output


def blend_imgs(source_tensor, target_tensor, mask_tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)

def alpha_blending_cv(source_img, target_img, mask):
    # Implement alpha blending here. You can use the built-in blend function in
    # opencv.
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    output = cv2.addWeighted(source_img, 0.5, target_img, 0.5, 0)
    plt.imshow(output)
    plt.show()
    return output

def alpha_blending(source_tensor, target_tensor, mask_tensor):
    # Implement alpha blending with torch tensor images. You can use the built-in blend function in opencv.
    # source_img = source_img.cpu().numpy()
    # target_img = target_img.cpu().numpy()
    # mask = mask.cpu().numpy()
    # output = np.zeros(source_img.shape)
    # #use cv2 add weighted
    # for i in range(source_img.shape[0]):
    #     output[i] = cv2.addWeighted(source_img[i], 1 - mask[i], target_img[i], mask[i], 0)

    # return torch.from_numpy(output).to(device)
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        #out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_alpha = alpha_blending_cv(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_alpha))

    return torch.cat(out_tensors, dim=0)



def Train(G, D, epoch_count, iter_count):
    G.train(True)
    D.train(True)
    epoch_count += 1
    # total_time = 0
    pbar = tqdm(enumerate(trainLoader), total=batches_train, leave=False)

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        images, _ = data
        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        # 2) Feed the data to the networks.
        # 4) Calculate the losses.
        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.
        source, target, swap, mask = images.values()
        t_hat = transfer_mask(target, source, mask)
        t = target
        m = mask
        x = torch.concat([t_hat, t, m], dim=1)

        D.zero_grad()
        y = laplacian_blnder(source, target, mask)
        #use alpha blending now
        #y = alpha_blending(source, target, mask)
        #print('Alpha blending done')

        y = y.to(device)
        output_D_real = D(y)
        loss_D_real = L_ga(output_D_real, True)

        x = x.to(device)
        output_G = G(x)
        output_D_G = D(output_G.detach())
        loss_D_fake = L_ga(output_D_G, False)

        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        G.zero_grad()
        output_D_G = D(output_G)


        L_G = L_ga(output_D_G, True)
        L_idd  = 0.5 * L_id(output_G, y)
        L_rec = L_idd + pix_weight * L_pi(output_G, y)
        loss_G = rec_weight * L_rec + gan_weight * L_G
        loss_G.backward()
        optimizer_G.step()

    # total_G_loss += loss_G.item()
    # total_D_loss += loss_D.item()
    # total_time += time.time() - Epoch_time
    print('[I] STATUS: Epoch %d/%d, Iter %d/%d, Loss_G: %.4f, Loss_D: %.4f, Time: %.4f' %
            (epoch_count, max_epochs, iter_count, batches_train, loss_G.item(), loss_D.item(), time.time() - Epoch_time))
    trainLogger.write('%d, %d, %d, %.4f, %.4f, %.4f\n' %
            (epoch_count, max_epochs, iter_count, loss_G.item(), loss_D.item(), time.time() - Epoch_time))

    if iter_count % displayIter == 0:
        # Write to the log file.

        trainLogger.write('%d, %d, %d, %.4f, %.4f, %.4f\n' %
        (epoch_count, epochs, iter_count, loss_G.item(), loss_D.item(), time.time() - Epoch_time))


    # Print out the losses here. Tqdm uses that to automatically print it
    # in front of the progress bar.
    pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator
    utils.saveModels(G, optimizer_G, iter_count,
    checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
    checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)

    return 0
    # return np.nanmean(total_loss_pix),\
    #     np.nanmean(total_loss_id), np.nanmean(total_loss_attr),\
    #     np.nanmean(total_loss_rec), np.nanmean(total_loss_G_Gan),\
    #     np.nanmean(total_loss_D_Gan), iter_count

def Test(G):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, (images, _) = next(t)

        source, target, swap, mask = images.values()
        t_hat = transfer_mask(target, source, mask)

        img_transfer_input = torch.concat([t_hat, target, mask], dim=1)
        img_transfer_input = img_transfer_input.to(device)

        #blend = blend_imgs(source, target, mask)
        blend = laplacian_blnder(source, target, mask)
        pred = G(img_transfer_input)

        return source, swap, target, pred, blend

iter_count = 0
# Print out the experiment configurations. You can also save these to a file if
# you want them to be persistent.
print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tSaving frequency (per epoch): ', saveIter)
print('\tModels Dumped at: ', check_point_loc)
print('\tVisuals Dumped at: ', visuals_loc)
print('\tExperiment Name: ', experiment_name)

for i in range(max_epochs):
    # Call the Train function here
    # Step through the schedulers if using them.
    # You can also print out the losses of the network here to keep track of
    # epoch wise loss.
    losses = Train(G, D, i, iter_count)
    scheduler_G.step()
    scheduler_D.step()
trainLogger.close()
