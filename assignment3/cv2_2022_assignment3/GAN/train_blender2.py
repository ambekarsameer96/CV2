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

import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils


# Configurations
######################################################################
# Fill in your experiment names and the other required components dataset\data_set\test.str
experiment_name = 'Blender'
data_root = '../dataset/data_set/data/'
train_list = '../dataset/data_set/train.str'
test_list = '../dataset/data_set/test.str'
batch_size = 4
batches_train = 4
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
        
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

done = u'\u2713'
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
G, _, _ = utils.loadModels(G, '../dataset\Pretrained_model\checkpoint_G.pth')
D, _, _ = utils.loadModels(D, '../dataset\Pretrained_model\checkpoint_D.pth')

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
L_id = L_id.to(device)
# pixel loss
L_pi = nn.L1Loss()
L_di = gan_loss.GANLoss()
L_ga = gan_loss.GANLoss()
L_ga = L_ga.to(device)
L_di = L_di.to(device)
L_pi = L_pi.to(device)

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

def alpha_blending(source_img, target_img, mask):
    # Implement alpha blending here. You can use the built-in blend function in
    # opencv.
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    output = cv2.addWeighted(source_img, 0.5, target_img, 0.5, 0)
    return output

def laplacian_pyramid_blending(source_img, target_img, mask):
    # Implement laplacian pyramid blending here. You can use the built-in
    # pyramidBlending function in opencv.
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    output - cv2.pyrDown(source_img)
    output = cv2.pyrUp(output)
    #output = cv2.addWeighted(source_img, 0.5, target_img, 0.5, 0)
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


def Train(G, D, epoch_count, iter_count):
    G.train(True)
    D.train(True)
    L_ga.train(False)
    epoch_count += 1
    pbar = tqdm(enumerate(trainLoader), total=batches_train, leave=False)

    Epoch_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        print("1")
        t_hat = transfer_mask(target, source, mask)
        t = target
        m = mask
        print("2")
        x = torch.concat([t_hat, t, m], dim=1)
        print("3")
        y = blend_imgs(source, target, mask)
        print("4")
        x = x.to(device)
        y = y.to(device)
        print("5")
        # images = images.to(device)
        output_G = G(x)
        print("G")
        output_D_G = D(output_G)
        
        print(type(output_D_G))
        output_D_y = D(y)
        print("D")
        # Reconstruction loss
        L_rec = 0.5 * L_id(output_G, y) + L_pi(output_G, y)
        print("rec")
        # Discriminator loss
        #loss_D = (L_ga(output_D_y, True) + L_ga(output_D_G, False)) / 2
        loss_D_fake = L_ga(output_D_G, False)
        loss_D_real = L_ga(output_D_y, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        print("D")
        # Fake passability loss
        L_G = L_ga(output_D_G, True)
        print("LG")
        # Who knows loss
        #loss_G = -torch.log(1 - np.array(output_D_G)).mean()
        # tmean = torch.mean(torch.stack(output_D_G))
        # loss_D_fake = L_ga(output_D_G, False)
        # loss_D_real = L_ga(output_D_y, True)
        # loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_G = -torch.log(1 - torch.mean(loss_D))
        #loss_G = 0
        # print("Who")
        total_loss = rec_weight * L_rec + gan_weight * L_G
        print("th")
        
        
        
        print("fuck")
        
        loss_D.backward()
        
        
        total_loss.backward()
        optimizer_D.step()
        optimizer_D.zero_grad()
        optimizer_G.step()
        optimizer_G.zero_grad()
        
        
        print("knows")

    total_G_loss += loss_G.item()
    total_D_loss += loss_D.item()
    total_time += time.time() - Epoch_time
    print('[I] STATUS: Epoch %d/%d, Iter %d/%d, Loss_G: %.4f, Loss_D: %.4f, Time: %.4f' %
            (epoch_count, epochs, iter_count, batches_train, loss_G.item(), loss_D.item(), time.time() - Epoch_time))
    trainLogger.write('%d, %d, %d, %.4f, %.4f, %.4f\n' %
            (epoch_count, epochs, iter_count, loss_G.item(), loss_D.item(), time.time() - Epoch_time))

    if iter_count % displayIter == 0:
        # Write to the log file.

        trainLogger.write('%d, %d, %d, %.4f, %.4f, %.4f\n' %
        (epoch_count, epochs, iter_count, loss_G.item(), loss_D.item(), time.time() - Epoch_time))


    # Print out the losses here. Tqdm uses that to automatically print it
    # in front of the progress bar.
    pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator

    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    return np.nanmean(total_loss_pix),\
        np.nanmean(total_loss_id), np.nanmean(total_loss_attr),\
        np.nanmean(total_loss_rec), np.nanmean(total_loss_G_Gan),\
        np.nanmean(total_loss_D_Gan), iter_count


def Test(G):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, (images) = next(t)

        # Feed the network with images from test set

        # Blend images
        pred = G(img_transfer_input)
        # You want to return 4 components:
            # 1) The source face.
            # 2) The 3D reconsturction.
            # 3) The target face.
            # 4) The prediction from the generator.
            # 5) The GT Blend that the network is targettting.


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
