import os
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from scipy.ndimage import rotate
from PIL import Image, ImageEnhance
from torchvision import utils
import matplotlib.pyplot as plt
import cv2
import skimage.morphology

def get_pr_auc(gt, pred):
    precision, recall, threshold = precision_recall_curve(gt, pred)
    pr_auc = auc(recall, precision)
    return precision, recall, threshold, pr_auc
def get_roc_auc(gt, pred):
    fpr, tpr, _ = roc_curve(gt, pred, pos_label=1.)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def prep_im_for_blob(im, label_rg, label_mask, is_training, db, crop_size, augmentation_opt=None):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    label_rg = label_rg.astype(np.float32, copy=False)
    label_mask = label_mask.astype(np.float32, copy=False)

    if is_training:
        im = random_perturbation(im)
        if augmentation_opt['USE_FLIPPED'] and np.random.random_sample() >= 0.5:
            im = np.flip(im,1).copy()
            label_rg = np.flip(label_rg, 1).copy()
            label_mask = np.flip(label_mask, 1).copy()

        if augmentation_opt['USE_BRIGHTNESS_ADJUSTMENT']:
            rand_btightness = np.random.uniform(-augmentation_opt['BRIGHTNESS_ADJUSTMENT_MAX_DELTA'],
                                    augmentation_opt['BRIGHTNESS_ADJUSTMENT_MAX_DELTA'])

            im += rand_btightness*255.
            im = np.clip(im, 0, 255)

        if augmentation_opt['USE_CONTRAST_ADJUSTMENT']:
            mmR = np.mean(im[:,:,0])
            mmG = np.mean(im[:,:,1])
            mmB = np.mean(im[:,:,2])
            rand_contrast = np.random.uniform(augmentation_opt['CONTRAST_ADJUSTMENT_LOWER_FACTOR'],
                                               augmentation_opt['CONTRAST_ADJUSTMENT_UPPER_FACTOR'],1)
            im[:,:,0] = (im[:,:,0] - mmR) * rand_contrast[0] + mmR
            im[:,:,1] = (im[:,:,1] - mmG) * rand_contrast[0] + mmG
            im[:,:,2] = (im[:,:,2] - mmB) * rand_contrast[0] + mmB
            im = np.clip(im, 0, 255)

    mean = [np.mean(im[label_mask[:,:,0]==1, 0]), np.mean(im[label_mask[:,:,0]==1, 1]), np.mean(im[label_mask[:,:,0]==1, 2])]
    std = [np.std(im[label_mask[:,:,0]==1, 0]), np.std(im[label_mask[:,:,0]==1, 1]), np.std(im[label_mask[:,:,0]==1, 2])]

    if std[0]==0:
        std[0] = 1
    if std[1]==0:
        std[1] = 1
    if std[2]==0:
        std[2] = 1

    if is_training:
        if augmentation_opt['USE_ROTATION']:
            angle = np.random.randint(0, augmentation_opt['ROTATION_RANGE'])*augmentation_opt['ROTATION_UNIT_ANGLE']
            im = np.array(Image.fromarray(im.astype(np.ubyte)).rotate(angle), dtype=np.float32)
            label_rg = np.array(Image.fromarray(label_rg[:,:,0].astype(np.ubyte)).rotate(angle), dtype=np.float32)
            label_rg = label_rg[:, :, np.newaxis]
        # label_mask = np.array(Image.fromarray(label_mask[:,:,0].astype(np.ubyte)).rotate(angle, fillcolor=0), dtype=np.float32)
        if (db == 'HRF' or db == 'CHASE') and augmentation_opt['USE_CROPPING']:
            h, w = im.shape[0], im.shape[1]
            crop_size = [h // 2, w // 2]
            LT = [np.random.randint(0, h - crop_size[0]), np.random.randint(0, w - crop_size[1])]
            im = im[LT[0]:LT[0] + crop_size[0], LT[1]:LT[1] + crop_size[1]]
            label_rg = label_rg[LT[0]:LT[0] + crop_size[0], LT[1]:LT[1] + crop_size[1]]

    im[:, :, 0] = (im[:, :, 0] - mean[0]) / std[0]
    im[:, :, 1] = (im[:, :, 1] - mean[1]) / std[1]
    im[:, :, 2] = (im[:, :, 2] - mean[2]) / std[2]

    return im, label_rg, label_mask

def random_perturbation(imgs):
    for i in range(imgs.shape[0]):
        im=Image.fromarray(imgs[i,...].astype(np.uint8))
        en=ImageEnhance.Color(im)
        im=en.enhance(np.random.uniform(0.8,1.2))
        imgs[i,...]= np.asarray(im).astype(np.float32)
    return imgs
    
def load_tr_img(img_set, bs, db, crop_size, augmentation_opt):
    randp = np.random.permutation(np.arange(img_set.__len__()))
    data = torch.FloatTensor()
    target = torch.FloatTensor()

    for i in range(bs):
        im = img_set[randp[i]][0].copy()
        im = im.reshape((im.shape[0], im.shape[1], 3))

        label = img_set[randp[i]][1].copy()
        label = label.reshape((label.shape[0], label.shape[1], 1))

        label_mask = img_set[randp[i]][2].copy()
        label_mask = label_mask.reshape((label_mask.shape[0], label_mask.shape[1], 1))

        im, label, _ = prep_im_for_blob(im, label, label_mask, True, db, crop_size, augmentation_opt)

        im = torch.from_numpy(im).float()
        im = im.clone().view(1, im.size(0), im.size(1), im.size(2))
        im = im.permute(0, 3, 1, 2)
        data = torch.cat([data, im], 0)

        label = torch.from_numpy(label)
        label = label.clone().view(1, label.size(0), label.size(1), label.size(2))
        label = label.permute(0, 3, 1, 2)
        target = torch.cat([target, label], 0)
    if torch.isinf(data).all() or torch.isnan(data).all():
        a = 0
        
    return data, target

def load_te_img(img_set, i, db, crop_size):
    data = torch.FloatTensor()
    target = torch.FloatTensor()
    target_mask = torch.FloatTensor()

    im = img_set[i][0].copy()

    label = img_set[i][1].copy()
    label = label.reshape([label.shape[0], label.shape[1], 1])

    mask = img_set[i][2].copy()
    mask = mask.reshape([mask.shape[0], mask.shape[1], 1])

    im, label, mask = prep_im_for_blob(im, label, mask, False, db, crop_size)

    im = torch.from_numpy(im).float()
    im = im.view(1, im.size(0), im.size(1), im.size(2))
    im = im.permute(0, 3, 1, 2)
    data = torch.cat([data, im], 0)

    label = torch.from_numpy(label)
    label = label.view(1, label.size(0), label.size(1), label.size(2))
    label = label.permute(0, 3, 1, 2)
    target = torch.cat([target, label], 0)

    mask = torch.from_numpy(mask)
    mask = mask.view(1, mask.size(0), mask.size(1), mask.size(2))
    mask = mask.permute(0, 3, 1, 2)
    target_mask = torch.cat([target_mask, mask], 0)

    return data, target, target_mask


def read_data(path_list):
    data = []
    for i in range(len(path_list[0])):
        img = Image.open(path_list[0][i])
        img_origin = np.array(img).astype(np.float32)
        # img = img.resize([img.width, img.height])
        img = np.array(img).astype(np.float32)
        label = Image.open(path_list[1][i])
        label_origin = np.array(label, dtype=np.ubyte).astype(np.float32)
        # label = label.resize([label.width, label.height])
        label = np.array(label, dtype=np.ubyte).astype(np.float32)
        if label.max() > 1:
            label /=255.
            label_origin /=255.
        label = np.round(label)
        label_origin = np.round(label_origin)
        mask = Image.open(path_list[2][i])
        mask_origin = (np.array(mask, dtype=np.ubyte) != 0).astype(np.float32)
        # mask = mask.resize([mask.width, mask.height])
        # mask = (np.asarray(mask, dtype=np.ubyte)).astype(np.float32)
        mask = (np.array(mask, dtype=np.ubyte) != 0).astype(np.float32)
        data.append([img, label, mask, img_origin, label_origin, mask_origin])
    return data
def get_mask(img, path_list, idx):
    mask = np.bitwise_or(img[:, :, 0] > 15, img[:, :, 1] > 15).astype(np.ubyte)  ## CHASE DB & HRF DB threshol
    mask = skimage.morphology.opening(mask)
    mask = skimage.morphology.closing(mask)
    cc = cv2.connectedComponents(mask, connectivity=4)
    max_cc_idx = 0
    max_cc_idx_sum = 0
    for j in range(cc[0]):
        cur_cc_img = cc[1] == j
        if cur_cc_img.sum() > max_cc_idx_sum:
            max_cc_idx_sum = cur_cc_img.sum()
            max_cc_idx = j
    mask = cc[1] == max_cc_idx

    mask_save_path = path_list[0][idx][:-(len(path_list[3][idx]) + 7)] + 'mask/' + path_list[3][idx][:-4] + '.png'
    Image.fromarray(mask.astype(np.ubyte) * 255).save(mask_save_path)

    return mask
def save_image(img, save_path):
    utils.save_image(img, save_path)

def search_in_dir(path):
    train_label = sorted(os.listdir(path + 'train/1st_manual/'))
    train_img = sorted(os.listdir(path + 'train/images/'))
    train_mask = sorted(os.listdir(path + 'train/mask/'))
    test_label = sorted(os.listdir(path + 'test/1st_manual/'))
    test_img = sorted(os.listdir(path + 'test/images/'))
    test_mask = sorted(os.listdir(path + 'test/mask/'))

    train_img_name = train_img[:]
    test_label_name = test_img[:]

    for i in range(len(train_img)):
        train_img[i] = path + 'train/images/'+train_img[i]
        train_label[i] = path + 'train/1st_manual/' + train_label[i]
        train_mask[i] = path + 'train/mask/' + train_mask[i]

    for i in range(len(test_img)):
        test_img[i] = path + 'test/images/'+test_img[i]
        test_label[i] = path + 'test/1st_manual/' + test_label[i]
        test_mask[i] = path + 'test/mask/' + test_mask[i]

    return [train_img, train_label, train_mask, train_img_name], [test_img, test_label, test_mask, test_label_name]