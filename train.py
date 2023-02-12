from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import csv
import utils

class tr():
    def __init__(self, augmentation_opt):
        self.augmentation_opt = augmentation_opt

    def train_iter(self, md, img_set, optimizer, scheduler, db, crop_size, criterion):
        md.train()
        scheduler.step()
        st = datetime.now()

        data, target_rg = utils.load_tr_img(img_set, 1, db, crop_size, self.augmentation_opt)

        data, target_rg = Variable(data.cuda(0)), Variable(target_rg.cuda(0))

        optimizer.zero_grad()
        output_rg = md(data)

        loss = criterion(output_rg, target_rg)
        lossv = loss.item()
        loss.backward()

        optimizer.step()

        et = datetime.now()
        run_t = (et-st).total_seconds()

        return lossv, run_t

    def test(self, md, img_set, save_dir, img_names, db, crop_size, criterion):
        md.eval()

        t = 0

        all_pred = []
        all_gt = []
        all_mask = []
        test_loss = []

        with torch.no_grad():
            for i in range(len(img_set)):
                st = datetime.now()

                data, target_rg, target_mask = utils.load_te_img(img_set, i, db, crop_size)
                if db=='HRF':
                    data, target_rg = Variable(data.cuda(0)), Variable(target_rg.cuda(0))
                    output_rg = torch.FloatTensor(target_rg.size()).cuda(0)
                    output_rg[:, :, :, : data.size(3) // 2 + 25] = md(data[:, :, :, : data.size(3) // 2 + 50])[:,:,:,:-25]
                    output_rg[:, :, :, data.size(3) // 2 - 25 :] = md(data[:, :, :, data.size(3) // 2 - 50:])[:,:,:,25:]
                    output_rg = Variable(output_rg)
                    loss = criterion(output_rg, target_rg)
                    test_loss.append(loss.item())

                    et = datetime.now()
                    t += (et - st).total_seconds()

                    pred = output_rg.data.cpu()

                    save_path = save_dir + img_names[3][i][:-4] + '.png'
                    utils.save_image(pred[0][0], save_path)

                    all_pred.append(pred[0][0].numpy())
                    all_gt.append(target_rg[0][0].data.cpu().numpy())
                    all_mask.append(target_mask[0][0].numpy())
                else:
                    data, target_rg = Variable(data.cuda()), Variable(target_rg.cuda())

                    output_rg = md(data)

                    loss = criterion(output_rg.view(-1), target_rg.view(-1))
                    test_loss.append(loss.item())

                    et = datetime.now()
                    t += (et - st).total_seconds()

                    pred = output_rg.data.cpu()

                    save_path = save_dir+img_names[3][i][:-4]+'.jpg'
                    utils.save_image(pred[0][0], save_path)

                    all_pred.append(pred[0][0].numpy())
                    all_gt.append(target_rg[0][0].data.cpu().numpy())
                    all_mask.append(target_mask[0][0].numpy())

        arr_all_pred = np.array(all_pred).astype(np.float32)
        # arr_all_pred *= 255.
        # arr_all_pred = np.around(arr_all_pred)/255.
        arr_all_gt = np.array(all_gt).astype(np.float32)
        arr_all_mask = np.array(all_mask).astype(np.float32)

        flat_pred = arr_all_pred[arr_all_mask!=0].flatten()
        flat_gt = arr_all_gt[arr_all_mask!=0].flatten()

        precision, recall, threshold, pr_auc_score = utils.get_pr_auc(flat_gt, flat_pred)
        _, _, roc_auc_score = utils.get_roc_auc(flat_gt, flat_pred)

        all_f1 = 2. * precision * recall / (precision + recall)
        best_f1 = np.nanmax(all_f1)
        index = np.nanargmax(all_f1)
        best_f1_threshold = threshold[index]
        binary_flat = (flat_pred >= best_f1_threshold).astype(np.float32)
        acc = (flat_gt == binary_flat).sum() / float(flat_gt.shape[0])

        tp = np.bitwise_and((flat_gt == 1).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
        tn = np.bitwise_and((flat_gt == 0).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
        fp = np.bitwise_and((flat_gt == 0).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
        fn = np.bitwise_and((flat_gt == 1).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
        se = tp / float(tp + fn)
        sp = tn / float(fp + tn)

        score = [pr_auc_score, roc_auc_score, best_f1, acc, best_f1_threshold, se, sp]
        # score = [0, 0, 0, 0, 0, 0, 0]

        return test_loss, t, score, all_pred

class ev():
    def evaluation(self, md, img_set, save_dir, img_names, db, crop_size):
        md.eval()

        t = 0

        all_pred = []
        all_gt = []
        all_mask = []

        with torch.no_grad():
            for i in range(len(img_set)):
                st = datetime.now()

                data, target_rg, target_mask = utils.load_te_img(img_set, i, db, crop_size)
                if db=='HRF':
                    data, target_rg = Variable(data.cuda(0)), Variable(target_rg.cuda(0))
                    output_rg = torch.FloatTensor(target_rg.size()).cuda(0)
                    output_rg[:, :, :, : data.size(3) // 2 + 25] = md(data[:, :, :, : data.size(3) // 2 + 50])[:, :, :, :-25]
                    output_rg[:, :, :, data.size(3) // 2 - 25:] = md(data[:, :, :, data.size(3) // 2 - 50:])[:, :, :, 25:]
                    output_rg = Variable(output_rg)

                    et = datetime.now()
                    t += (et - st).total_seconds()

                    pred = output_rg.data.cpu()
                    mask = target_mask[0][0].numpy()
                    pred[0][0][mask==0] = 0

                    save_path = save_dir + img_names[3][i][:-4] + '.png'
                    utils.save_image(pred[0][0], save_path)

                    all_pred.append(pred[0][0].numpy())
                    all_gt.append(target_rg[0][0].data.cpu().numpy())
                    all_mask.append(mask)
                else:
                    data, target_rg = Variable(data.cuda()), Variable(target_rg.cuda())
                    output_rg = md(data)

                    et = datetime.now()
                    t += (et - st).total_seconds()

                    pred = output_rg.data.cpu()
                    mask = target_mask[0][0].numpy()
                    pred[0][0][mask==0] = 0

                    save_path = save_dir+img_names[3][i][:-4]+'.png'
                    utils.save_image(pred[0][0], save_path)

                    all_pred.append(pred[0][0].numpy())
                    all_gt.append(target_rg[0][0].data.cpu().numpy())
                    all_mask.append(mask)

        arr_all_pred = np.array(all_pred).astype(np.float32)
        # arr_all_pred *= 255.
        # arr_all_pred = np.around(arr_all_pred)/255.
        arr_all_gt = np.array(all_gt).astype(np.float32)
        arr_all_mask = np.array(all_mask).astype(np.float32)

        # np.savetxt('eval_data_pred.txt',arr_all_pred.flatten())
        # np.savetxt('eval_data_gt.txt', arr_all_gt.flatten())
        # np.savetxt('eval_data_mask.txt', arr_all_mask.flatten())

        flat_pred = arr_all_pred[arr_all_mask!=0].flatten()
        flat_gt = arr_all_gt[arr_all_mask!=0].flatten()

        precision, recall, threshold, pr_auc_score = utils.get_pr_auc(flat_gt, flat_pred)
        fpr, tpr, roc_auc_score = utils.get_roc_auc(flat_gt, flat_pred)

        all_f1 = 2. * precision * recall / (precision + recall)
        best_f1 = np.max(all_f1)
        index = all_f1.argmax()
        best_f1_threshold = threshold[index]
        binary_flat = (flat_pred >= best_f1_threshold).astype(np.float32)
        acc = (flat_gt == binary_flat).sum() / float(flat_gt.shape[0])

        tp = np.bitwise_and((flat_gt == 1).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
        tn = np.bitwise_and((flat_gt == 0).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
        fp = np.bitwise_and((flat_gt == 0).astype(np.ubyte), (binary_flat == 1).astype(np.ubyte)).sum()
        fn = np.bitwise_and((flat_gt == 1).astype(np.ubyte), (binary_flat == 0).astype(np.ubyte)).sum()
        se = tp / float(tp + fn)
        sp = tn / float(fp + tn)

        score = [pr_auc_score, roc_auc_score, best_f1, acc, best_f1_threshold, se, sp]

        # f = open('eavl_data.csv', 'w')
        # csv_writer = csv.writer(f)
        # csv_writer.writerow(precision)
        # csv_writer.writerow(recall)
        # csv_writer.writerow([pr_auc_score])
        # csv_writer.writerow(fpr)
        # csv_writer.writerow(tpr)
        # csv_writer.writerow([roc_auc_score])
        # csv_writer.writerow([best_f1])
        # csv_writer.writerow([acc])
        # csv_writer.writerow([best_f1_threshold])
        # f.close()

        # all_pr_auc_list = []
        # all_roc_auc_list = []
        # all_f1_list = []

        # f = open('image_wise_eval.csv', 'w')
        # csv_writer = csv.writer(f)
        # for i in range(len(all_pred)):
        #     cur_pred = all_pred[i]
        #     cur_gt = all_gt[i]
        #     cur_mask = all_mask[i]

        #     flat_pred = cur_pred[cur_mask==1].flatten()
        #     flat_gt = cur_gt[cur_mask==1].flatten()

        #     precision, recall, threshold, pr_auc_score = utils.get_pr_auc(flat_gt, flat_pred)
        #     fpr, tpr, roc_auc_score = utils.get_roc_auc(flat_gt, flat_pred)

        #     all_f1 = 2. * precision * recall / (precision + recall)
        #     best_f1 = np.max(all_f1)
        #     index = all_f1.argmax()
        #     best_f1_threshold = threshold[index]
        #     binary_flat = (flat_pred >= best_f1_threshold).astype(np.float32)
        #     acc = (flat_gt == binary_flat).sum() / float(flat_gt.shape[0])
        #     all_pr_auc_list.append(pr_auc_score)
        #     all_roc_auc_list.append(roc_auc_score)
        #     all_f1_list.append(best_f1)

        #     csv_writer.writerow([i, pr_auc_score, roc_auc_score, best_f1])

        # avg_pr_auc = np.array(all_pr_auc_list).mean()
        # avg_roc_auc = np.array(all_roc_auc_list).mean()
        # avg_f1 = np.array(all_f1_list).mean()
        # std_pr_auc = np.array(all_pr_auc_list).std()
        # std_roc_auc = np.array(all_roc_auc_list).std()
        # std_f1 = np.array(all_f1_list).std()
        # csv_writer.writerow(['average', avg_pr_auc, avg_roc_auc, avg_f1])
        # csv_writer.writerow(['std', std_pr_auc, std_roc_auc, std_f1])
        # csv_writer.writerow(['total', avg_pr_auc, std_pr_auc,avg_roc_auc, std_roc_auc,avg_f1,std_f1,
        #                     np.array(all_pr_auc_list).max(), np.array(all_pr_auc_list).min(),
        #                     np.array(all_roc_auc_list).max(), np.array(all_roc_auc_list).min(),
        #                     np.array(all_f1_list).max(), np.array(all_f1_list).min()])

        # f.close()

        return t, score, all_pred