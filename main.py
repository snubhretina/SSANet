from __future__ import print_function

import argparse
import os
import torch
from torch import nn, optim
import torch.nn.functional as F 
from train import tr, ev
import PIL.Image as Image
from torchvision import models
from build_model import get_model
import utils
import numpy as np
import matplotlib.pyplot as plt
import csv
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='SSANet for vessel segmentation')
parser.add_argument('--batch', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--iter', type=int, default=10000, metavar='N',
                    help='number of itertation to train (default: 10000)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-interval', type=int, default=200, metavar='N')
parser.add_argument('--DB', type=int, default=0, metavar='N', help='0=DRIVE, 1=STARE, 2=CHASE, 3=HRF')
parser.add_argument('--arc', type=int, default=0, metavar='N', help='0=SSA3_ResNet34, 1=SSA3_VGG16, 2=DRIU')
parser.add_argument('--is_eval', type=bool, default=False, metavar='B', help='evaluation mode')
parser.add_argument('--model_weight', type=str, default='', help='load model weight')
parser.add_argument('--tensorboard_dir', type=str, default='runs', help='tensorboard event diretory')
parser.add_argument('--cropping', type=bool, default=False, help='cropping')

# Use imagea flip 
USE_FLIPPED = True

# Use brightness adjusted images during training
USE_BRIGHTNESS_ADJUSTMENT = True
BRIGHTNESS_ADJUSTMENT_MAX_DELTA = 0.3

# Use contrast adjusted images during training
USE_CONTRAST_ADJUSTMENT = True
CONTRAST_ADJUSTMENT_LOWER_FACTOR = 0.2
CONTRAST_ADJUSTMENT_UPPER_FACTOR = 1.8

# Use rotation images during training
USE_ROTATION = True
ROTATION_MAX_ANGLE = 360
ROTATION_UNIT_ANGLE = 3
ROTATION_RANGE = ROTATION_MAX_ANGLE / ROTATION_UNIT_ANGLE

# defined network architecture list
architecture_list= [
            'SSA3_ResNet34', # SSANet3 with ResNet34 backbone
            'SSA3_VGG16' # SSANet with VGG16 (DRIU style SSANet)
            'DRIU' # DRIU with VGG16
            ]

class training():
    def __init__(self, args):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        ## configuration ##
        self.max_iter = args.iter # maximum iterations
        self.batch_size = args.batch # batch size
        self.log_interval = args.log_interval # print interval for training log
        self.log_test = args.test_interval # test interval
        self.model_weight = args.model_weight # model weight
        self.augmentation_opt = {
            'USE_FLIPPED':USE_FLIPPED,
            'USE_BRIGHTNESS_ADJUSTMENT':USE_BRIGHTNESS_ADJUSTMENT,
            'BRIGHTNESS_ADJUSTMENT_MAX_DELTA':BRIGHTNESS_ADJUSTMENT_MAX_DELTA,
            'USE_CONTRAST_ADJUSTMENT':USE_CONTRAST_ADJUSTMENT,
            'CONTRAST_ADJUSTMENT_LOWER_FACTOR':CONTRAST_ADJUSTMENT_LOWER_FACTOR,
            'CONTRAST_ADJUSTMENT_UPPER_FACTOR':CONTRAST_ADJUSTMENT_UPPER_FACTOR,
            'USE_ROTATION':USE_ROTATION,
            'ROTATION_MAX_ANGLE':ROTATION_MAX_ANGLE,
            'ROTATION_UNIT_ANGLE':ROTATION_UNIT_ANGLE,
            'ROTATION_RANGE':ROTATION_RANGE,
            'USE_CROPPING': args.cropping,
            }

        # db : database name
        # lr_step : learning rate decay step. gamma(=decay rate) is 0.1
        # DB_path : prefixed DB directory path
        if args.DB == 0:
            self.DB_name = 'DRIVE'
            self.lr_step = 3000
            self.DB_path = './DB/DRIVE/'
            self.upsmaple_rate = 1
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
        elif args.DB == 1:
            self.DB_name = 'STARE'
            self.lr_step = 1500
            self.DB_path = './DB/STARE/'
            self.upsmaple_rate = 1
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
        elif args.DB == 2:
            self.DB_name = 'CHASE'
            self.lr_step = 3000
            self.DB_path = './DB/CHASEDB/'
            self.upsmaple_rate = 2
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
        elif args.DB == 3:
            self.DB_name = 'HRF'
            self.lr_step = 3000
            self.DB_path = './DB/HRF/'
            self.upsmaple_rate = 2
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
        
        # all DB load and read images
        [self.train_file_name_list,
        self.test_file_name_list,
        self.train_set, 
        self.test_set]  = self.all_DB_load_and_read(self.DB_path)

        self.n_test_img = len(self.test_set) # get test images size
        self.crop_size = [600, 700] # crop size, if crop option is turn on

        self.visualization_path = '/res/' # for testing time visualization
        self.best_model_save_dir_path = '/best/' # for model weight save directory

        # get model
        self.architecture_name = architecture_list[args.arc]
        self.model = get_model(self.architecture_name, self.sel_gpu, self.upsmaple_rate) 
        if self.model_weight != '':
            self.load_model_weight()

        # set loss function
        self.criterion = nn.BCELoss(size_average=False).cuda(0)
        # set optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        # set training scheduler for learning late decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step, gamma=0.1)
        
        self.trainer = tr(self.augmentation_opt)

        self.make_res_dir()
        self.train_t = 0
        self.train_loss_list = []

    def all_DB_load_and_read(self, DB_path):
        ## load DB
        # load DB path
        train_file_name_list, test_file_name_list = utils.search_in_dir(DB_path)
        # read images
        train_set = utils.read_data(train_file_name_list)
        test_set = utils.read_data(test_file_name_list)
        
        return [train_file_name_list, test_file_name_list, train_set, test_set]

    def train_iter(self):
        #
        # md, img_set, optimizer, scheduler, db, crop_size, criterion
        cur_train_loss, cur_train_run_t = \
        self.trainer.train_iter(
            self.model, 
            self.train_set, 
            self.optimizer, 
            self.scheduler, 
            self.DB_name, 
            self.crop_size, 
            self.criterion
            )
        self.cur_train_loss = cur_train_loss
        self.train_t += cur_train_run_t
        self.train_loss_list.append(self.cur_train_loss)
    
    def test(self):
        #
        # md, img_set, save_dir, img_names, db, crop_size
        self.test_loss_list, self.test_t, self.score, self.all_pred = \
        self.trainer.test(
            self.model,
            self.test_set,
            self.cur_visualization_path, 
            self.test_file_name_list, 
            self.DB_name, 
            self.crop_size,
            self.criterion
        )
    def load_model_weight(self):
        self.model.load_state_dict(torch.load(self.model_weight), False)
        ## custom model weights load
        # for k, v in torch.load(self.model_weight).items():
        #     if k in self.model.state_dict():
        #         try:
        #             self.model.load_state_dict({k:v}, False)
        #         except:
        #             continue
        
    def make_res_dir(self):
        self.cur_setting_save_dir_path = '%s_%s'%(self.DB_name, self.architecture_name)
        if not os.path.exists(self.cur_setting_save_dir_path): os.mkdir(self.cur_setting_save_dir_path)
        self.cur_visualization_path = self.cur_setting_save_dir_path + self.visualization_path
        if not os.path.exists(self.cur_visualization_path): os.mkdir(self.cur_visualization_path)
        self.cur_best_model_save_dir_path = self.cur_setting_save_dir_path + self.best_model_save_dir_path
        if not os.path.exists(self.cur_best_model_save_dir_path): os.mkdir(self.cur_best_model_save_dir_path)

    def save_best_model(self, best_model, best_model_file_path, _test_scores):
        # save best model weights
        # would be remained best model wieghts only!!
        # automatly removed before model weights.
        if best_model < _test_scores["pr_auc"][-1]:
            best_model = _test_scores["pr_auc"][-1]

            save_path = self.cur_best_model_save_dir_path+'model_%d_iter_loss_%.4f.pth.tar' % (iter+1, losses["test"][-1])
            torch.save(self.model.state_dict(), save_path)

            self.cur_best_img_save_dir_path = self.cur_best_model_save_dir_path +'res/'
            if os.path.exists(self.cur_best_img_save_dir_path)==False:
                os.mkdir(self.cur_best_img_save_dir_path)
            for idx, img in enumerate(self.all_pred):
                save_img_path = self.cur_best_img_save_dir_path + self.test_file_name_list[3][idx][:-4] + '.png'
                img = Image.fromarray((img*255).astype(np.ubyte))
                img.save(save_img_path)

            if best_model_file_path != '':
                os.remove(best_model_file_path)
            best_model_file_path = save_path
        return best_model, best_model_file_path

    def print_train_log(self, iter):
        # print log every log interval step
        print ('Train iter: %d [%d/%d (%.0f)] Loss: %.4f, time per frame: %.4f, total time: %.4f(bs:%d)'\
            %(self.max_iter, 
            iter+1, 
            self.max_iter, 
            100. * (iter+1) / self.max_iter, 
            self.cur_train_loss, 
            self.train_t / float(self.log_interval) / float(self.batch_size),
            self.train_t,
            self.batch_size)
            )
        self.init_train_time()

    def print_test_log(self, losses, scores):
        print('---Test time---')
        print('current learning rate: %e' % (self.get_cur_lr()))
        print('Average loss: %.4f, time per frame: %.4f, total time: %.4f(%d)'%\
            (losses["test"][-1], 
            self.test_t / float(self.n_test_img), 
            self.test_t, self.n_test_img)
            )
        print('ROC: %.4f, PR: %.4f, F1: %.4f, ACC: %.4f, Se: %.4f, Sp: %.4f'%\
            (scores["roc_auc"][-1],
            scores["pr_auc"][-1],
            scores["f1"][-1],
            scores["acc"][-1],
            scores["se"][-1],
            scores["sp"][-1])
            )

    def init_train_time(self):
        self.train_t = 0
    def init_train_loss_list(self):
        self.train_loss_list = []
    def get_mean_train_loss(self):
        return np.mean(self.train_loss_list)
    def get_mean_test_loss(self):
        return np.mean(self.test_loss_list)
    def get_cur_lr(self):
        return self.optimizer.param_groups[0]['lr'] # current learning rate check
    def get_test_score(self):
        return self.score
    def get_train_t(self):
        return self.train_t
    def get_test_t(self):
        return self.test_t
    
def write_all_test_score_to_csv(csv_path, x_range, losses, scores):
    with open(csv_path, 'w') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(['iter', 'train_loss', 'test_loss', 'pr_auc', 'roc_auc', 'f1', 'acc', 'se', 'sp', 'thresh'])
        for i in range(len(x_range)):
            csv_file.writerow(
                [x_range[i], 
                losses["train"][i], 
                losses["test"][i],
                scores["pr_auc"][i], 
                scores["roc_auc"][i], 
                scores["f1"][i], 
                scores["acc"][i], 
                scores["se"][i], 
                scores["sp"][i],
                scores["threshold"][i], 
                ])

class evaluation():
    def __init__(self, args):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        ## configuration ##
        self.batch_size = args.batch # batch size
        self.model_weight = args.model_weight # model weight
        # db : database name
        # lr_step : learning rate decay step. gamma(=decay rate) is 0.1
        # DB_path : prefixed DB directory path
        if args.DB == 0:
            self.DB_name = 'DRIVE'
            self.lr_step = 3000
            self.DB_path = './DB/DRIVE/'
            self.upsmaple_rate = 1
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')]
        elif args.DB == 1:
            self.DB_name = 'STARE'
            self.lr_step = 1500
            self.DB_path = './DB/STARE/'
            self.upsmaple_rate = 1
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')]
        elif args.DB == 2:
            self.DB_name = 'CHASE'
            self.lr_step = 3000
            self.DB_path = './DB/CHASEDB/'
            self.upsmaple_rate = 2
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')]
        elif args.DB == 3:
            self.DB_name = 'HRF'
            self.lr_step = 3000
            self.DB_path = './DB/HRF/'
            self.upsmaple_rate = 2
            self.sel_gpu = [torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0'), torch.device('cuda:0')]
        
        # all DB load and read images
        [self.train_file_name_list,
        self.test_file_name_list,
        self.train_set, 
        self.test_set]  = self.all_DB_load_and_read(self.DB_path)

        self.n_test_img = len(self.test_set) # get test images size
        self.crop_size = [600, 700] # crop size, if crop option is turn on
        self.visualization_path = '/evaluation_res/' # for testing time visualization

        # get model
        self.architecture_name = architecture_list[args.arc]
        self.model = get_model(self.architecture_name, self.sel_gpu,  self.upsmaple_rate) 
        self.load_model_weight()
    
        self.evaluator = ev()
        self.make_res_dir()

    def load_model_weight(self):
        # self.model.load_state_dict(torch.load(self.model_weight))
        self.model.load_state_dict(torch.load(self.model_weight),False)

    def all_DB_load_and_read(self, DB_path):
        ## load DB
        # load DB path
        train_file_name_list, test_file_name_list = utils.search_in_dir(DB_path)
        # read images
        train_set = utils.read_data(train_file_name_list)
        test_set = utils.read_data(test_file_name_list)
        
        return [train_file_name_list, test_file_name_list, train_set, test_set]
    
    def exc_eval(self):
        #
        # md, img_set, save_dir, img_names, db, crop_size
        self.test_t, self.score, self.all_pred = \
        self.evaluator.evaluation(
            self.model,
            self.test_set,
            self.cur_visualization_path, 
            self.test_file_name_list, 
            self.DB_name, 
            self.crop_size
        )

    def make_res_dir(self):
        self.cur_setting_save_dir_path = '%s_%s'%(self.DB_name, self.architecture_name)
        if not os.path.exists(self.cur_setting_save_dir_path): os.mkdir(self.cur_setting_save_dir_path)
        self.cur_visualization_path = self.cur_setting_save_dir_path + self.visualization_path
        if not os.path.exists(self.cur_visualization_path): os.mkdir(self.cur_visualization_path)

    def print_test_log(self, scores):
        print('---Evaluation time---')
        print('time per frame: %.4f, total time: %.4f(%d)'%\
            (self.test_t / float(self.n_test_img), 
            self.test_t, self.n_test_img)
            )
        print('ROC: %.4f, PR: %.4f, F1: %.4f, ACC: %.4f, Se: %.4f, Sp: %.4f'%\
            (scores[1],
            scores[0],
            scores[2],
            scores[3],
            scores[5],
            scores[6])
            )

    def get_test_score(self):
        return self.score
    def get_test_t(self):
        return self.test_t
    def draw_roc_curve(self):
        a = 0
    def darw_pr_curve(self):
        a = 0

## main ##
if __name__ == "__main__":
    ## argement pars ##
    args = parser.parse_args()
    writer = SummaryWriter(args.tensorboard_dir)

    max_iter = args.iter
    log_interval = args.log_interval
    test_interval = args.test_interval
    is_eval = args.is_eval
    model_weight = args.model_weight

    if is_eval:
        _eval = evaluation(args)
        _eval.exc_eval()
        scores = _eval.get_test_score()
        _eval.print_test_log(scores)

    else:
        # build trainner
        _train = training(args)
        
        best_model = 0 # for best model value(ACU of precision-recall curve)
        best_model_file_path = '' # init recording best model file path
        losses = {"train":[], "test":[]} ## init losses list
        test_scores = {"all_score":[], "roc_auc":[], "pr_auc":[], "f1":[], "threshold":[], "acc":[], "se":[], "sp":[]} ## init test score list
        leaning_rate_history = [] # init learning rate list
        ## training loop ##
        for iter in range(max_iter):
            # cur_lr = optimizer.param_groups[0]['lr'] # current learning rate check

            # trining on current iter
            _train.train_iter()

            # print log every log interval step
            if (iter+1) % args.log_interval == 0:
                _train.print_train_log(iter)

            ## test time ##
            if (iter+1) % test_interval == 0:
                # test
                _train.test()

                # store mean train/test loss
                losses["train"].append(_train.get_mean_train_loss())
                losses["test"].append(_train.get_mean_test_loss())
                _train.init_train_loss_list() ## init 

                # store score 
                # pr_auc_score, roc_auc_score, best_f1, acc, best_f1_threshold, se, sp
                score = _train.get_test_score()
                test_scores["all_score"].append(score)
                test_scores["pr_auc"].append(score[0])
                test_scores["roc_auc"].append(score[1])
                test_scores["f1"].append(score[2])
                test_scores["acc"].append(score[3])
                test_scores["threshold"].append(score[4])
                test_scores["se"].append(score[5])
                test_scores["sp"].append(score[6])
                # score current learning rate
                leaning_rate_history.append(_train.get_cur_lr())

                # print test log
                _train.print_test_log(losses, test_scores)

                # # save best model weights
                # # would be remained best model wieghts only!!
                # # automatly removed before model weights.
                best_model, best_model_file_path = _train.save_best_model(best_model, best_model_file_path, test_scores)

                ## hand craft result plot
                cnt = len(losses["test"])
                x_range = range(test_interval, test_interval * (cnt + 1), test_interval)
                # lw = 1
                # # loss plot
                # fig = plt.figure(figsize=(10, 8))
                # plt.plot(x_range, np.array(losses["train"]), 'g', lw=lw, label='train loss')
                # plt.plot(x_range, np.array(losses["test"]), 'b', lw=lw, label='test loss')
                # plt.grid(True)
                # plt.title('train-test loss')
                # plt.legend(loc="upper right")
                # fig.savefig(_train.cur_setting_save_dir_path+'loss.png')
                # plt.close()
                # # precision-recall and roc curve AUC value plot
                # fig = plt.figure(figsize=(10, 8))
                # plt.plot(x_range, np.array(test_scores["pr_auc"]), 'g', lw=lw, label='pr auc')
                # plt.plot(x_range, np.array(test_scores["roc_auc"]), 'b', lw=lw, label='roc auc')
                # plt.grid(True)
                # plt.title('auc')
                # plt.legend(loc="lower right")
                # fig.savefig(_train.cur_setting_save_dir_path+'auc.png')
                # plt.close()

                # plot to tensorboard 
                writer.add_scalars('1_loss/train_loss', {"train_loss":losses["train"][-1]}, iter)
                writer.add_scalars('1_loss/test_loss', {"test_loss":losses["test"][-1]}, iter)
                writer.add_scalars('2_auc/pr_auc', {"pr_auc":test_scores["pr_auc"][-1]}, iter)
                writer.add_scalars('2_auc/roc_auc', {"roc_auc":test_scores["roc_auc"][-1]}, iter)
                writer.add_scalars('3_binary_score/acc', {"acc":test_scores["acc"][-1]}, iter)
                writer.add_scalars('3_binary_score/se', {"se":test_scores["se"][-1]}, iter)
                writer.add_scalars('3_binary_score/sp', {"sp":test_scores["sp"][-1]}, iter)
                writer.add_scalars('3_binary_score/f1', {"f1":test_scores["f1"][-1]}, iter)
                writer.add_scalars('4_threshold', {"threhsold":test_scores["threshold"][-1]}, iter)
                writer.add_scalars('5_learning rate', {"learning rate":leaning_rate_history[-1]}, iter)

                ## write all test results
                write_all_test_score_to_csv(_train.cur_setting_save_dir_path+'/log_test_result.csv', x_range, losses, test_scores)
