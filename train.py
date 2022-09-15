import time
import torch

import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from evaluate import evaluate
from network.ftaformer_pytorch import FTAFormer
from network.ftaformer_pytorch2 import FTA_U_Net
from network.ftaformer_pytorch3 import FTAFormer_FUSE
from network.ftaformer_pytorch_iaff import FTAFormer_IAFF
from network.psa_pytorch import PSAnet
from network.psa_pytorch2 import Parallel_PSAnet

feature_dir = "/home/qx/Feature/cfp_48_128_22050/"

M = 3
sigma = 1
gaussian_lable = [np.exp(-0.5 * i * i / sigma) for i in range(-M, M + 1, 1)]


def Gaussianlabel(y1):
    for i in range(y1.shape[0]):
        for j in range(y1.shape[1]):
            if (y1[i][j][0]) == 0:
                index = np.where(y1[i][j] == 1)[0]
                if len(index) == 1:
                    for k in range(-M, min(M + 1, (321 - index[0])), 1):
                        y1[i][j][index + k] = gaussian_lable[k + M]

    return y1


class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def train(fp, gid):
    Net = FTAFormer()
    # Net = FTAFormer_IAFF(mode=1)

    if gid is not None:
        Net.cuda(device=gid)
    else:
        Net.cpu()
    Net.float()

    epoch_num = 30
    bs = 16
    learn_rate = 0.0001
    """
    Loading training data:
        training data shape should be x: (n, 3, freq_bins, time_frames) extract from audio by cfp_process
                                      y: (n, freq_bins+1, time_frames) from ground-truth
    """

    print('Loading training data ...')
    x = np.load(fp + '/x_train.npy')
    y = np.load(fp + '/y1_train.npy')
    # y2_train = np.load(fp + 'y2_train.npy')
    x = x.transpose(0, 3, 2, 1)
    # y = Gaussianlabel(y)
    y = y.transpose(0, 2, 1)
    # y2_train = y2_train.transpose(0, 2, 1)
    # reshape
    print(x.shape)
    print(y.shape)

    # """
    # Loading Validation data
    # """
    # print('Loading validation data ...')
    # x_valid_list = np.load(fp + '/x_valid.npy')
    # y_valid_list = np.load(fp + '/y1_valid.npy')
    # y2_valid = np.load(fp + 'y2_valid.npy')
    #
    # x_valid_list = x_valid_list.transpose(0, 3, 2, 1)
    # # y_valid_list = Gaussianlabel(y_valid_list)
    # y_valid_list = y_valid_list.transpose(0, 2, 1)
    # y2_valid = y2_valid.transpose(0, 2, 1)
    # # reshape
    # print(x_valid_list.shape)
    # print(y_valid_list.shape)

    x_train = torch.from_numpy(x.copy()).float()
    y_train = torch.from_numpy(y.copy()).float()
    # y2_train = torch.from_numpy(y2_train.copy()).float()
    # x_valid = torch.from_numpy(x_valid_list.copy()).float()
    # y_valid = torch.from_numpy(y_valid_list.copy()).float()
    # y2_valid = torch.from_numpy(y2_valid.copy()).float()
    traindata_set = Dataset(data_tensor=x_train, target_tensor=y_train)
    traindatadata_loader = Data.DataLoader(dataset=traindata_set, batch_size=bs, shuffle=True)

    # validdata_set = Dataset(data_tensor=x_valid, target_tensor=y_valid, target_tensor2=y2_valid)
    # validdatadata_loader = Data.DataLoader(dataset=validdata_set, batch_size=bs)
    #
    """
    Training
    """
    best_OA = 0
    best_epoch = 0
    best_loss = np.inf
    Loss1 = nn.BCELoss()
    Loss2 = nn.BCELoss()
    opt = optim.Adam(Net.parameters(), lr=learn_rate)

    for epoch in range(epoch_num):

        start_time = time.time()
        Net.train()
        train_loss = 0
        val_loss = 0

        for step, (batch_x, batch_y) in enumerate(tqdm(traindatadata_loader)):
            opt.zero_grad()
            if gid is not None:
                pred, y2 = Net(batch_x.cuda(device=gid))
                pred = pred[:, 0]
                # y2 = y2[:, 0]
                y1loss = Loss1(pred, batch_y.cuda(device=gid))
                # y2loss = Loss2(y2, batch_y2.cuda(device=gid))
                loss = y1loss
                loss.backward()
                opt.step()
                train_loss += loss.item()
            else:
                pred, _ = Net(batch_x)
                pred = pred[:, 0]
                # print(pred.shape)
                # print(batch_y.shape)
                loss = Loss1(pred, batch_y)
                loss.backward()
                opt.step()
                train_loss += loss.item()

        Net.eval()
        with torch.no_grad():
            #     for step, (batch_x, batch_y, batch_y2) in enumerate(tqdm(validdatadata_loader)):
            #         if gid is not None:
            #             pred, y2 = Net(batch_x.cuda(device=gid))
            #             pred = pred[:, 0]
            #             # y2 = y2[:, 0]
            #             y1loss = Loss1(pred, batch_y.cuda(device=gid))
            #             # y2loss = Loss2(y2, batch_y2.cuda(device=gid))
            #             loss = y1loss
            #             val_loss += loss.item()
            #         else:
            #             pred, _ = Net(batch_x)
            #             pred = pred[:, 0]
            #             loss = Loss1(pred, batch_y)
            #             val_loss += loss.item()
            #
            # print('=========================')
            # print('Epoch: ', epoch, ' | train_loss: %.8f ' % train_loss, 'val_loss: %.8f' % val_loss)
            # # print('Valid | VR: {:.2f}% VFA: {:.2f}% RPA: {:.2f}% RCA: {:.2f}% OA: {:.2f}%'.format(
            # #     avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))
            #
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     best_epoch = epoch
            #     torch.save(Net.state_dict(), './model/PSAnet_mode2')
            #
            if epoch - best_epoch == 5:
                learn_rate = learn_rate * 0.5
            avg_eval_arr = evaluate(Net, gid, 16)
            # save to model
            if avg_eval_arr[-1] > best_OA:
                best_OA = avg_eval_arr[-1]
                best_epoch = epoch
                torch.save(Net.state_dict(), './model/ftaformer_321_sa_5')

            print('VR {:.2f}% VFA {:.2f}% RPA {:.2f}% RCA {:.2f}% OA {:.2f}% BestOA {:.2f}%'.format(
                avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4], best_OA))
            print('Epoch: ', epoch, 'Best Epoch: ', best_epoch, 'learn_rate: %.8f' % learn_rate)

            print('Time: ', int(time.time() - start_time), '(s)')

            # print('Best Epoch: ', best_epoch, ' | Best valid loss: %.8f' % best_loss, ' | learn_rate: %.8f' % learn_rate)


id =0
with torch.cuda.device(id):
    train(feature_dir, gid=id)

#     if args.gpu_index is not None:
#         with torch.cuda.device(args.gpu_index):
#             train(args.filepath, args.model_type, args.gpu_index, args.output_dir, args.epoch_num, args.learn_rate,
#                   args.batch_size)
#
# def parser():
#     p = argparse.ArgumentParser()
#
#     p.add_argument('-fp', '--filepath',
#                    help='Path to input training data (h5py file) and validation data (pickle file) (default: %(default)s)',
#                    type=str, default='./data/')
#     p.add_argument('-t', '--model_type',
#                    help='Model type: vocal or melody (default: %(default)s)',
#                    type=str, default='vocal')
#     p.add_argument('-gpu', '--gpu_index',
#                    help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
#                    type=int, default=0)
#     p.add_argument('-o', '--output_dir',
#                    help='Path to output folder (default: %(default)s)',
#                    type=str, default='./train/model/')
#     p.add_argument('-ep', '--epoch_num',
#                    help='the number of epoch (default: %(default)s)',
#                    type=int, default=100)
#     p.add_argument('-lr', '--learn_rate',
#                    help='the number of learn rate (default: %(default)s)',
#                    type=float, default=0.0001)
#     p.add_argument('-bs', '--batch_size',
#                    help='The number of batch size (default: %(default)s)',
#                    type=int, default=50)
#     return p.parse_args()
#
#
# if __name__ == '__main__':
#
#     args = parser()
#     if args.gpu_index is not None:
#         with torch.cuda.device(args.gpu_index):
#             train(args.filepath, args.model_type, args.gpu_index, args.output_dir, args.epoch_num, args.learn_rate,
#                   args.batch_size)
#     else:
#         train(args.filepath, args.model_type, args.gpu_index, args.output_dir, args.epoch_num, args.learn_rate,
#               args.batch_size)
