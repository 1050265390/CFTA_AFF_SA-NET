# -*- coding: utf-8 -*-
import os

import mir_eval
import numpy as np

from feature_extract.cfp_feature_extraction import freq2ind, cfp_process
from feature_extract.constant import SAMPLE_RATE, HOP_SIZE

procedure = "train"
path = '/home/qx/Dataset/MIR1K/'
ref_dir = "/home/qx/Dataset/"
resdir = "/home/qx/Feature/cfp_mir1k_128_22050/"

SIZE = 128
f = open('./mir.txt')
file_lists = f.readlines()
file_lists = [''.join(i).strip('\n') for i in file_lists]


def train(resample=False):
    TOTAL = len(file_lists)
    u = 0
    namelist = []
    for filename in file_lists:
        if filename not in namelist:
            namelist.append(filename)
        else:
            continue
        X = np.empty((0, 128, 320, 3))
        Y1 = np.empty((0, 128, 321))
        Y2 = np.empty((0, 128, 2))
        u += 1
        print("Processing %d/%d" % (u, TOTAL))
        # print(audiofile)
        refFile = os.path.join(ref_dir, filename.split(".")[0] + "REF.txt")

        ref_time, ref_freq = mir_eval.io.load_time_series(refFile)
        if resample:
            interval = np.around(HOP_SIZE / SAMPLE_RATE, 10)

            new_time = np.arange(0, ref_time.shape[0] * (ref_time[1] - ref_time[0]), interval)
            ref_freq, _ = mir_eval.melody.resample_melody_series(ref_time, ref_freq, ref_freq, new_time, kind='nearest')

        cfp, _, _ = cfp_process(os.path.join(path, filename + ".wav"), sr=22050, hop=128)

        if ref_freq.shape[0] > cfp.shape[2]:
            ref_freq = ref_freq[0:cfp.shape[2]]
        else:
            cfp = cfp[:, :, 0:ref_freq.shape[0]]

        total_ref = []
        for i in range(ref_freq.shape[0]):
            single_ref = np.zeros(shape=(321))
            index = freq2ind(ref_freq[i], 31, 1250, 60)
            single_ref[index] = 1
            # print(index)
            total_ref.append(single_ref)
        total_ref = np.array(total_ref)


        cfp = cfp.transpose(2, 1, 0)
        #
        x_test = []
        # for padding
        padNum = cfp.shape[0] % SIZE
        if padNum != 0:
            len_pad = SIZE - padNum
            padding_feature = np.zeros(shape=(len_pad, 320, 3))
            cfp = np.vstack((cfp, padding_feature))

        for j in range(0, cfp.shape[0], SIZE):
            x_test_tmp = cfp[range(j, j + SIZE), :, :]
            x_test.append(x_test_tmp)
        x_test = np.array(x_test)

        y_test = []
        # for padding
        if padNum != 0:
            # a = np.ones(shape=(len_pad, 1), dtype=int)
            b = np.zeros(shape=(len_pad, 321), dtype=int)

            # padding_feature = np.concatenate((a, b), axis=1)
            total_ref = np.vstack((total_ref, b))

        for j in range(0, total_ref.shape[0], SIZE):
            y_test_tmp = total_ref[range(j, j + SIZE), :]
            y_test.append(y_test_tmp)
        y_test = np.array(y_test)

        X = np.vstack((X, x_test))
        Y1 = np.vstack((Y1, y_test))
        print(X.shape, Y1.shape)
        savepath = os.path.join(resdir, filename + "x.npy")
        np.save(savepath, X)
        savepath = os.path.join(resdir, filename + "y.npy")
        np.save(savepath, Y1)

        # savepath = os.path.join(resdir, filename + "y2.npy")
        # np.save(savepath, Y2)


# train()
# # train()
X = np.empty((0, 128, 320, 3))
Y1 = np.empty((0, 128, 321))
# Y2 = np.empty((0, 128, 2))
#
#
f = open('./mir.txt')
file_lists = f.readlines()
file_lists = [''.join(i).strip('\n') for i in file_lists]
# u = 0
# TOTAL = len(file_lists)
# path = "/home/qx/Feature/cfp_origin/"
# for filename in file_lists:
#     u += 1
#     print("Processing %d/%d" % (u, TOTAL))
#     x = np.load(os.path.join(path, filename) + "x.npy")
#     y = np.load(os.path.join(path, filename) + "y.npy")
#     X = np.vstack((X, x))
#     Y1 = np.vstack((Y1, y))
# #     y2 = np.load(os.path.join(path,filename) + "y2.npy")
# #     Y2 = np.vstack((Y2,y2))
# print(X.shape, Y1.shape)
# # print(Y2.shape)
# savepath = os.path.join("/home/qx/Feature/cfp_mir1k_128_22050/",  "x_train.npy")
# np.save(savepath, X)
# savepath = os.path.join("/home/qx/Feature/cfp_mir1k_128_22050/", "y1_train.npy")
# np.save(savepath, Y1)
path = "/home/qx/Feature/cfp_mir1k_128_22050/"
for filename in file_lists:
    os.rename(os.path.join(path, filename) + "x.npy",os.path.join(path, filename) + ".npy")