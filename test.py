import numpy as np
import torch
import mir_eval
import pandas as pd

from cfp import get_CenFreq
from evaluate import iseg, melody_eval, est
from network.ftaformer_pytorch import FTAFormer
from network.ftanet_pytorch import FTAnet
from network.psa_pytorch import PSAnet


def evaluatebymodel(model, modelname, gid, dataset, batch_size=16):
    avg_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
    f = open('./test.txt')
    file_lists = f.readlines()
    file_lists = [''.join(i).strip('\n') for i in file_lists]
    file_lists[0] = file_lists[0:1000]
    file_lists[1] = file_lists[1000:1012]
    file_lists[2] = file_lists[1012:1021]
    file_lists[3] = file_lists[1021:]
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    for i in range(len(file_lists[dataset])):
        print("Processing %d/%d" % (i, len(file_lists[dataset])))
        X = np.load('/home/qx/Test/%s.npy' % file_lists[dataset][i])
        y = mir_eval.io.load_time_series("/home/qx/Dataset/%sREF.txt" % file_lists[dataset][i])
        # predict and concat
        X = X.transpose(2, 1, 0)
        SIZE = 128
        # for padding
        x_test = []
        # for padding
        padNum = X.shape[0] % SIZE
        len_pad = 0
        if padNum != 0:
            len_pad = SIZE - padNum
            padding_feature = np.zeros(shape=(len_pad, 320, 3))
            X = np.vstack((X, padding_feature))

        for j in range(0, X.shape[0], SIZE):
            x_test_tmp = X[range(j, j + SIZE), :, :]
            x_test.append(x_test_tmp)
        x_test = np.array(x_test)
        X = np.array(x_test)
        X = X.transpose(0, 3, 2, 1)
        num = X.shape[0] // batch_size
        if X.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                x = X[j * batch_size:]

            else:
                x = X[j * batch_size: (j + 1) * batch_size]

            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k])
            x = torch.from_numpy(x).float()
            x = x.cuda(device=gid)
            prediction, _ = model(x)

            prediction = prediction[:, 0].cpu().detach().numpy()
            preds.append(prediction)

        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)

        # ground-truth
        ref_arr = np.array(y).transpose(1, 0)
        interval = np.around(128 / 22050, 10)

        new_time = np.arange(0, ref_arr[:, 0].shape[0] * (ref_arr[:, 0][1] - ref_arr[:, 0][0]), interval)
        time_arr = new_time

        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        est_arr = est(preds, CenFreq, time_arr, len_pad)

        # evaluate
        eval_arr = melody_eval(ref_arr, est_arr)
        avg_eval_arr += eval_arr
        # print(eval_arr)
        a.append(file_lists[dataset][i])
        b.append(eval_arr[0])
        c.append(eval_arr[1])
        d.append(eval_arr[2])
        e.append(eval_arr[3])
        f.append(eval_arr[4])
    avg_eval_arr /= len(file_lists[dataset])
    # VR, VFA, RPA, RCA, OA
    # print(avg_eval_arr)
    a.append(" ")
    b.append(avg_eval_arr[0])
    c.append(avg_eval_arr[1])
    d.append(avg_eval_arr[2])
    e.append(avg_eval_arr[3])
    f.append(avg_eval_arr[4])
    print(len(file_lists[dataset]))

    dataframe = pd.DataFrame({'filename': a, 'VR': b, 'VFA': c, 'RPA': d, 'RCA': e, 'OA': f})
    print(dataframe)
    testDataset = ''
    if dataset == 0:
        testDataset = 'MIR_1K'
    elif dataset == 1:
        testDataset = 'ADC04'
    elif dataset == 2:
        testDataset = 'MIREX05'
    elif dataset == 3:
        testDataset = 'MEDLEYDB_12'
    filecsv = './Results/' + modelname + testDataset
    dataframe.to_csv(r"%s.csv" % filecsv, sep=',')
    a, b, c, d, e, f = [], [], [], [], [], []
    a.append(modelname)
    a.append(avg_eval_arr[0])
    a.append(avg_eval_arr[1])
    a.append(avg_eval_arr[2])
    a.append(avg_eval_arr[3])
    a.append(avg_eval_arr[4])
    dataframe2 = pd.DataFrame([a])
    resDir = './Results/' + testDataset
    dataframe2.to_csv(r"%s.csv" % resDir, sep=',', mode="a", header=None)
    return avg_eval_arr


def test():
    gid = 0

    for i in range(0, 5):
        modelname = 'ftaformer_321_sa_'
        modelname = modelname + str(i + 1)
        model = FTAFormer()
        model.float()
        model.eval()
        model.cuda(device=gid)
        model.load_state_dict(torch.load("./model/%s" % modelname, map_location='cuda:%d' % gid))

        print(modelname)
        for j in range(4):
            evaluatebymodel(model, modelname, gid, j)


def plotbymodel(model, gid, batch_size=16):
    f = open('./testsingle.txt')
    file_lists = f.readlines()
    file_lists = [''.join(i).strip('\n') for i in file_lists]
    print(len(file_lists))

    for i in range(len(file_lists)):
        print("Processing %d/%d" % (i, len(file_lists)))
        X = np.load('/home/qx/Test/%s.npy' % file_lists[i])
        y = mir_eval.io.load_time_series("/home/qx/Dataset/%sREF.txt" % file_lists[i])
        # predict and concat
        X = X.transpose(2, 1, 0)
        SIZE = 128
        # for padding
        x_test = []
        # for padding
        padNum = X.shape[0] % SIZE
        len_pad = 0
        if padNum != 0:
            len_pad = SIZE - padNum
            padding_feature = np.zeros(shape=(len_pad, 320, 3))
            X = np.vstack((X, padding_feature))

        for j in range(0, X.shape[0], SIZE):
            x_test_tmp = X[range(j, j + SIZE), :, :]
            x_test.append(x_test_tmp)
        x_test = np.array(x_test)
        X = np.array(x_test)
        X = X.transpose(0, 3, 2, 1)
        num = X.shape[0] // batch_size
        if X.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                x = X[j * batch_size:]

            else:
                x = X[j * batch_size: (j + 1) * batch_size]

            # for k in range(length): # normalization
            #     X[k] = std_normalize(X[k])
            x = torch.from_numpy(x).float()
            x = x.cuda(device=gid)
            prediction, _ = model(x)

            prediction = prediction[:, 0].cpu().detach().numpy()
            preds.append(prediction)

        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)

        # ground-truth
        ref_arr = np.array(y).transpose(1, 0)
        interval = np.around(128 / 22050, 10)

        new_time = np.arange(0, ref_arr[:, 0].shape[0] * (ref_arr[:, 0][1] - ref_arr[:, 0][0]), interval)
        time_arr = new_time

        # trnasform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        est_arr = est(preds, CenFreq, time_arr, len_pad)

        # evaluate
        ref_time = ref_arr[:, 0]
        ref_freq = ref_arr[:, 1]

        est_time = est_arr[:, 0]
        est_freq = est_arr[:, 1]

        import matplotlib.pyplot as plt

        # plt.plot(ref_time, ref_freq, 'b', linestyle='dashdot', )
        plt.plot(ref_time, ref_freq, 'b', )
        plt.plot(est_time, est_freq, 'g', )

        plt.ylim(ymax=1000,ymin=-20)
        plt.legend(['Ground Truth', 'Prediction'])
        plt.xlabel('Time(s)', )  # 注意后面的字体属性
        plt.ylabel('Frequency(Hz)')
        plt.title('Melody')
        plt.show()
        # plt.imshow([ref_time,ref_freq], aspect='auto', cmap='binary')
        # plt.show()


def plot():
    gid = 1

    #
    # modelname = 'fta_48_oa_Bce_VAD_1DFT_4'
    # model = FTAFormer()
    #
    modelname = 'ftanet_48_oa_3'
    model = FTAnet()
    model.float()
    model.eval()
    model.cuda(device=gid)
    model.load_state_dict(torch.load("./model/%s" % modelname, map_location='cuda:%d' % gid))

    print(modelname)

    plotbymodel(model, gid)


plot()
# test()