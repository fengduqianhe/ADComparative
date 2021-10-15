from tqdm import tqdm
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from wrap.ADNI_joint import AdniDataSet
from setting.pram_setting import parse_opts
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [4]))
start = datetime.now()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FirstNet(nn.Module):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer4.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer4.add_module('relu4', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64 * f, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.fc4 = nn.Linear(512 * 3 * 3 * 3, 256)
        self.fc5 = nn.Linear(256, 3)



    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)

        x1 = self.layer4(x1)
        x2 = self.layer4(x2)

        x_score1 = x1
        x_score2 = x2
        x_score1 = self.avgpool(x_score1)
        x_score2 = self.avgpool(x_score2)
        x_score1 = x_score1.view(x_score1.shape[0], -1)
        x_score2 = x_score2.view(x_score2.shape[0], -1)
        x_score = torch.cat((x_score1, x_score2), 1)
        x_score = self.fc4(x_score)
        x_s = self.fc5(x_score)

        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)

        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)

        x2 = self.fc1(x2)
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)

        return x1, x2, x_s


if __name__ == "__main__":
    sets = parse_opts()
    sets.gpu_id = [0]

    train_data_path1 = '/train_label_single.csv'
    train_data_path2 = '/train_label_single_shuffle.csv'
    val_data_path1 = '/val_label.csv'
    val_data_path2 = '/val_label_shuffle.csv'

    train_img_path = '/train/'
    val_img_path = '/val/'

    train_db = AdniDataSet(train_data_path1, train_data_path2, train_img_path, sets)
    val_db = AdniDataSet(val_data_path1, val_data_path2, val_img_path, sets)
    """将训练集划分为训练集和验证集"""

    # train_db, val_db = torch.utils.data.random_split(train_data, [700, 180])
    print('train:', len(train_db), 'validation:', len(val_db))

    train_loader = DataLoader(dataset=train_db, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=val_db, batch_size=12, shuffle=True)

    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=40, verbose=True)
    model.to(device)
    result_list = []
    epochs = 100
    running_loss = 0

    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs1, inputs2, labels1, labels2, labels3 = data
            inputs1, inputs2, labels1, labels2, labels3 = inputs1.to(device), inputs2.to(
                device), labels1.to(device), labels2.to(device), labels3.to(device)
            optimizer.zero_grad()
            logps1, logps2, logps_s = model.forward(inputs1,inputs2)

            loss1 = criterion(logps1, labels1)
            loss2 = criterion(logps2, labels2)
            loss_group = loss1 + loss2
            loss_compare = criterion(logps_s, labels3)
            loss = 0.5 * loss_group + 0.5 * loss_compare

            _, predict1 = torch.max(logps1, 1)
            _, predict2 = torch.max(logps2, 1)

            correct1 += (predict1 == labels1).sum().item()
            total1 += labels1.size(0)
            correct2 += (predict2 == labels2).sum().item()
            total2 += labels2.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, running_loss / len(train_loader)))
        print("The accuracy of total {} images: {}%".format(total1, 100 * correct1 / total1))

        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        classnum = 2
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))
        roc_label = []
        roc_predict = []
        model.eval()
        with torch.no_grad():
            print("validation...")
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                inputs1, inputs2, labels1, labels2, labels3 = data
                inputs1, inputs2, labels1, labels2, labels3 = inputs1.to(device), inputs2.to(
                    device),labels1.to(device), labels2.to(
                    device), labels3.to(device)
                output1, output2, embedding1, embedding2, logps_s = model.forward(inputs1, inputs2)

                loss1 = criterion(output1, labels1)
                loss2 = criterion(output2, labels2)
                loss_group = loss1 + loss2
                print("loss_group", loss_group)

                loss_compare = criterion(logps_s, labels3)
                print("loss_compare", loss_compare)

                loss = 0.5 * loss_group + 0.5 * loss_compare
                print("loss", loss)

                _, predict1 = torch.max(output1, 1)
                _, predict2 = torch.max(output2, 1)


                running_loss += loss.item()

                total1 += labels1.size(0)
                total2 += labels2.size(0)

                correct1 += (predict1 == labels1).sum().item()
                correct2 += (predict2 == labels2).sum().item()

                '''calculate  Recall Precision F1'''
                pre_mask = torch.zeros(output1.size()).scatter_(1, predict1.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(output1.size()).scatter_(1, labels1.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)

            val_loss = running_loss / len(val_loader)

            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            # 精度调整
            recall = (recall.numpy()[0] * 100).round(3)
            precision = (precision.numpy()[0] * 100).round(3)
            F1 = (F1.numpy()[0] * 100).round(3)
            # print(roc_label)
            # print(roc_predict)
            fpr, tpr, threshold = roc_curve(roc_label, roc_predict, pos_label=1)  ###计算真正率和假正率
            roc_auc = auc(fpr, tpr)  ###计算auc的值

            print("The accuracy of valid {} images: {}%".format(total1, 100 * correct1 / total1))
            print(
                "The accuracy of valid {} images: recall {}, precision {}, F1 {}".format(total1, recall, precision, F1))
            result_list.append(
                [epoch, round(train_loss, 3), round(val_loss, 3), round(correct1 / total1, 3),
                 round(correct2 / total2, 3), round(recall[1], 3), round(precision[1], 3), round(F1[1], 3),round(roc_auc, 3)
               ])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc1', 'val_acc2', 'recall', 'precision', 'F1',  'AUC'
                   ]

            result = pd.DataFrame(columns=name, data=result_list)

            result.to_csv("subject_mmse.csv", mode='w', index=False,
                          header=False)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:

                print("Early stopping")
                break
    stop = datetime.now()
    print("Running time: ", stop - start)
