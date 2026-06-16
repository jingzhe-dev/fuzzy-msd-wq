import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """按配置调整学习率；`constant` 表示训练中不改变学习率。"""
    if getattr(args, 'lradj', 'type1') in ('constant', 'none'):
        return
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        return
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    """验证集 loss 长期不下降时提前停止，并保存当前最优模型。"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # np.Inf was removed in NumPy 2.0
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'[训练] 验证集未提升: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print('[训练] 验证集 loss 下降，保存模型。')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """保存真实值和预测值对比图。"""
    output_dir = os.path.dirname(str(name))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class record_to_xls():
    def __init__(self,xls,sht1):
        self.xls = xls
        self.sht1 = sht1
            
    def save(self,pos,name,Moudle=None,MSE=None,MAE=None,RMSE=None,MAPE= None,MSPE=None,R2=None,SMAP=None,KEG=None,SDE=None,T_U=None):
        self.sht1.write(pos,0,str(Moudle))
        self.sht1.write(pos,1,str(MSE))
        self.sht1.write(pos,2,str(MAE))
        self.sht1.write(pos,3,str(RMSE))
        self.sht1.write(pos,4,str(MAPE))
        self.sht1.write(pos,5,str(MSPE))
        self.sht1.write(pos,6,str(R2))
        self.sht1.write(pos,7,str(SMAP))
        self.sht1.write(pos,8,str(KEG))
        self.sht1.write(pos,9,str(SDE))
        self.sht1.write(pos,10,str(T_U))
        
        self.xls.save('./Record/'+ str(name) +'_record.xls')
    
    def save2(self,pos,name,Moudle=None,MSE=None,MAE=None,RMSE=None,MAPE= None,MSPE=None,R2=None,SMAP=None,KEG=None,SDE=None,T_U=None):
        self.sht1.write(pos,0,str(Moudle))
        self.sht1.write(pos,1,str(MSE))
        self.sht1.write(pos,2,str(MAE))
        self.sht1.write(pos,3,str(RMSE))
        self.sht1.write(pos,4,str(MAPE))
        self.sht1.write(pos,5,str(MSPE))
        self.sht1.write(pos,6,str(R2))
        self.sht1.write(pos,7,str(SMAP))
        self.sht1.write(pos,8,str(KEG))
        self.sht1.write(pos,9,str(SDE))
        self.sht1.write(pos,10,str(T_U))
        
        self.xls.save(str(name) +'_record.xls')
    
    def add_title(self):
        self.sht1.write(0, 0, '模型名称')
        self.sht1.write(0, 1, 'MSE')
        self.sht1.write(0, 2, 'MAE')
        self.sht1.write(0, 3, 'RMSE')
        self.sht1.write(0, 4, 'MAPE')
        self.sht1.write(0, 5, 'MSPE')
        self.sht1.write(0, 6, 'R2')
        self.sht1.write(0, 7, 'SMAP')
        self.sht1.write(0, 8, 'KEG')
        self.sht1.write(0, 9, 'SDE')
        self.sht1.write(0, 10, 'T_U')
        
def get_column_names(file_path):
    data = pd.read_csv(file_path)
    return data.columns[1:].tolist()

def get_len(file_path,target):
    data = pd.read_csv(file_path)
    return len(data[target])
