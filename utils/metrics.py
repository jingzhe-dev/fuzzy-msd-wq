import numpy as np
from sklearn.metrics import r2_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

# def KGE(pred,true):
#     pred = pred[:,0]
#     true = true[:,0]
#     error1 = []
#     for i in range(len(true)):
#         error1.append((true[i] - pred[i]) ** 2)
#     RSS = sum(error1)
#     error2 = []
#     for j in range(len(true)):
#         error2.append((true[j] - np.mean(true)) ** 2)
#     TSS = sum(error2)
#     R_square = 1 - RSS / TSS
#     pre_sigma = np.std(pred)
#     rea_sigma = np.std(true)
#     pre_mean = np.mean(pred)
#     rea_mean = np.mean(true)
#     x1 = (np.sqrt(R_square)-1)**2
#     x2 = (pre_sigma/rea_sigma - 1)**2
#     x3 = (pre_mean/rea_mean - 1)**2
#     KGE = 1-(np.sqrt(x1+x2+x3))
#     return 1-(np.sqrt(x1+x2+x3))

def KGE(pred, true):
    # 计算 Pearson 相关系数
    r = np.corrcoef(true.flatten(), pred.flatten())[0, 1]
    # 计算均值比
    alpha = np.mean(pred) / np.mean(true)
    # 计算标准差比
    beta = np.std(pred) / np.std(true)
    
    # 计算 KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge

def NSE(pred, true):
    """
    计算Nash-Sutcliffe Efficiency (NSE)
    """
    true = true.flatten()
    pred = pred.flatten()
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - np.mean(true)) ** 2)
    return 1 - (numerator / denominator)


def R2(pred, true):
    """
    计算Nash-Sutcliffe Efficiency (NSE)
    """
    true = true.flatten()
    pred = pred.flatten()
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - np.mean(true)) ** 2)
    return 1 - (numerator / denominator)

def SDE(pred, true):
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    error = (true - pred) / true
    error_mean = np.mean(error)
    error2 = (true - pred - error_mean) ** 2
    sde = np.sum(error2) / len(true)
    return float(sde)

def SMAP(pred, true):
    return 2.0 * np.mean(np.abs(pred - true) / (np.abs(pred) + np.abs(true))) * 100

def T_U(pred, true):
    numerator = np.sqrt(np.mean((pred - true)**2))
    denominator = np.sqrt(np.mean(true**2)) + np.sqrt(np.mean(pred**2))
    return numerator / denominator

def metric(pred, true):
    # 确保输入是一维数组
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    
    mae = float(MAE(pred, true))
    mse = float(MSE(pred, true))
    rmse = float(RMSE(pred, true))
    mape = float(MAPE(pred, true))
    mspe = float(MSPE(pred, true))
    r2 = float(R2(pred, true))
    smap = float(SMAP(pred, true))
    kge = float(KGE(pred, true))
    sde = float(SDE(pred, true))
    theil_u = float(T_U(pred, true))
    return mae, mse, rmse, mape, mspe, r2, smap, kge, sde, theil_u
