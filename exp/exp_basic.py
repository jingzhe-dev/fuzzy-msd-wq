import os
import torch
from models import Proposed


class Exp_Basic(object):
    """实验基类：统一管理设备、模型注册和训练/测试接口。"""

    def __init__(self, args):
        self.args = args
        self.model_dict = {

            'Proposed': Proposed,

        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('[设备] 使用 GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('[设备] 使用 CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
