"""长时序预测实验流程。

该文件负责连接数据、模型、训练、验证、测试和绘图输出。
"""
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    """长时序预测任务封装。"""

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _show_progress(self):
        """是否显示 tqdm 进度条。"""
        return bool(getattr(self.args, 'show_progress', False))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """在验证集或测试集上计算 MSE loss。"""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        print(f"\n[训练] 开始训练: {setting}")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            train_iter = tqdm(
                enumerate(train_loader),
                desc=f'{self.args.model} 训练',
                total=len(train_loader),
                disable=not self._show_progress(),
                leave=False,
            )
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in train_iter:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            epoch_cost = time.time() - epoch_time
            
            # 降低验证开销：默认只在最后一个 epoch 计算 test loss。
            if (epoch + 1) % 100 == 0 or epoch == self.args.train_epochs - 1:
                test_loss = self.vali(
                    test_data,
                    test_loader,
                    criterion,
                )
                print(
                    "[训练] Epoch {0:03d}/{1:03d} | steps={2} | "
                    "train={3:.6f} | val={4:.6f} | test={5:.6f} | time={6:.1f}s".format(
                        epoch + 1, self.args.train_epochs, train_steps,
                        train_loss, vali_loss, test_loss, epoch_cost
                    )
                )
            else:
                print(
                    "[训练] Epoch {0:03d}/{1:03d} | steps={2} | "
                    "train={3:.6f} | val={4:.6f} | time={5:.1f}s".format(
                        epoch + 1, self.args.train_epochs, train_steps,
                        train_loss, vali_loss, epoch_cost
                    )
                )
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("[训练] 早停触发，停止训练。")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        print(f"[训练] 最优模型已加载: {best_model_path}")

        return self.model

    def test(self, setting, test=1,old=None):
        print(f"\n[测试] 开始测试: {setting}")
        test_data, test_loader = self._get_data(flag='test')
        if test:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            print(f"[测试] 加载模型: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds = []
        trues = []
        test_results_root = getattr(self.args, 'test_results_path', './Record/test_results')
        folder_path = os.path.join(test_results_root, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        results_root = getattr(self.args, 'results_path', './Record/results')
        results_folder_path = os.path.join(results_root, setting)
        if not os.path.exists(results_folder_path):
            os.makedirs(results_folder_path)

        self.model.eval()
        visual_interval = getattr(self.args, 'visual_interval', 0)
        with torch.no_grad():
            test_iter = tqdm(
                enumerate(test_loader),
                desc=f'{self.args.model} 测试',
                total=len(test_loader),
                disable=not self._show_progress(),
                leave=False,
            )
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_iter:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if visual_interval and i % visual_interval == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    figure_name = str(i) + '.pdf'
                    visual(gt, pd, os.path.join(folder_path, figure_name))

        preds = np.array(preds)
        trues = np.array(trues)
        pred_batch_shape, true_batch_shape = preds.shape, trues.shape
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print(f"[测试] 预测数组形状: batch={pred_batch_shape} -> flat={preds.shape}")
        print(f"[测试] 真实数组形状: batch={true_batch_shape} -> flat={trues.shape}")

        # 保存指标、预测数组和汇总预测图。
        folder_path = results_folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe ,r2,smap,kge,sde,t_u= metric(preds, trues)
        print(
            "[结果] MSE={:.6f} | MAE={:.6f} | RMSE={:.6f} | R2={:.6f} | KGE={:.6f}".format(
                mse, mae, rmse, r2, kge
            )
        )
        result_log_path = getattr(self.args, 'result_log_path', 'result_forecast.txt')
        result_log_parent = os.path.dirname(result_log_path)
        if result_log_parent:
            os.makedirs(result_log_parent, exist_ok=True)
        f = open(result_log_path, 'a', encoding='utf-8')
        f.write(setting + "\n")
        f.write(
            "MSE={:.6f}, MAE={:.6f}, RMSE={:.6f}, MAPE={:.6f}, "
            "MSPE={:.6f}, R2={:.6f}, SMAP={:.6f}, KGE={:.6f}, "
            "SDE={:.6f}, T_U={:.6f}\n\n".format(
                mse, mae, rmse, mape, mspe, r2, smap, kge, sde, t_u
            )
        )
        f.close()
        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mse, mae, rmse, mape, mspe,r2,smap,kge,sde,t_u]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        plot_true = trues.reshape(-1)
        plot_pred = preds.reshape(-1)
        visual(plot_true, plot_pred, os.path.join(folder_path, 'All_date.pdf'))
        if len(plot_true) > 800:
            visual(
                plot_true[200:800],
                plot_pred[200:800],
                os.path.join(folder_path, '200-800.pdf'),
            )
        print(f"[结果] 指标和预测数组已保存: {folder_path}")
        print(f"[结果] 汇总预测图已保存: {os.path.join(folder_path, 'All_date.pdf')}")

        return

    def model_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        定义模型的前向传播函数，用于 Captum 的 IG 分析。

        参数:
        - x_enc: 编码器的输入序列
        - x_mark_enc: 编码器的时间标记
        - x_dec: 解码器的输入序列
        - x_mark_dec: 解码器的时间标记

        返回:
        - outputs: 模型的预测输出
        """
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if model.task_name in ['long_term_forecast', 'short_term_forecast']:
            return outputs
        else:
            raise NotImplementedError("Only forecasting tasks are supported for IG analysis.")

    def plot_heatmap(self, attributions, sample_id, pred_step, feature_idx, save_path):
        """
        绘制单个特征在特定预测步的属性值热力图。
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(attributions, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title(f'IG Heatmap - Sample {sample_id} - Step {pred_step} - Feature {feature_idx}')
        plt.xlabel('Time Steps')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'sample_{sample_id}_step_{pred_step}_feature_{feature_idx}_heatmap.png'))
        plt.close()

    def plot_aggregated_attributions(self, attributions, sample_id, save_path, aggregation='mean'):
        """
        聚合并绘制属性值。
        """
        if aggregation == 'mean':
            aggregated = np.mean(attributions, axis=(0, 1))
        elif aggregation == 'sum':
            aggregated = np.sum(attributions, axis=(0, 1))
        else:
            raise ValueError("Aggregation must be 'mean' or 'sum'")
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(aggregated)), aggregated)
        plt.title(f'Aggregated IG Attributions for Sample {sample_id}')
        plt.xlabel('Features')
        plt.ylabel('Attribution')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'sample_{sample_id}_aggregated_attributions.png'))
        plt.close()
        
    def interpret_with_ig_all_outputs(
        self, 
        setting, 
        num_samples=5, 
        baseline_type='zeros', 
        save_path='./Record/IG_results/',
        aggregation='mean'
    ):
        """
        使用 Integrated Gradients (IG) 对模型的所有输出进行可解释性分析，并优化可视化。
        
        参数:
        - setting: 模型设置名称，用于加载相应的检查点和保存结果。
        - num_samples: 要进行 IG 分析的样本数量。
        - baseline_type: 基线类型，支持 'zeros' 或 'mean'。
        - save_path: 保存 IG 结果的根目录。
        - aggregation: 'mean' 或 'sum'，聚合属性值的方式。
        """
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise RuntimeError(
                "Integrated Gradients analysis requires optional dependency "
                "`captum`. Install it with `pip install captum`."
            ) from exc

        test_data, test_loader = self._get_data(flag='test')

        print('[IG] 加载模型检查点...')
        checkpoint_path = os.path.join(
            getattr(self.args, 'checkpoints', './Record/Model_Save'),
            setting,
            'checkpoint.pth',
        )
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        ig_folder = os.path.join(save_path, setting, 'IG')
        os.makedirs(ig_folder, exist_ok=True)

        print('[IG] 初始化 Integrated Gradients 对象')
        ig = IntegratedGradients(self.model_forward)

        processed_samples = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(test_loader),
            desc='IG 分析',
            total=len(test_loader),
            disable=not self._show_progress(),
            leave=False,
        ):
            if processed_samples >= num_samples:
                break

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            for sample_idx in range(batch_x.size(0)):
                if processed_samples >= num_samples:
                    break

                sample_x = batch_x[sample_idx].unsqueeze(0)
                sample_x_mark = batch_x_mark[sample_idx].unsqueeze(0)
                sample_y_mark = batch_y_mark[sample_idx].unsqueeze(0)
                sample_dec_inp = dec_inp[sample_idx].unsqueeze(0)

                sample_x.requires_grad = True

                if baseline_type == 'zeros':
                    baseline = torch.zeros_like(sample_x).to(self.device)
                elif baseline_type == 'mean':
                    baseline = torch.mean(sample_x, dim=1, keepdim=True).detach()
                else:
                    raise ValueError("Unsupported baseline_type. Choose 'zeros' or 'mean'.")

                outputs = self.model_forward(sample_x, sample_x_mark, sample_dec_inp, sample_y_mark)

                all_attributions = []

                for pred_step in range(outputs.shape[1]):
                    for feature_idx in range(outputs.shape[2]):
                        target = (pred_step, feature_idx)
                        try:
                            attributions, delta = ig.attribute(
                                inputs=sample_x,
                                additional_forward_args=(sample_x_mark, sample_dec_inp, sample_y_mark),
                                baselines=baseline,
                                target=target,
                                n_steps=50,
                                return_convergence_delta=True
                            )
                        except Exception as e:
                            print(f'[IG] 目标 {target} 计算失败: {e}')
                            continue

                        attributions = attributions.detach().cpu().numpy()
                        all_attributions.append(attributions[0])

                if not all_attributions:
                    print(f'[IG] 样本 {processed_samples} 未得到有效 attribution。')
                    processed_samples += 1
                    continue

                all_attributions = np.array(all_attributions)

                if aggregation == 'mean':
                    aggregated_attributions = np.mean(all_attributions, axis=0)
                elif aggregation == 'sum':
                    aggregated_attributions = np.sum(all_attributions, axis=0)
                else:
                    raise ValueError("Aggregation must be 'mean' or 'sum'")

                self.plot_heatmap(
                    attributions=aggregated_attributions, 
                    sample_id=processed_samples, 
                    pred_step='all', 
                    feature_idx='all', 
                    save_path=ig_folder
                )

                self.plot_aggregated_attributions(
                    attributions=all_attributions, 
                    sample_id=processed_samples, 
                    save_path=ig_folder, 
                    aggregation=aggregation
                )

                processed_samples += 1

        print(f'[IG] 分析完成，已处理样本数: {processed_samples}')
        print(f'[IG] 结果已保存: {ig_folder}')
