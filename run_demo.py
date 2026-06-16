"""水质预测模型的训练与测试入口。

新手优先修改 DemoConfig 中的常用参数；命令行参数会覆盖这里的默认值。
示例：
    python run_demo.py
    python run_demo.py --target CODMn --epochs 5 --device cpu
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TARGET_CHOICES = ("pH", "DO", "NH3N", "CODMn")


@dataclass
class DemoConfig:
    # ===== 常用参数：一般只需要改这一组 =====
    target: str = "DO"  # 预测指标，可选：pH、DO、NH3N、CODMn。
    data_file: str = "CN_WQ_selected_sites.csv"  # 已整理好的 demo 数据。
    epochs: int = 100  # 训练轮数；调试用 1-3，正式实验可适当增大。
    batch_size: int = 64  # 训练批大小；显存不足时调小。
    device: str = "auto"  # auto 自动用 GPU；cpu 强制 CPU；cuda 强制 GPU。
    seed: int = 2026  # 随机种子，便于复现实验。

    # ===== 时间窗口参数 =====
    seq_len: int = 10  # 历史输入长度。
    label_len: int = 5  # decoder 可见的历史长度。
    pred_len: int = 3  # 未来预测长度。
    patch_len: int = 4  # Transformer patch 长度。
    stride: int = 2  # patch 滑动步长。
    freq: str = "d"  # 时间频率；日尺度数据用 d。

    # ===== 训练与模型参数：不确定时保持默认 =====
    learning_rate: float = 5e-4
    patience: int = 5  # 早停耐心轮数；验证集连续若干轮不提升时停止训练。
    d_model: int = 512
    d_ff: int = 256
    n_heads: int = 8
    e_layers: int = 1
    dropout: float = 0.05
    lradj: str = "constant"

    # ===== 模糊规则与残差预测参数 =====
    membership_beta: float = 0.5
    residual_scale_init: float = 0.1
    use_residual_forecast: bool = True
    use_local_residual_anchor: bool = True

    # ===== 输出和可视化参数 =====
    visual_interval: int = 0  # 每隔多少个测试 batch 保存一张 PDF；0 表示关闭。
    show_progress: bool = True  # 是否显示 tqdm 进度条。

    # ===== 设备细节 =====
    gpu: int = 0  # 多块 GPU 时选择 cuda:<gpu>。


DEFAULT_CONFIG = DemoConfig()


class DemoHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """普通参数显示默认值，开关参数不显示容易误解的 default。"""

    def _get_help_string(self, action):
        hidden_default_actions = (
            argparse._StoreFalseAction,
            argparse._StoreConstAction,
        )
        if isinstance(action, hidden_default_actions) or action.default is argparse.SUPPRESS:
            return action.help
        return super()._get_help_string(action)


def load_torch():
    """运行训练前再导入 PyTorch，保证 --help 不依赖深度学习环境。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[依赖缺失] 未找到 PyTorch。请先安装 requirements.txt，"
            "并按你的 CUDA/CPU 环境安装对应的 torch 版本。"
        ) from exc
    return torch


def resolve_gpu_usage(config):
    """根据 device 参数判断是否启用 GPU。"""
    torch = load_torch()

    if config.device == "cpu":
        return False
    if config.device == "cuda" and not torch.cuda.is_available():
        print("[提示] 已指定 --device cuda，但当前 PyTorch 未检测到 CUDA，将改用 CPU。")
        return False
    return torch.cuda.is_available()


def build_args(config):
    """把直观的 DemoConfig 转成实验代码需要的完整参数对象。"""
    project_dir = Path(__file__).resolve().parent
    record_dir = project_dir / "Record"
    use_gpu = resolve_gpu_usage(config)

    fixed_args = {
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model_id": "forecast",
        "model": "Proposed",
        "data": "Dataset_DO_hour",
        "root_path": str(project_dir / "dataset" / "Public-dataset") + "/",
        "features": "MS",
        "seasonal_patterns": "Weekly",
        "inverse": True,
        "top_k": 5,
        "num_kernels": 6,
        "enc_in": 4,
        "dec_in": 4,
        "c_out": 4,
        "d_layers": 1,
        "moving_avg": 25,
        "factor": 3,
        "distil": True,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "num_workers": 0,
        "itr": 1,
        "des": "forecast",
        "loss": "mse",
        "use_amp": False,
        "use_multi_gpu": False,
        "devices": str(config.gpu),
        "p_hidden_dims": [128, 128],
        "p_hidden_layers": 2,
        "use_station_embedding": False,
        "num_stations": 149,
        "use_defuzzification": True,
        "x_min": -3.0,
        "x_max": 3.0,
        "fuzzy_N": 5,
        "fuzzy_alpha": 1.5,
    }
    user_args = {
        "data_path": config.data_file,
        "target": config.target,
        "freq": config.freq,
        "checkpoints": str(record_dir / "Model_Save"),
        "seq_len": config.seq_len,
        "label_len": config.label_len,
        "pred_len": config.pred_len,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "e_layers": config.e_layers,
        "d_ff": config.d_ff,
        "dropout": config.dropout,
        "patch_len": config.patch_len,
        "stride": config.stride,
        "train_epochs": config.epochs,
        "batch_size": config.batch_size,
        "patience": config.patience,
        "learning_rate": config.learning_rate,
        "lradj": config.lradj,
        "use_gpu": use_gpu,
        "gpu": config.gpu,
        "membership_beta": config.membership_beta,
        "use_residual_forecast": config.use_residual_forecast,
        "use_local_residual_anchor": config.use_local_residual_anchor,
        "residual_scale_init": config.residual_scale_init,
        "results_path": str(record_dir / "results"),
        "test_results_path": str(record_dir / "test_results"),
        "result_log_path": str(record_dir / "result_forecast.txt"),
        "visual_interval": config.visual_interval,
        "show_progress": config.show_progress,
    }
    return SimpleNamespace(**fixed_args, **user_args)


def build_setting(args):
    """生成本次实验的目录名。"""
    return (
        f"forecast_{args.model}_{args.target}_"
        f"sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
    )


def ensure_output_dirs(args):
    """提前创建输出目录，避免训练结束保存时报错。"""
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(args.results_path).mkdir(parents=True, exist_ok=True)
    Path(args.test_results_path).mkdir(parents=True, exist_ok=True)


def validate_config(config):
    """给常见错误提供更直接的中文提示。"""
    positive_int_fields = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "label_len": config.label_len,
        "pred_len": config.pred_len,
        "patch_len": config.patch_len,
        "stride": config.stride,
        "patience": config.patience,
    }
    for name, value in positive_int_fields.items():
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0，当前值为 {value}。")
    if config.label_len > config.seq_len:
        raise ValueError("label_len 不能大于 seq_len。")
    if config.patch_len > config.seq_len:
        raise ValueError("patch_len 不能大于 seq_len。")
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model 必须能被 n_heads 整除。")
    if config.visual_interval < 0:
        raise ValueError("visual_interval 不能小于 0；设置为 0 表示关闭批量图保存。")


def seed_everything(seed):
    """固定随机种子，尽量保证复现。"""
    torch = load_torch()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_run_config(args, setting):
    """用中文打印本次运行最关键的配置。"""
    rows = [
        ("实验目录", setting),
        ("预测指标", args.target),
        ("数据文件", args.data_path),
        ("运行设备", f"cuda:{args.gpu}" if args.use_gpu else "cpu"),
        ("训练轮数", args.train_epochs),
        ("批大小", args.batch_size),
        ("窗口长度", f"seq={args.seq_len}, label={args.label_len}, pred={args.pred_len}"),
        ("Patch 设置", f"patch_len={args.patch_len}, stride={args.stride}"),
        ("学习率", args.learning_rate),
        ("学习率策略", args.lradj),
        ("残差预测", "开启" if args.use_residual_forecast else "关闭"),
        ("局部残差锚点", "开启" if args.use_local_residual_anchor else "关闭"),
        ("测试范围", "完整测试集"),
        ("画图间隔", "关闭" if args.visual_interval == 0 else args.visual_interval),
    ]
    print("\n[配置] 本次运行参数")
    for name, value in rows:
        print(f"- {name}: {value}")


def parse_args():
    defaults = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="训练并测试 Proposed 水质预测模型。",
        formatter_class=DemoHelpFormatter,
    )

    common = parser.add_argument_group("常用参数")
    common.add_argument("--target", default=defaults.target, choices=TARGET_CHOICES, help="预测指标。")
    common.add_argument("--data-file", default=defaults.data_file, help="dataset/Public-dataset 下的数据文件名。")
    common.add_argument("--epochs", type=int, default=defaults.epochs, help="训练轮数。")
    common.add_argument("--batch-size", type=int, default=defaults.batch_size, help="训练批大小。")
    common.add_argument("--device", choices=["auto", "cpu", "cuda"], default=defaults.device, help="运行设备。")
    common.add_argument("--seed", type=int, default=defaults.seed, help="随机种子。")

    window = parser.add_argument_group("时间窗口参数")
    window.add_argument("--seq-len", type=int, default=defaults.seq_len, help="历史输入长度。")
    window.add_argument("--label-len", type=int, default=defaults.label_len, help="decoder 可见的历史长度。")
    window.add_argument("--pred-len", type=int, default=defaults.pred_len, help="未来预测长度。")
    window.add_argument("--patch-len", type=int, default=defaults.patch_len, help="Transformer patch 长度。")
    window.add_argument("--stride", type=int, default=defaults.stride, help="patch 滑动步长。")
    window.add_argument("--freq", default=defaults.freq, help="时间频率，日尺度用 d。")

    train = parser.add_argument_group("训练与模型参数")
    train.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="学习率。")
    train.add_argument("--patience", type=int, default=defaults.patience, help="早停耐心轮数。")
    train.add_argument("--d-model", type=int, default=defaults.d_model, help="Transformer 隐层维度。")
    train.add_argument("--d-ff", type=int, default=defaults.d_ff, help="前馈网络维度。")
    train.add_argument("--n-heads", type=int, default=defaults.n_heads, help="注意力头数。")
    train.add_argument("--e-layers", type=int, default=defaults.e_layers, help="encoder 层数。")
    train.add_argument("--dropout", type=float, default=defaults.dropout, help="dropout 比例。")
    train.add_argument("--lradj", default=defaults.lradj, choices=["constant", "type1", "type2"], help="学习率调整策略。")

    fuzzy = parser.add_argument_group("模糊规则与残差参数")
    fuzzy.add_argument("--membership-beta", type=float, default=defaults.membership_beta, help="模糊隶属度平滑系数。")
    fuzzy.add_argument("--residual-scale-init", type=float, default=defaults.residual_scale_init, help="残差分支初始缩放。")
    fuzzy.add_argument("--disable-residual-forecast", action="store_false", dest="use_residual_forecast", default=defaults.use_residual_forecast, help="关闭残差预测分支。")
    fuzzy.add_argument("--disable-local-residual-anchor", action="store_false", dest="use_local_residual_anchor", default=defaults.use_local_residual_anchor, help="关闭局部残差锚点。")

    output = parser.add_argument_group("测试与输出参数")
    output.add_argument("--visual-interval", type=int, default=defaults.visual_interval, help="每隔多少个测试 batch 保存 PDF；0 表示关闭。")
    output.add_argument("--hide-progress", action="store_false", dest="show_progress", default=defaults.show_progress, help="隐藏训练和测试进度条。")

    device = parser.add_argument_group("设备兼容参数")
    device.add_argument("--gpu", type=int, default=defaults.gpu, help="GPU 编号。")
    device.add_argument("--use-gpu", action="store_const", const="cuda", dest="device", help="兼容旧参数，等同于 --device cuda。")

    parsed = parser.parse_args()
    return DemoConfig(**vars(parsed))


def main():
    config = parse_args()
    validate_config(config)
    seed_everything(config.seed)

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    args = build_args(config)
    ensure_output_dirs(args)
    setting = build_setting(args)
    print_run_config(args, setting)

    exp = Exp_Long_Term_Forecast(args)
    exp.train(setting)
    exp.test(setting)

    result_dir = Path(args.results_path) / setting
    test_plot_dir = Path(args.test_results_path) / setting
    print("\n[完成] 运行结束")
    print(f"- 指标和预测数组: {result_dir}")
    print(f"- 汇总预测图: {result_dir / 'All_date.pdf'}")
    print(f"- 分批预测图: {test_plot_dir}")


if __name__ == "__main__":
    main()
