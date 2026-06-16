from data_provider.data_loader import Dataset_DO_hour
from torch.utils.data import DataLoader


data_dict = {
    "Dataset_DO_hour": Dataset_DO_hour,
}


def data_provider(args, flag):
    """根据 train/val/test 标记构造 Dataset 和 DataLoader。"""
    if args.task_name not in ("long_term_forecast", "short_term_forecast"):
        raise ValueError(
            "Only forecasting tasks are supported by Dataset_DO_hour."
        )

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1

    if flag == "train":
        # 训练阶段打乱样本；不丢弃最后一个不满 batch，确保数据完整使用。
        shuffle_flag = True
        batch_size = args.batch_size
    elif flag == "test":
        # 测试阶段逐条预测，便于保存样本级预测曲线。
        shuffle_flag = False
        batch_size = 1
    else:
        # 验证阶段保持时间顺序，并完整使用所有验证窗口。
        shuffle_flag = False
        batch_size = args.batch_size

    drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
    )

    num_workers = getattr(args, "num_workers", 0)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle_flag,
        "num_workers": num_workers,
        "drop_last": drop_last,
        # 仅 GPU 训练时启用 pinned memory。
        "pin_memory": getattr(args, "pin_memory", True) and args.use_gpu,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 4)

    return data_set, DataLoader(data_set, **loader_kwargs)
