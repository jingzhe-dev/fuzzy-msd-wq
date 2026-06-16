"""Extrapolate future water-quality values from a trained checkpoint."""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler

from models import Proposed
from utils.timefeatures import time_features


def build_args(cli):
    project_dir = Path(__file__).resolve().parent
    record_dir = project_dir / "Record"

    use_gpu = bool(cli.use_gpu and torch.cuda.is_available())
    if cli.device == "cpu":
        use_gpu = False

    return SimpleNamespace(
        task_name="long_term_forecast",
        model_id="forecast",
        model="Proposed",
        data="Dataset_DO_hour",
        root_path=str(project_dir / "dataset" / "Public-dataset") + "/",
        data_path=cli.data_file,
        features="MS",
        target=cli.target,
        freq=cli.freq,
        checkpoints=str(record_dir / "Model_Save"),
        seq_len=cli.seq_len,
        label_len=cli.label_len,
        pred_len=cli.pred_len,
        seasonal_patterns="Weekly",
        inverse=True,
        enc_in=4,
        dec_in=4,
        c_out=4,
        d_model=cli.d_model,
        n_heads=cli.n_heads,
        e_layers=cli.e_layers,
        d_layers=1,
        d_ff=cli.d_ff,
        factor=3,
        dropout=cli.dropout,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        patch_len=cli.patch_len,
        stride=cli.stride,
        use_amp=False,
        use_gpu=use_gpu,
        gpu=cli.gpu,
        use_multi_gpu=False,
        devices=str(cli.gpu),
        use_station_embedding=False,
        num_stations=149,
        use_defuzzification=True,
        x_min=-3.0,
        x_max=3.0,
        fuzzy_N=5,
        fuzzy_alpha=1.5,
        membership_beta=cli.membership_beta,
        use_residual_forecast=not cli.disable_residual_forecast,
        use_local_residual_anchor=not cli.disable_local_residual_anchor,
        residual_scale_init=cli.residual_scale_init,
    )


def build_setting(args):
    return (
        f"forecast_{args.model}_{args.target}_"
        f"sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
    )


def resolve_checkpoint(cli, args, setting):
    if cli.checkpoint:
        checkpoint_path = Path(cli.checkpoint)
    else:
        checkpoint_path = Path(args.checkpoints) / setting / "checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run training first or pass --checkpoint with an existing checkpoint path."
        )
    return checkpoint_path


def reorder_columns(df, target):
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    feature_cols = [col for col in df.columns if col != "date"]
    feature_cols = [col for col in feature_cols if col != target] + [target]
    return df[["date"] + feature_cols].copy(), feature_cols


def infer_future_dates(observed_dates, pred_len, freq, start_date=None):
    if start_date is not None:
        start = pd.to_datetime(start_date)
    else:
        start = pd.to_datetime(observed_dates.iloc[-1]) + to_offset(freq)
    return pd.date_range(start=start, periods=pred_len, freq=freq)


def prepare_inputs(args, start_date=None):
    data_path = Path(args.root_path) / args.data_path
    df_raw = pd.read_csv(data_path)
    df_raw, feature_cols = reorder_columns(df_raw, args.target)
    target_idx = feature_cols.index(args.target)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    if len(df_raw) < args.seq_len:
        raise ValueError(
            f"Need at least seq_len={args.seq_len} rows, got {len(df_raw)} rows."
        )

    values = df_raw[feature_cols]
    num_train = int(len(df_raw) * 0.4)
    scaler = StandardScaler()
    scaler.fit(values.iloc[:num_train].values)
    scaled_values = scaler.transform(values.values)

    history_values = scaled_values[-args.seq_len:]
    history_dates = df_raw["date"].iloc[-args.seq_len:]
    future_dates = infer_future_dates(df_raw["date"], args.pred_len, args.freq, start_date)

    label_values = scaled_values[-args.label_len:]
    decoder_values = np.zeros((args.label_len + args.pred_len, len(feature_cols)))
    decoder_values[:args.label_len] = label_values

    decoder_dates = pd.DatetimeIndex(
        list(df_raw["date"].iloc[-args.label_len:]) + list(future_dates)
    )
    x_mark_enc = time_features(pd.DatetimeIndex(history_dates), freq=args.freq).transpose(1, 0)
    x_mark_dec = time_features(decoder_dates, freq=args.freq).transpose(1, 0)

    tensors = {
        "x_enc": torch.tensor(history_values, dtype=torch.float32).unsqueeze(0),
        "x_mark_enc": torch.tensor(x_mark_enc, dtype=torch.float32).unsqueeze(0),
        "x_dec": torch.tensor(decoder_values, dtype=torch.float32).unsqueeze(0),
        "x_mark_dec": torch.tensor(x_mark_dec, dtype=torch.float32).unsqueeze(0),
    }
    return df_raw, feature_cols, target_idx, scaler, future_dates, tensors


def plot_extrapolation(history_dates, history_target, future_dates, pred_target, save_path, target):
    plt.figure(figsize=(9, 4))
    plt.plot(history_dates, history_target, label="History", linewidth=2)
    extrap_dates = pd.DatetimeIndex([history_dates.iloc[-1], *future_dates])
    extrap_values = np.concatenate([[history_target.iloc[-1]], pred_target])
    plt.plot(extrap_dates, extrap_values, label="Extrapolation", linewidth=2, marker="o")
    plt.axvline(history_dates.iloc[-1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrapolate future values with a trained Proposed model."
    )
    parser.add_argument("--target", default="DO", choices=["pH", "DO", "NH3N", "CODMn"])
    parser.add_argument("--data-file", default="CN_WQ_selected_sites_model.csv")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--label-len", type=int, default=5)
    parser.add_argument("--pred-len", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--membership-beta", type=float, default=0.5)
    parser.add_argument("--residual-scale-init", type=float, default=0.1)
    parser.add_argument("--disable-residual-forecast", action="store_true")
    parser.add_argument("--disable-local-residual-anchor", action="store_true")
    parser.add_argument("--freq", default="d")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    args = build_args(cli)
    setting = build_setting(args)
    checkpoint_path = resolve_checkpoint(cli, args, setting)
    output_dir = Path(cli.output_dir) if cli.output_dir else Path(__file__).resolve().parent / "Record" / "extrapolation" / setting
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    print("\n[配置] 外推参数")
    print(json.dumps({
        "setting": setting,
        "target": args.target,
        "data": args.data_path,
        "device": str(device),
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))

    df_raw, feature_cols, target_idx, scaler, future_dates, tensors = prepare_inputs(
        args, cli.start_date
    )
    tensors = {name: value.to(device) for name, value in tensors.items()}

    model = Proposed.Model(args).float().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(
            tensors["x_enc"],
            tensors["x_mark_enc"],
            tensors["x_dec"],
            tensors["x_mark_dec"],
        ).detach().cpu().numpy()[0]

    pred_target = (
        pred_scaled[:, target_idx] * scaler.scale_[target_idx]
        + scaler.mean_[target_idx]
    )
    pred_df = pd.DataFrame({
        "step": np.arange(1, args.pred_len + 1),
        "date": future_dates,
        args.target: pred_target,
    })

    csv_path = output_dir / "extrapolation.csv"
    pdf_path = output_dir / "extrapolation.pdf"
    npy_path = output_dir / "extrapolation.npy"
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    np.save(npy_path, pred_target)

    history_dates = df_raw["date"].iloc[-args.seq_len:]
    history_target = df_raw[args.target].iloc[-args.seq_len:]
    plot_extrapolation(
        history_dates,
        history_target,
        future_dates,
        pred_target,
        pdf_path,
        args.target,
    )

    print("\n[完成] 外推结束")
    print(f"- 外推结果: {csv_path}")
    print(f"- 外推数组: {npy_path}")
    print(f"- 外推图: {pdf_path}")


if __name__ == "__main__":
    main()
