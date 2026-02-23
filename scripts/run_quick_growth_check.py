import os
import sys
import time
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable


def resolve_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def run_cmd(cmd, log_path=None, dry_run=False):
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return 0
    if log_path:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(cmd)}\n")
            return subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=log)
    return subprocess.call(cmd, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="快速验证动态神经元生长（baseline/random/actgrad）")
    parser.add_argument("--data_path", default="dataset/pretrain_hq.jsonl", type=str, help="预训练数据路径")
    parser.add_argument("--save_dir", default="out", type=str, help="权重输出目录")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    parser.add_argument("--epochs", default=1, type=int, help="训练轮数")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--accumulation_steps", default=1, type=int, help="梯度累积步数")
    parser.add_argument("--max_seq_len", default=256, type=int, help="序列长度")
    parser.add_argument("--hidden_size", default=256, type=int, help="模型维度")
    parser.add_argument("--num_hidden_layers", default=4, type=int, help="层数")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="学习率")
    parser.add_argument("--dtype", default="float16", type=str, help="混合精度类型")
    parser.add_argument("--log_dir", default="logs", type=str, help="日志目录")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args()

    data_path = resolve_path(args.data_path)
    save_dir = resolve_path(args.save_dir)
    log_dir = resolve_path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"quick_growth_{time.strftime('%Y%m%d_%H%M%S')}.log")

    base_args = [
        "--data_path", data_path,
        "--save_dir", save_dir,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--accumulation_steps", str(args.accumulation_steps),
        "--max_seq_len", str(args.max_seq_len),
        "--hidden_size", str(args.hidden_size),
        "--num_hidden_layers", str(args.num_hidden_layers),
        "--learning_rate", str(args.learning_rate),
        "--dtype", args.dtype
    ]

    runs = [
        ("quick_base", ["--neuron_growth", "0"]),
        ("quick_rand", [
            "--neuron_growth", "1",
            "--grow_method", "random",
            "--init_active_ratio", "0.8",
            "--grow_interval", "50",
            "--grow_ratio", "0.05",
            "--max_active_ratio", "0.99"
        ]),
        ("quick_actgrad", [
            "--neuron_growth", "1",
            "--grow_method", "act_grad",
            "--init_active_ratio", "0.8",
            "--grow_interval", "50",
            "--grow_ratio", "0.05",
            "--max_active_ratio", "0.99",
            "--grow_score_alpha", "1.0",
            "--grow_score_beta", "1.0",
            "--neuron_ema_beta", "0.1"
        ])
    ]

    for name, extra in runs:
        cmd = [PYTHON, "trainer/train_pretrain.py", "--save_weight", name] + base_args + extra
        ret = run_cmd(cmd, log_path=log_path, dry_run=args.dry_run)
        if ret != 0:
            print(f"[FAIL] {name} failed, check log: {log_path}")
            sys.exit(ret)

    print("[OK] Quick growth check finished.")
    print(f"[LOG] {log_path}")


if __name__ == "__main__":
    main()
