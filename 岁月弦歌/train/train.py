import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from model import ACappella
from tokenizer_utils import load_tokenizer
from train.dataset import DialogueDataset


def _is_new_format(ck):
    """新格式 = 我们自己存的 dict，含 model/optimizer/epoch 等元信息"""
    return isinstance(ck, dict) and "model" in ck and "epoch" in ck


def train(fresh=False):
    print("=== 开始训练 ===", flush=True)

    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}", flush=True)

    # 加载 tokenizer
    print("正在加载 tokenizer...", flush=True)
    tokenizer = load_tokenizer(cfg.tokenizer_name)
    print("tokenizer 加载完成", flush=True)

    # 加载数据集
    print("正在加载数据集...", flush=True)
    dataset = DialogueDataset(cfg.data_path, tokenizer, cfg.max_seq_len)
    print(f"数据集条数: {len(dataset)}", flush=True)

    if len(dataset) == 0:
        print("错误：数据集为空，请检查 dialogue.jsonl 格式", flush=True)
        return

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print(f"每个 epoch 的 batch 数: {len(loader)}", flush=True)

    # 构建模型
    print("正在初始化模型...", flush=True)
    model = ACappella(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e8:.2f} 亿", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # ---------- 断点续训 ----------
    # 不再每次从零覆盖：存在 ckpt 就接着上次的位置和权重继续训。
    # 同时恢复 AdamW 的动量/二阶矩，续训才是数学上连续的（否则动量清零，
    # 前几步会有一次明显抖动）。
    start_epoch = 0
    best_loss = float("inf")
    done_steps = 0

    if fresh:
        print("[续训] fresh=True，忽略已有 checkpoint，从零开始训练", flush=True)
    elif os.path.exists(cfg.ckpt_path):
        try:
            ck = torch.load(cfg.ckpt_path, map_location=device)
        except Exception as e:  # 文件损坏等
            print(f"[续训] checkpoint 读取失败，从零开始: {e}", flush=True)
            ck = None

        if ck is not None:
            if _is_new_format(ck):
                model.load_state_dict(ck["model"])
                # optimizer 状态只有在 param_groups 结构一致时才能安全恢复
                opt_state = ck.get("optimizer")
                if opt_state and opt_state.get("state"):
                    try:
                        optimizer.load_state_dict(opt_state)
                        print("[续训] 已恢复 AdamW 动量状态", flush=True)
                    except Exception as e:
                        print(f"[续训] optimizer 状态不兼容，仅恢复权重: {e}", flush=True)
                start_epoch = int(ck.get("epoch", 0))
                done_steps = int(ck.get("step", 0))
                best_loss = float(ck.get("best_loss", float("inf")))
                print(
                    f"[续训] 已从 {cfg.ckpt_path} 续训：起始 epoch={start_epoch}，"
                    f"累计 step={done_steps}，历史 best_loss={best_loss:.4f}",
                    flush=True,
                )
            else:
                # 旧格式：纯 state_dict，无 epoch/optimizer 元信息。
                # 无法判断它是在哪一版对话模板下训的（本项目改过 BOS/EOS 编码，
                # 旧模板权重与现在的编码不兼容），因此默认跳过而不是盲目续训。
                print(
                    "[续训] 检测到旧格式 checkpoint（无 epoch/optimizer 元信息）。\n"
                    "       它可能是在旧对话模板下训练的，续训会学到错误编码，故跳过。\n"
                    "       将从零开始训练并覆盖该文件。",
                    flush=True,
                )
    else:
        print("[续训] 未找到 checkpoint，从零开始训练", flush=True)

    # ---------- 训练循环 ----------
    # cfg.epochs 的语义 = 本次运行再训多少个 epoch（累加在 start_epoch 之上），
    # 因此可以分多次运行、每次训一点，成果持续累积。
    end_epoch = start_epoch + cfg.epochs
    print(f"本次运行: epoch {start_epoch + 1} → {end_epoch}（共 {cfg.epochs} 个）", flush=True)

    best_path = os.path.splitext(cfg.ckpt_path)[0] + "_best.pt"

    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0

        for step, batch in enumerate(loader):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            logits, _ = model(x)
            # 语言模型：位置 t 的输出预测 t+1 的 token（错位对齐）
            loss = loss_fn(
                logits[:, :-1, :].reshape(-1, cfg.vocab_size),
                y[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            done_steps += 1

            print(
                f"Epoch {epoch+1}/{end_epoch}  "
                f"Step {step+1}/{len(loader)}  "
                f"Loss {loss.item():.4f}",
                flush=True
            )

        avg_loss = total_loss / len(loader)
        print(f"===== Epoch {epoch+1} 完成  avg_loss={avg_loss:.4f} =====", flush=True)

        # 主 checkpoint：完整续训状态（覆盖写，只占一份磁盘）
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,      # 下次从这里继续
                "step": done_steps,
                "best_loss": best_loss,
                "cfg_vocab_size": cfg.vocab_size,
            },
            cfg.ckpt_path,
        )
        print(f"已保存 {cfg.ckpt_path}（epoch={epoch+1}, step={done_steps}）", flush=True)

        # best 备份：只在创新低时存，且只存权重（省 optimizer 的 2/3 体积）
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "best_loss": best_loss}, best_path)
            print(f"★ 创新低 avg_loss={avg_loss:.4f}，已备份 {best_path}", flush=True)

    print("=== 训练结束 ===", flush=True)
    print(f"累计训练 epoch={end_epoch}, step={done_steps}, best_loss={best_loss:.4f}", flush=True)


if __name__ == "__main__":
    import sys
    train(fresh=("--fresh" in sys.argv))
