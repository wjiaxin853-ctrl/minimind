import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 真正拿来反向传播的总损失，是res.loss + res.aux_loss的和
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0 # 辅助损失，主要给像 MoE 这种结构用
            current_logits_loss = current_loss - current_aux_loss  # 主任务损失，也就是语言模型预测下一个 token 的损失
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 参数名：模型保存目录。
    # 作用：保存最终模型权重的目录。
    # 范围：【任意可写目录路径】。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")

    # 参数名：保存权重前缀名。
    # 作用：控制保存权重文件的前缀名。
    # 范围：【任意字符串】。
    # 影响：不影响训练效果，只影响文件命名和实验区分。
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")

    # 参数名：训练轮数。
    # 作用：控制整个数据集重复训练多少轮。
    # 范围：【1，2，3，4，5，...】整数。
    # 影响：越大训练越久、看数据次数越多，可能收敛更充分，但也更容易过拟合；越小训练更快，但可能学不够。
    # 推荐值：`2`；先跑通或做小实验可用 `1`。
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")

    # 参数名：单步批大小。
    # 作用：控制单次送入模型的 micro-batch 大小。每次更新参数时，你让模型参考了多少样本
    # 范围：【1，2，4，8，16，32，64】。
    # 影响：
    # 【越大】显存占用越高、吞吐通常越高、梯度更稳定；每次看到很多样本才更新
    # 越小更省显存，但训练可能更慢、噪声更大。每次只看少量样本就更新，容易被噪声带偏
    # 推荐值：两张 `3090` 先用 `8`；显存很宽裕再升到 `16`。
    # 越大不代表队模型效果更好，而是对训练过程更加稳定，不会波动很多
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    # 参数名：学习率。
    # 作用：控制参数更新步长。
    # 范围：【1e-5，2e-5，5e-5，1e-4，2e-4，5e-4，1e-3】。
    # 影响：越大学得越快，但更容易震荡或发散；越小更稳，但收敛更慢。
    # 推荐值：`5e-4`，先不要改。
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # 参数名：训练设备。
    # 作用：指定训练设备。
    # 范围：【cpu，cuda:0，cuda:1，...】。
    # 影响：选 GPU 速度会快很多；选 CPU 只适合调试。多卡时通常由 `torchrun` 自动分配。
    # 推荐值：两卡训练时不手写 `--device`，直接用 `torchrun` + `CUDA_VISIBLE_DEVICES=0,2`。
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")

    # 参数名：混合精度类型。
    # 作用：指定混合精度类型。
    # 范围：【bfloat16，float16】。
    # 影响：`bfloat16` 通常更稳；`float16` 更依赖 GradScaler，数值更容易不稳定。
    # 推荐值：`bfloat16`。
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")

    # 参数名：数据加载线程数。
    # 作用：控制 DataLoader 并行加载数据的 worker 数。
    # 范围：【0，1，2，4，8，16】。
    # 影响：越大数据准备通常更快，但更占 CPU 和内存；越小更省系统资源，但 GPU 可能等数据。
    # 推荐值：`4`；如果 CPU 很空闲且数据加载跟不上，再试 `8`。如果gpu利用率不高，cpu很闲，就升高
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 参数名：梯度累积步数。
    # 作用：控制累积多少个 micro-batch 后再更新一次参数。模型不是看一小批数据就立刻更新一次参数，而是先攒几批，再一起更新。一次“真正动刀改参数”之前，先听多少批样本的意见
    # 范围：【1，2，4，8，16，32】。
    # 影响：
    # 【越大】越省显存、等效 batch 越大，但一次真正更新更慢；
    # 【越小】更新更频繁，但显存压力更高。曲线也会越抖
    # 推荐值：`8`；如果显存非常紧张可升到 `16`，如果显存很宽裕可降到 `4`。
    # 补充：等效 batch 约为 `batch_size * accumulation_steps * GPU卡数`。
    # 更新参数的频次，越大不是效果越好，而是会稳定，训练的时候不会抖
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")

    # 参数名：梯度裁剪阈值。
    # 作用：限制梯度范数上限，防止梯度爆炸。
    # 范围：【0.1，0.5，1.0，2.0，5.0】。
    # 影响：越大限制越弱，训练更自由但更容易不稳定；越小限制越强，更稳但可能抑制学习。
    # 推荐值：`1.0`。
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 参数名：日志打印间隔。
    # 作用：控制每隔多少个 step 打印一次训练日志。
    # 范围：【1，5，10，20，50，100，200】。
    # 影响：越小日志越密，便于观察；越大控制台更干净，但排查问题反馈更慢。
    # 推荐值：`20`；长时间后台跑可以用 `50`。
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")

    # 参数名：模型保存间隔。
    # 作用：控制每隔多少个 step 保存一次模型和检查点。
    # 范围：【50，100，200，500，1000，2000】。
    # 影响：越小保存越频繁，更安全但更耗磁盘和 I/O；越大更省资源，但中断时丢失进度更多。
    # 推荐值：`500`；如果你担心中断，可改成 `200`。
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 参数名：隐藏层维度。
    # 作用：控制模型隐藏层维度，决定每层表示宽度。
    # 范围：【256，384，512，640，768，1024】。
    # 影响：越大模型容量更强，但参数量、显存和计算量也更高；越小更轻量，但能力上限更低。
    # 推荐值：`768`，保持作者主线配置不动。
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")

    # 参数名：隐藏层数量。
    # 作用：控制 Transformer 堆叠层数。
    # 范围：【4，6，8，10，12，16，24】。
    # 影响：越大模型更深、能力通常更强，但训练更慢更吃显存；越小更快，但容量更低。
    # 推荐值：`8`，保持作者主线配置不动。
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")

    # 参数名：最大序列长度。
    # 作用：控制单条样本训练时最多保留多少个 token。即：【训练时，模型一次最多看多长的一段文字】
    # 范围：【128，256，340，512，768，1024】。
    # 影响：影响到模型学习的时候看多长
    # 【越大】能保留更长上下文，但显存和计算量明显上升，能学到长上下文里的前后关系；
    # 【越小】更省资源，但长文本截断更多，对长上下文学习不充分。
    # 推荐值：`340`；如果你只求更稳更省显存可降到 `256`，显存很宽裕再试 `512`。
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")

    # 参数名：是否启用 MoE。
    # 作用：控制是否启用 MoE 结构。
    # 范围：【0，1】；`0`=关闭，`1`=开启。
    # 影响：取 `1` 时模型容量更大，但训练更复杂、更慢、更依赖显存；取 `0` 时是普通 Dense 结构，更稳更简单。
    # 推荐值：`0`。
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 参数名：预训练数据路径。
    # 作用：指定预训练数据路径。
    # 范围：【任意符合 `PretrainDataset` 格式的数据文件路径】。
    # 影响：数据集不同会直接影响训练内容、训练时长和最终效果；数据越大通常训练越久。
    # 推荐值：`../dataset/pretrain_t2t_mini.jsonl`；你当前先跑通和做对照更合适。
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")

    # 参数名：起始权重来源。
    # 作用：控制是否从已有权重开始继续训练。
    # 范围：【none，已有权重前缀字符串】。
    # 影响：`none` 表示从头训练；指定已有权重会在此基础上继续学，但前提是结构匹配。
    # 推荐值：`none`。
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")

    # 参数名：是否断点续训。
    # 作用：控制是否自动从 checkpoint 断点续训。
    # 范围：【0，1】；`0`=否，`1`=是。
    # 影响：取 `1` 时会恢复模型、优化器、scaler 和 step；取 `0` 时每次都当新任务启动。
    # 推荐值：首次训练用 `0`；中断后恢复时改成 `1`。
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # 参数名：是否开启实验监控。
    # 作用：控制是否开启实验监控记录。
    # 范围：【不传，传入该 flag】；不传=关闭，传入 `--use_wandb`=开启。
    # 影响：开启后会把 loss、学习率等记录到 swanlab/wandb；关闭则只打印本地日志。
    # 推荐值：开启，也就是命令里带上 `--use_wandb`。
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")

    # 参数名：实验监控项目名。
    # 作用：指定实验监控平台上的项目名。
    # 范围：【任意字符串】。
    # 影响：不影响训练效果，只影响 swanlab/wandb 中的项目归类。
    # 推荐值：`MiniMind-Pretrain`；如果你想区分机器，可写成 `MiniMind-Pretrain-2x3090`。
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")

    # 参数名：是否启用编译加速。
    # 作用：控制是否启用 `torch.compile` 加速。
    # 范围：【0，1】；`0`=关闭，`1`=开启。
    # 影响：取 `1` 可能提升训练速度，但首次编译更慢，也可能有兼容性问题；取 `0` 更稳，适合先跑通。
    # 推荐值：`0`，先不要开。
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
