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
from dataset.lm_dataset import SFTDataset
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
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
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
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
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
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")

    # 参数名：模型保存目录。
    # 作用：保存最终 SFT 权重的目录。
    # 范围：【任意可写目录路径】。
    # 影响：不影响训练效果，只影响权重保存位置；目录不可写会导致保存失败。
    # 推荐值：`../out`。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")

    # 参数名：保存权重前缀名。
    # 作用：控制输出权重文件名的前缀。
    # 范围：【任意字符串】。
    # 影响：不影响训练效果，只影响文件命名和实验区分。
    # 推荐值：`full_sft`。
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")

    # 参数名：训练轮数。
    # 作用：控制 SFT 数据集重复训练多少轮。
    # 范围：【1，2，3，4，5，...】整数。
    # 影响：越大训练越久、拟合指令风格更充分，但也更容易过拟合；越小训练更快，但可能对齐不够。
    # 推荐值：`2`；先做小实验可用 `1`。
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")

    # 参数名：单步批大小。
    # 作用：控制单次送入模型的 micro-batch 大小。SFT 时每次让模型同时看多少条指令样本。
    # 范围：【1，2，4，8，16，32，64】。
    # 影响：
    # 【越大】显存占用越高、吞吐通常越高、梯度更稳定；
    # 【越小】更省显存，但训练可能更慢、波动更大。
    # 推荐值：两张 `3090` 建议先用 `16`；显存很宽裕再试 `32`。
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    # 参数名：学习率。
    # 作用：控制 SFT 阶段参数更新步长。
    # 范围：【1e-6，2e-6，5e-6，1e-5，2e-5，5e-5，1e-4】。
    # 影响：越大学得越快，但更容易把预训练好的基础能力“冲坏”；越小更稳，但收敛更慢。
    # 推荐值：`1e-5`；先不要轻易加大。
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")

    # 参数名：训练设备。
    # 作用：指定训练运行设备。
    # 范围：【cpu，cuda:0，cuda:1，...】。
    # 影响：选 GPU 训练会快很多；选 CPU 只适合调试。多卡时通常由 `torchrun` 自动分配。
    # 推荐值：两卡训练时不手写 `--device`，直接用 `torchrun` + `CUDA_VISIBLE_DEVICES=0,1`。
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")

    # 参数名：混合精度类型。
    # 作用：指定混合精度训练的数据类型。
    # 范围：【bfloat16，float16】。
    # 影响：`bfloat16` 通常更稳；`float16` 兼容性更广，但更依赖 GradScaler。
    # 推荐值：`bfloat16`。
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")

    # 参数名：数据加载线程数。
    # 作用：控制 DataLoader 并行加载 SFT 数据的 worker 数量。
    # 范围：【0，1，2，4，8，16】。
    # 影响：越大数据准备通常更快，但更占 CPU 和内存；越小更省系统资源，但 GPU 可能等数据。
    # 推荐值：`4`；如果 CPU 很空闲且数据加载跟不上，再试 `8`。
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 参数名：梯度累积步数。
    # 作用：控制累积多少个 micro-batch 后再更新一次参数。
    # 范围：【1，2，4，8，16，32】。
    # 影响：越大越省显存、等效 batch 越大，但一次真正更新更慢；越小更新更频繁，但显存压力更高。
    # 推荐值：`4` 或 `8`；两张 `3090` 上若 `batch_size=16`，可先用 `4`。
    # 补充：等效 batch 约为 `batch_size * accumulation_steps * GPU卡数`。
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")

    # 参数名：梯度裁剪阈值。
    # 作用：限制梯度范数上限，防止训练中梯度爆炸。
    # 范围：【0.1，0.5，1.0，2.0，5.0】。
    # 影响：越大限制越弱，训练更自由但更容易不稳定；越小限制越强，更稳但可能抑制学习。
    # 推荐值：`1.0`。
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 参数名：日志打印间隔。
    # 作用：控制每隔多少个 step 打印一次训练日志。
    # 范围：【1，5，10，20，50，100，200】。
    # 影响：越小日志越密，便于观察；越大输出更干净，但发现问题更慢。
    # 推荐值：`20`。
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")

    # 参数名：模型保存间隔。
    # 作用：控制每隔多少个 step 保存一次 SFT 权重和 checkpoint。
    # 范围：【50，100，200，500，1000，2000】。
    # 影响：越小保存越频繁，更安全但更耗磁盘和 I/O；越大更省资源，但中断时丢失进度更多。
    # 推荐值：`500`；如果你担心中断，可改成 `200`。
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 参数名：隐藏层维度。
    # 作用：控制模型每层隐藏表示宽度。
    # 范围：【256，384，512，640，768，1024】。
    # 影响：越大模型容量更强，但参数量、显存和计算量也更高；越小更轻量，但能力上限更低。
    # 推荐值：`768`，保持和预训练模型一致。
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")

    # 参数名：隐藏层数量。
    # 作用：控制 Transformer 堆叠层数。
    # 范围：【4，6，8，10，12，16，24】。
    # 影响：越大模型更深、能力通常更强，但训练更慢更吃显存；越小更快，但容量更低。
    # 推荐值：`8`，保持和预训练模型一致。
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")

    # 参数名：最大序列长度。
    # 作用：控制单条 SFT 样本训练时最多保留多少个 token。
    # 范围：【128，256，340，512，768，1024】。
    # 影响：越大能保留更完整的多轮对话与指令上下文，但显存和计算量明显上升；越小更省资源，但长样本截断更多。
    # 推荐值：`768`；如果你更重视稳和快，可降到 `512`。 针对预训练数值变大，因为SFT阶段需要使用多轮对话，需要模型学会更完整的问答结构，所以要输入的长一点
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")

    # 参数名：是否启用 MoE。
    # 作用：控制是否使用 MoE 结构训练。
    # 范围：【0，1】；`0`=关闭，`1`=开启。
    # 影响：取 `1` 时模型容量更大，但训练更复杂、更慢；取 `0` 时是普通 Dense 结构，更稳更简单。
    # 推荐值：`0`。
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # 参数名：训练数据路径。
    # 作用：指定 SFT 训练数据文件路径。
    # 范围：【任意符合 `SFTDataset` 格式的数据文件路径】。
    # 影响：数据集不同会直接影响模型的指令风格、对话表现和训练时长。
    # 推荐值：`../dataset/sft_t2t_mini.jsonl`；当前先快速复现更合适。
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")

    # 参数名：起始权重来源。
    # 作用：指定 SFT 开始训练时加载哪个已有权重。一般是预训练权重
    # 范围：【none，pretrain，其他已有权重前缀字符串】。
    # 影响：`pretrain` 表示在预训练模型上继续做指令微调；`none` 表示从零开始做 SFT，一般不推荐。
    # 推荐值：`pretrain`。
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")

    # 参数名：是否断点续训。
    # 作用：控制是否自动从 SFT checkpoint 断点恢复。
    # 范围：【0，1】；`0`=否，`1`=是。
    # 影响：取 `1` 时会恢复模型、优化器、scaler 和 step；取 `0` 时每次都按新任务启动。
    # 推荐值：首次训练用 `0`；中断后恢复时改成 `1`。
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # 参数名：是否开启实验监控。
    # 作用：控制是否把训练日志记录到 swanlab/wandb。
    # 范围：【不传，传入该 flag】；不传=关闭，传入 `--use_wandb`=开启。
    # 影响：开启后会记录 loss、学习率等信息；关闭则只打印本地日志。
    # 推荐值：开启，也就是命令里带上 `--use_wandb`。
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")

    # 参数名：实验监控项目名。
    # 作用：指定 swanlab/wandb 中的项目归类名称。
    # 范围：【任意字符串】。
    # 影响：不影响训练效果，只影响实验记录归类。
    # 推荐值：`MiniMind-Full-SFT-2x3090` 或保持默认。
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")

    # 参数名：是否启用编译加速。
    # 作用：控制是否使用 `torch.compile` 做图编译加速。
    # 范围：【0，1】；`0`=关闭，`1`=开启。
    # 影响：取 `1` 可能提速，但首轮编译更慢，也可能带来兼容性问题；取 `0` 更稳，更适合先跑通。
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
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
