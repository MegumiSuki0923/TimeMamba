import argparse
import json
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeMamba

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os

# 1. 针对 4090D 优化的显存管理设置
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 禁用代理设置以防止连接被拒绝
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
            'all_proxy', 'ALL_PROXY', 'socks_proxy', 'SOCKS_PROXY']:
    if key in os.environ:
        del os.environ[key]

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali

# 设置为 false，禁止 tokenizers 自身的并行化，避免与 PyTorch 的多进程冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='TimeMamba')

def str2bool(v):
    """argparse 显式布尔值解析：支持 True/False/yes/no/1/0 等写法"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 基础配置 ---
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix')
parser.add_argument('--model', type=str, required=True, default='TimeMamba', help='model name')
parser.add_argument('--seed', type=int, default=2025, help='random seed')

# --- 数据加载 ---
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='M:multivariate, S:univariate, MS:mixed')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h', help='freq for time features')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints location')
parser.add_argument('--save_checkpoint', type=str2bool, nargs='?', const=True, default=False,
                    help='是否在磁盘保存最优权重；默认 False（不创建目录、不写任何 checkpoint 文件，最优权重仅保留在内存中供 save_forecast_vis 使用）；'
                         '传入 --save_checkpoint 或 --save_checkpoint True 开启保存')

# --- 预测任务参数 ---
parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# --- 模型定义 ---
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='output attention')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--small_patch', type=int, default=8, help='small branch patch length')
parser.add_argument('--small_stride', type=int, default=4, help='small branch stride')
parser.add_argument('--large_patch', type=int, default=24, help='large branch patch length')
parser.add_argument('--large_stride', type=int, default=4, help='large branch stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='MAMBA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default=2560, help='Mamba-2.8B: 2560')
parser.add_argument('--llm_layers', type=int, default=24, help='Mamba-130m: 24 layers')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--des', type=str, default='Exp', help='exp description')

# --- 优化参数 ---
parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Adam weight decay')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use mixed precision', default=True)
parser.add_argument('--clip_grad', type=float, default=1.0, help='maximum gradient norm for clipping, set to 0 to disable')
parser.add_argument('--result_json', type=str, default='', help='optional path to save structured run metrics')
parser.add_argument('--save_forecast_vis', action='store_true', default=False, help='save last-window forecast CSV after training')

args = parser.parse_args()
set_random_seed(args.seed)

# 记录脚本开始时间
script_start_time = time.time()

if args.batch_size <= 0:
    raise ValueError(f"[TimeMamba] batch_size must be positive, got {args.batch_size}")

# 初始化 Accelerator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def format_time(seconds):
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    if m > 0:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"


run_summaries = []

for ii in range(args.itr):
    run_start_time = time.time()
    # Generate timestamp for the checkpoint name
    current_time = time.strftime('%Y%m%d%H%M%S')
    
    # 命名格式: datasetname_seqlen_pred_len_comment_time
    setting_name = f"{args.data}_{args.seq_len}_{args.pred_len}_{args.model_comment}_{current_time}"

    config_message = (
        f"[Config] model={args.model} data={args.data} "
        f"seq_len={args.seq_len} pred_len={args.pred_len} batch_size={args.batch_size} "
        f"early_stop=val_loss patience={args.patience}"
    )
    accelerator.print(config_message)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    accelerator.print(
        f"[Data] train={len(train_data)} val={len(vali_data)} test={len(test_data)} "
        f"train_steps={len(train_loader)} val_steps={len(vali_loader)} test_steps={len(test_loader)}"
    )
    best_metrics = {
        'epoch': 0,
        'train_loss': None,
        'val_loss': float('inf'),
        'test_loss': None,
        'test_mae_loss': None,
    }
    run_status = 'completed'
    nonfinite_reason = None

    # 3. 构建 TimeMamba 模型
    model = TimeMamba.Model(args)
    if hasattr(model, 'llm_model'):
        model.llm_model = model.llm_model.to(accelerator.device)
        if any(True for _ in model.llm_model.parameters()):
            accelerator.print(f"[Diag] llm_model device: {next(model.llm_model.parameters()).device}")

    path = os.path.join(args.checkpoints, setting_name)
    # save_mode=False 时 EarlyStopping 不写磁盘；最优权重保留在内存中（供 save_forecast_vis 恢复）
    if args.save_checkpoint and not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    # 4. 仅收集需要训练的参数（在 accelerator.prepare 包装前记录原始参数名，用于内存恢复）
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    best_trainable_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    best_state_cpu = None
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lradj == 'COS':
        # CosineAnnealingLR 按 epoch 调度（见训练循环末尾），T_max 取实际训练轮数
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.train_epochs, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=len(train_loader),
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    # =========================================================
    # 初始化 EarlyStopping
    # =========================================================
    early_stopping = EarlyStopping(
        accelerator=accelerator,
        patience=args.patience,
        save_mode=args.save_checkpoint,
    )

    # 记录每个 epoch 的耗时，用于计算预计剩余时间
    epoch_times = []
    epochs_completed = 0

    for epoch in range(args.train_epochs):
        train_loss = []
        model.train()
        epoch_time = time.time()
        epoch_aborted = False
        
        t_data_start = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            t_data_end = time.time()
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            # 混合精度计算
            with accelerator.autocast():
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                loss = criterion(outputs, batch_y)

            train_loss.append(loss.item())

            accelerator.backward(loss)

            # 防止单点梯度 NaN 或 Inf 感染全局 clip_grad_norm
            if accelerator.sync_gradients:
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
                if args.clip_grad > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                
            model_optim.step()

            # 学习率统一在 epoch 末尾调度：COS → 余弦退火；type1 等 → adjust_learning_rate 阶梯衰减。
            if epoch == 0 and i < 10:
                torch.cuda.synchronize()
            t_compute_end = time.time()
            if epoch == 0 and i < 10:
                accelerator.print(f"[Profile] Step {i} | Data Load: {(t_data_end - t_data_start)*1000:.1f} ms | Compute: {(t_compute_end - t_data_end)*1000:.1f} ms")
            
            # 重置计时起点，用于下一次 DataLoader 取数据的耗时计算
            t_data_start = time.time()

        if epoch_aborted:
            break

        epoch_cost_time = time.time() - epoch_time
        epoch_times.append(epoch_cost_time)

        # =========================================================
        # 学习率调度（参考 Time-LLM run_main.py）
        # - COS：按 epoch 步进余弦退火
        # - type1 等：adjust_learning_rate 阶梯衰减（每轮减半），直接写 optimizer 的 lr
        # =========================================================
        if args.lradj == 'COS':
            scheduler.step()
            accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        elif args.lradj != 'TST':
            if epoch == 0:
                args.learning_rate = model_optim.param_groups[0]['lr']
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
        
        # 计算预计剩余时间
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = args.train_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        train_loss = np.average(train_loss)
        vali_loss, _ = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        epochs_completed = epoch + 1

        if not np.isfinite(train_loss) or not np.isfinite(vali_loss):
            nonfinite_reason = f"nonfinite_eval_metrics epoch={epoch + 1}"
            run_status = 'nonfinite_eval_metrics'
            message = (
                f"[Abort] {nonfinite_reason} "
                f"train={train_loss} val={vali_loss}"
            )
            accelerator.print(message)
            break

        is_best_epoch = vali_loss < best_metrics['val_loss']
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        if not np.isfinite(test_loss) or not np.isfinite(test_mae_loss):
            nonfinite_reason = f"nonfinite_test_metrics epoch={epoch + 1}"
            run_status = 'nonfinite_test_metrics'
            message = (
                f"[Abort] {nonfinite_reason} "
                f"test={test_loss} mae={test_mae_loss}"
            )
            accelerator.print(message)
            break

        cost_message = "Epoch: {}/{} | Cost: {} | ETA: {}".format(
            epoch + 1,
            args.train_epochs,
            format_time(epoch_cost_time),
            format_time(estimated_remaining),
        )
        metric_message = "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss
        )
        accelerator.print(cost_message)
        accelerator.print(metric_message)

        if is_best_epoch:
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(vali_loss),
                'test_loss': float(test_loss),
                'test_mae_loss': float(test_mae_loss),
            }
            if not args.save_checkpoint:
                # 不落盘模式：仅在内存中保留最优权重的可训练参数副本，供 save_forecast_vis 恢复
                best_state_cpu = {
                    k: v.detach().clone().cpu()
                    for k, v in accelerator.unwrap_model(model).state_dict().items()
                    if k in best_trainable_keys
                }

        # =========================================================
        # 早停判断逻辑
        # =========================================================
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            run_status = 'early_stopped'
            break


    if args.save_forecast_vis:
        # 训练结束：恢复最优权重，遍历测试集，保存最后测试窗口 CSV。
        # save_checkpoint 开启时从磁盘读取；关闭时使用训练期间保留的内存副本（不落盘）。
        best_model_path = os.path.join(path, 'checkpoint')
        if args.save_checkpoint and os.path.exists(best_model_path):
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(
                torch.load(best_model_path, map_location=accelerator.device)
            )
        elif (not args.save_checkpoint) and best_state_cpu:
            # 只恢复可训练参数；冻结主干权重在训练中从未变化，无需恢复。
            unwrapped_model = accelerator.unwrap_model(model)
            missing, unexpected = unwrapped_model.load_state_dict(best_state_cpu, strict=False)
            accelerator.print(
                f"[Vis] 已从内存恢复 best-val 权重（{len(best_state_cpu)} 个可训练张量），"
                f"unexpected={len(unexpected)}"
            )
            del best_state_cpu

        model.eval()
        total_test_batches = len(test_loader)
        last_window_data = None

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :args.label_len, :], dec_inp], dim=1
                ).float().to(accelerator.device)

                with accelerator.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if i == total_test_batches - 1:
                    f_dim = -1 if args.features == 'MS' else 0
                    pred_slice = outputs[:, -args.pred_len:, f_dim:]
                    true_slice = batch_y[:, -args.pred_len:, f_dim:]
                    last_window_data = (
                        batch_x[-1].cpu().numpy(),
                        true_slice[-1].cpu().numpy(),
                        pred_slice[-1].cpu().detach().numpy()
                    )

        if accelerator.is_local_main_process and last_window_data is not None:
            hist_arr, fut_true_arr, fut_pred_arr = last_window_data
            hist_vals = hist_arr[:, 0]
            true_vals = fut_true_arr[:, 0]
            pred_vals = fut_pred_arr[:, 0]

            vis_dir = './forecast_vis'
            os.makedirs(vis_dir, exist_ok=True)
            csv_path = os.path.join(vis_dir, setting_name + '.csv')

            with open(csv_path, 'w') as csv_f:
                csv_f.write('time_step,ground_truth,prediction\n')
                for t, v in enumerate(hist_vals):
                    csv_f.write(f'{t},{v:.6f},\n')
                offset = len(hist_vals)
                for t, (tv, pv) in enumerate(zip(true_vals, pred_vals)):
                    csv_f.write(f'{offset + t},{tv:.6f},{pv:.6f}\n')

            msg = f"[Vis] 最后测试窗口已保存: {csv_path}"
            accelerator.print(msg)

        model.train()

    run_total_time = time.time() - run_start_time
    run_summaries.append({
        'run_index': ii + 1,
        'setting_name': setting_name,
        'best_epoch': best_metrics['epoch'],
        'best_train_loss': best_metrics['train_loss'],
        'best_val_loss': best_metrics['val_loss'],
        'best_test_loss': best_metrics['test_loss'],
        'best_test_mae_loss': best_metrics['test_mae_loss'],
        'epochs_completed': epochs_completed,
        'run_total_time_sec': float(run_total_time),
        'status': run_status,
        'nonfinite_reason': nonfinite_reason,
    })
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Run total time: {format_time(run_total_time)}")
    accelerator.print(f"{'=' * 60}")

accelerator.wait_for_everyone()

# 打印脚本总耗时
script_total_time = time.time() - script_start_time
hours, remainder = divmod(script_total_time, 3600)
minutes, seconds = divmod(remainder, 60)
accelerator.print(f"\n{'='*60}")
accelerator.print(f"脚本总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
accelerator.print(f"{'='*60}")

if args.result_json and accelerator.is_local_main_process:
    result_dir = os.path.dirname(args.result_json)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    result_payload = {
        'model': args.model,
        'data': args.data,
        'objective': 'minimize_best_val_loss',
        'script_total_time_sec': float(script_total_time),
        'args': vars(args),
        'runs': run_summaries,
    }
    if run_summaries:
        result_payload['best_run'] = min(run_summaries, key=lambda item: item['best_val_loss'])
    with open(args.result_json, 'w', encoding='utf-8') as result_file:
        json.dump(result_payload, result_file, ensure_ascii=False, indent=2)
