import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    # save_mode 三态：False（不落盘）/ True（仅单轮 best 'checkpoint'）/
    # 'top3'（落盘 checkpoint / checkpoint_ema / checkpoint_last 三份）
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, logger=None,
                 es_mode='single', ema_alpha=0.5):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode
        self.logger = logger
        self.es_mode = es_mode
        self.ema_alpha = ema_alpha
        self.ema_val_loss = None
        self.ema_history = []
        self.best_single_epoch = None
        self.best_ema_epoch = None

    def _log(self, message):
        if self.logger is not None:
            self.logger(message)
        elif self.accelerator is None:
            print(message)
        else:
            self.accelerator.print(message)

    def __call__(self, val_loss, model, path, epoch=None):
        if self.es_mode == 'ema':
            self.ema_history.append(val_loss)
            if self.ema_val_loss is None:
                self.ema_val_loss = val_loss
            else:
                self.ema_val_loss = self.ema_alpha * val_loss + (1.0 - self.ema_alpha) * self.ema_val_loss
            if self.save_mode == 'top3' and val_loss < self.val_loss_min:
                self.save_checkpoint(val_loss, model, path, tag='checkpoint')
                self.best_single_epoch = epoch
            score = -self.ema_val_loss
            if self.best_score is None:
                self.best_score = score
                self.best_ema_epoch = epoch
                if self.save_mode:
                    self.save_checkpoint(val_loss, model, path, tag='checkpoint_ema')
            elif score < self.best_score + self.delta:
                self.counter += 1
                self._log(
                    f'EarlyStopping counter: {self.counter} out of {self.patience} '
                    f'(ema_val={self.ema_val_loss:.6f})'
                )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_ema_epoch = epoch
                if self.save_mode:
                    self.save_checkpoint(val_loss, model, path, tag='checkpoint_ema')
                self.counter = 0
            if self.save_mode == 'top3':
                self._save_raw(model, path, 'checkpoint_last')
            return self.ema_val_loss

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self._log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0
        return None

    def _save_raw(self, model, path, tag):
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
        torch.save(model.state_dict(), path + '/' + tag)

    def save_checkpoint(self, val_loss, model, path, tag='checkpoint'):
        if self.verbose:
            self._log(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self._save_raw(model, path, tag)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, show_progress=True):
    total_loss = []
    total_mae_loss = []
    total_samples = 0
    model.eval()
    with torch.no_grad():
        progress = tqdm(
            vali_loader,
            disable=(not show_progress) or (not accelerator.is_local_main_process),
            leave=False
        )
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(progress):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            # 与训练循环（run_main.py 的 accelerator.autocast()）同源：
            # --mixed_precision bf16 下即 bf16，消除原 fp16 autocast 的口径差
            if args.use_amp:
                with accelerator.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            # 按样本数加权（drop_last=False 后尾部小 batch 不能等权平均）
            n_samples = pred.shape[0]
            total_loss.append(loss.item() * n_samples)
            total_mae_loss.append(mae_loss.item() * n_samples)
            total_samples += n_samples

    total_loss = np.sum(total_loss) / total_samples
    total_mae_loss = np.sum(total_mae_loss) / total_samples

    model.train()
    return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss
