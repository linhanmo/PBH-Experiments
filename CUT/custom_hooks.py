import os
import time
import torch
from mmcv.runner import HOOKS, Hook
from torch.cuda import Event


@HOOKS.register_module()
class KeepMultiplesOfTenCheckpointHook(Hook):
    """
    自定义 Hook：在每 10 个 Epoch 验证结束后，清理前面的 9 个 epoch 的权重。
    例如：第 30 个 epoch 结束后，删除 epoch_21.pth 到 epoch_29.pth，仅保留 epoch_30.pth。
    """
    def __init__(self, keep_interval=10):
        self.keep_interval = keep_interval

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch % self.keep_interval == 0:
            work_dir = runner.work_dir
            deleted_count = 0
            for i in range(1, self.keep_interval):
                epoch_to_delete = current_epoch - i
                ckpt_path = os.path.join(work_dir, f'epoch_{epoch_to_delete}.pth')
                if os.path.exists(ckpt_path):
                    try:
                        os.remove(ckpt_path)
                        deleted_count += 1
                    except Exception as e:
                        runner.logger.warning(f'Failed to delete checkpoint {ckpt_path}: {e}')


@HOOKS.register_module()
class EarlyStopByAPHook(Hook):
    """
    早停 Hook：当模型性能不再提升时自动停止训练。
    基于验证 AP，如果连续多个验证周期 AP 提升不超过 min_delta，则停止训练。
    """
    def __init__(self, baseline_ap=0.7728, target_loss_pct=2.0, min_delta=0.001, patience=2, interval=10, monitor='AP'):
        self.baseline_ap = baseline_ap
        self.target_loss_pct = target_loss_pct
        self.min_delta = min_delta
        self.patience = patience
        self.interval = interval
        self.monitor = monitor
        self.wait_count = 0
        self.best_ap = 0.0

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch % self.interval != 0:
            return

        if self.monitor not in runner.log_buffer:
            return

        ap_value = runner.log_buffer.get_last(self.monitor)
        if ap_value is None:
            return

        target_ap = self.baseline_ap * (1.0 - self.target_loss_pct / 100.0)
        if ap_value >= target_ap:
            runner.logger.info(f'EarlyStopByAPHook: AP {ap_value:.4f} >= target AP {target_ap:.4f}, target reached, stopping early.')
            runner._stop = True
            return

        if ap_value > self.best_ap + self.min_delta:
            self.best_ap = ap_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                runner.logger.info(f'EarlyStopByAPHook: AP did not improve for {self.wait_count} validation periods. Stopping training.')
                runner._stop = True


@HOOKS.register_module()
class FPSBenchmarkHook(Hook):
    """
    FPS 基准测试 Hook：在验证时测量模型推理速度。
    """
    def __init__(self, interval=10, num_warmup=2, num_iters=10, log_key='fps_val'):
        self.interval = interval
        self.num_warmup = num_warmup
        self.num_iters = num_iters
        self.log_key = log_key

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch % self.interval != 0:
            return

        runner.logger.info('FPSBenchmarkHook: Starting FPS measurement...')

        model = runner.model
        model.eval()

        try:
            data_batch = next(iter(runner.val_loader))
            if isinstance(data_batch, dict):
                inputs = data_batch.get('img', None)
                if inputs is None:
                    inputs = data_batch.get('inputs', None)
            else:
                inputs = data_batch[0] if isinstance(data_batch, (list, tuple)) else data_batch

            if hasattr(inputs, 'cuda'):
                inputs = inputs.cuda()
            elif isinstance(inputs, (list, tuple)):
                inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]

            for _ in range(self.num_warmup):
                with torch.no_grad():
                    model(return_loss=False, return_preds=True, **data_batch)

            starter = Event(enable_timing=True)
            ender = Event(enable_timing=True)

            starter.record()
            for _ in range(self.num_iters):
                with torch.no_grad():
                    model(return_loss=False, return_preds=True, **data_batch)
            ender.record()

            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) / 1000.0
            fps = self.num_iters / elapsed_time

            runner.log_buffer.update({self.log_key: fps})
            runner.logger.info(f'FPSBenchmarkHook: {self.log_key} = {fps:.4f}')

        except Exception as e:
            runner.logger.warning(f'FPSBenchmarkHook: Failed to measure FPS: {e}')

        finally:
            model.train()