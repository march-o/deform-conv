# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional

from mmengine.dist import master_only
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmengine.hooks.logger_hook import LoggerHook


@HOOKS.register_module()
class ClearMLLoggerHook(LoggerHook):
    """Class to log metrics with clearml.

    It requires `clearml`_ to be installed.


    Args:
        init_kwargs (dict): A dict contains the `clearml.Task.init`
            initialization keys. See `taskinit`_  for more details.
        interval (int): Logging interval (every k iterations). Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.

    .. _clearml:
        https://clear.ml/docs/latest/docs/
    .. _taskinit:
        https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit
    """

    def __init__(
        self,
        init_kwargs: Optional[Dict] = None,
        interval: int = 10,
        ignore_last: bool = True,
        reset_flag: bool = False,
        by_epoch: bool = True,
    ):
        super().__init__(interval, ignore_last)
        self.import_clearml()
        self.init_kwargs = init_kwargs

        self.dont_log = ["time", "memory", "epoch", "iter", "data_time"]

    def import_clearml(self):
        try:
            import clearml
        except ImportError:
            raise ImportError('Please run "pip install clearml" to install clearml')
        self.clearml = clearml

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        task_kwargs = self.init_kwargs if self.init_kwargs else {}
        self.task = self.clearml.Task.init(**task_kwargs)
        self.task_logger = self.task.get_logger()

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        for tag, val in tags.items():
            self.task_logger.report_scalar(tag, tag, val, self.get_iter(runner))

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch=None,
        outputs: Optional[dict] = None,
    ) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(runner, self.interval_exp_name) or (
            self.end_of_epoch(runner.train_dataloader, batch_idx)
        ):
            exp_info = f"Exp name: {runner.experiment_name}"
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, "train"
            )
        elif self.end_of_epoch(runner.train_dataloader, batch_idx) and (
            not self.ignore_last or len(runner.train_dataloader) <= self.interval
        ):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, "train"
            )
        else:
            return

        for t, val in tag.items():
            if t in self.dont_log:
                continue
            self.task_logger.report_scalar(t, "train", val, runner.iter + 1)
