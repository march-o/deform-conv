# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional, Sequence

from mmengine.dist import master_only
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmengine.hooks.logger_hook import LoggerHook
import pandas as pd


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

    def after_val_epoch(
        self, runner, metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), "val"
        )

        for t, val in tag.items():
            if t in self.dont_log:
                continue
            self.task_logger.report_scalar(t, "valid", val, runner.epoch + 1)

        if self.log_metric_by_epoch:
            # Accessing the epoch attribute of the runner will trigger
            # the construction of the train_loop. Therefore, to avoid
            # triggering the construction of the train_loop during
            # validation, check before accessing the epoch.
            if isinstance(runner._train_loop, dict) or runner._train_loop is None:
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(tag, step=epoch, file_path=self.json_log_path)
        else:
            if isinstance(runner._train_loop, dict) or runner._train_loop is None:
                iter = 0
            else:
                iter = runner.iter
            runner.visualizer.add_scalars(tag, step=iter, file_path=self.json_log_path)

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        super().after_test_epoch(runner, metrics)
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.test_dataloader), 'test', with_non_scalar=True)
        
        df = pd.DataFrame(tag)
        self.task_logger.report_table('test', df, iteration=runner.epoch + 1)
