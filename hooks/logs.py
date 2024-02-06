# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional, Sequence

from mmengine.dist import master_only
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmengine.hooks.logger_hook import LoggerHook
import pandas as pd
import clearml


@HOOKS.register_module()
class MyLogger(LoggerHook):
    def import_clearml(self):
        try:
            import clearml
        except ImportError:
            raise ImportError('Please run "pip install clearml" to install clearml')
        self.clearml = clearml

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        import clearml

        self.task: clearml.Task = clearml.Task.current_task()

        task_kwargs = self.init_kwargs if self.init_kwargs else {}
        self.task = self.clearml.Task.init(**task_kwargs)
        self.task_logger: clearml.Logger = self.task.get_logger()
