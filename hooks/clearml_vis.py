from typing import List, Optional, Tuple, Union
import os
import os.path as osp
import numpy as np
import torch

from mmengine.visualization import BaseVisBackend
from mmengine.visualization.vis_backend import force_init_env
from mmengine.config import Config
from mmengine.hooks.logger_hook import SUFFIX_TYPE
from mmengine.registry import VISBACKENDS
from mmengine.utils import scandir


@VISBACKENDS.register_module()
class CleamlBackend(BaseVisBackend):
    """Clearml visualization backend class. It requires `clearml`_ to be
    installed.

    Examples:
        >>> from mmengine.visualization import ClearMLVisBackend
        >>> from mmengine import Config
        >>> import numpy as np
        >>> vis_backend = ClearMLVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img.png', img)
        >>> vis_backend.add_scalar('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': 0.1,'acc':0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): Useless parameter. Just for
            interface unification. Defaults to None.
        init_kwargs (dict, optional): A dict contains the arguments of
            ``clearml.Task.init`` . See `taskinit`_  for more details.
            Defaults to None
        artifact_suffix (Tuple[str] or str): The artifact suffix.
            Defaults to ('.py', 'pth').

    .. _clearml:
        https://clear.ml/docs/latest/docs/

    .. _taskinit:
        https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        init_kwargs: Optional[dict] = None,
        artifact_suffix: SUFFIX_TYPE = (".py", ".pth"),
    ):
        super().__init__(save_dir)  # type: ignore
        self._init_kwargs = init_kwargs
        self._artifact_suffix = artifact_suffix

    def _init_env(self) -> None:
        try:
            import clearml
        except ImportError:
            raise ImportError('Please run "pip install clearml" to install clearml')

        task_kwargs = self._init_kwargs or {}
        self._clearml = clearml
        self._task = self._clearml.Task.init(**task_kwargs)
        self._logger: clearml.Logger = self._task.get_logger()

    @property  # type: ignore
    @force_init_env
    def experiment(self):
        """Return clearml object."""
        return self._clearml

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to clearml.

        Args:
            config (Config): The Config object
        """
        self.cfg = config
        self._task.connect_configuration(config.to_dict())

    @force_init_env
    def add_image(self, name: str, image: np.ndarray, step: int = 0, **kwargs) -> None:
        """Record the image to clearml.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Defaults to 0.
        """
        self._logger.report_image(title=name, series=name, iteration=step, image=image)

    @force_init_env
    def add_scalar(
        self,
        name: str,
        value: Union[int, float, torch.Tensor, np.ndarray],
        step: int = 0,
        **kwargs
    ) -> None:
        """Record the scalar data to clearml.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        self._logger.report_scalar(title=name, series=name, value=value, iteration=step)

    @force_init_env
    def add_scalars(
        self,
        scalar_dict: dict,
        step: int = 0,
        file_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Record the scalar's data to clearml.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        assert "step" not in scalar_dict, (
            "Please set it directly " "through the step parameter"
        )
        for key, value in scalar_dict.items():

            keys = key.split("/")
            if len(keys) == 1:
                title = key
                series = key
            else:
                title = "/".join(keys[:-1])
                series = keys[-1]

            self._logger.report_scalar(
                title=title, series=series, value=value, iteration=step
            )

    def close(self) -> None:
        """Close the clearml."""
        if not hasattr(self, "_clearml"):
            return

        file_paths: List[str] = list()
        if hasattr(self, "cfg") and osp.isdir(getattr(self.cfg, "work_dir", "")):
            for filename in scandir(self.cfg.work_dir, self._artifact_suffix, False):
                file_path = osp.join(self.cfg.work_dir, filename)
                file_paths.append(file_path)

        for file_path in file_paths:
            self._task.upload_artifact(os.path.basename(file_path), file_path)
        self._task.close()
