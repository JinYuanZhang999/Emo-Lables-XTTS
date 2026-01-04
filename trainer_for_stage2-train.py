# -*- coding: utf-8 -*-
import gc
import importlib
import logging
import os
import platform
import shutil
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass, field
from inspect import signature
from typing import Callable, Dict, List, Tuple, Union
from collections import Counter

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader

from trainer.analytics import ping_training_run
from trainer.callbacks import TrainerCallback
from trainer.generic_utils import (
    KeepAverage,
    count_parameters,
    get_experiment_folder_path,
    get_git_branch,
    isimplemented,
    remove_experiment_folder,
    set_partial_state_dict,
    to_cuda,
)
from trainer.io import (
    copy_model_files,
    get_last_checkpoint,
    load_fsspec,
    save_best_model,
    save_checkpoint,
)
from trainer.logging import ConsoleLogger, DummyLogger, logger_factory
from trainer.trainer_utils import (
    get_optimizer,
    get_scheduler,
    is_apex_available,
    print_training_env,
    setup_torch_training_env,
)
from trainer.utils.cuda_memory import cuda_meminfo, should_reduce_batch_size
from trainer.utils.distributed import (
    get_rank,
    init_distributed,
    rank_zero_logger_info,
    rank_zero_only,
)

logger = logging.getLogger("trainer")

if is_apex_available():
    from apex import amp  # pylint: disable=import-error


@dataclass
class TrainerConfig(Coqpit):
    """Config fields tweaking the Trainer for a model.
    A ````ModelConfig```, by inheriting ```TrainerConfig``` must be defined for using 👟.
    Inherit this by a new model config and override the fields as needed.
    All the fields can be overridden from comman-line as ```--coqpit.arg_name=value```.

    Example::

        Run the training code by overriding the ```lr``` and ```plot_step``` fields.

        >>> python train.py --coqpit.plot_step=22 --coqpit.lr=0.001

        Defining a model using ```TrainerConfig```.

        >>> from trainer import TrainerConfig
        >>> class MyModelConfig(TrainerConfig):
        ...     optimizer: str = "Adam"
        ...     lr: float = 0.001
        ...     epochs: int = 1
        ...     ...
        >>> class MyModel(nn.module):
        ...    def __init__(self, config):
        ...        ...
        >>> model = MyModel(MyModelConfig())

    """

    # Fields for the run
    output_path: str = field(default="output")
    logger_uri: str = field(
        default=None,
        metadata={
            "help": "URI to save training artifacts by the logger. If not set, logs will be saved in the output_path. Defaults to None"
        },
    )
    run_name: str = field(default="run", metadata={"help": "Name of the run. Defaults to 'run'"})
    project_name: str = field(default=None, metadata={"help": "Name of the project. Defaults to None"})
    run_description: str = field(
        default="🐸Coqui trainer run.",
        metadata={"help": "Notes and description about the run. Defaults to '🐸Coqui trainer run.'"},
    )
    # Fields for logging
    print_step: int = field(
        default=25, metadata={"help": "Print training stats on the terminal every print_step steps. Defaults to 25"}
    )
    plot_step: int = field(
        default=100, metadata={"help": "Plot training stats on the logger every plot_step steps. Defaults to 100"}
    )
    model_param_stats: bool = field(
        default=False, metadata={"help": "Log model parameters stats on the logger dashboard. Defaults to False"}
    )
    wandb_entity: str = field(default=None, metadata={"help": "Wandb entity to log the run. Defaults to None"})
    dashboard_logger: str = field(
        default="tensorboard", metadata={"help": "Logger to use for the tracking dashboard. Defaults to 'tensorboard'"}
    )
    # Fields for checkpointing
    save_on_interrupt: bool = field(
        default=True, metadata={"help": "Save checkpoint on interrupt (Ctrl+C). Defaults to True"}
    )
    log_model_step: int = field(
        default=None,
        metadata={
            "help": "Save checkpoint to the logger every log_model_step steps. If not defined `save_step == log_model_step`."
        },
    )
    save_step: int = field(
        default=10000, metadata={"help": "Save local checkpoint every save_step steps. Defaults to 10000"}
    )
    save_n_checkpoints: int = field(default=5, metadata={"help": "Keep n local checkpoints. Defaults to 5"})
    save_checkpoints: bool = field(default=True, metadata={"help": "Save checkpoints locally. Defaults to True"})
    save_all_best: bool = field(
        default=False, metadata={"help": "Save all best checkpoints and keep the older ones. Defaults to False"}
    )
    save_best_after: int = field(
        default=0, metadata={"help": "Wait N steps to save best checkpoints. Defaults to 0"}
    )
    target_loss: str = field(
        default=None, metadata={"help": "Target loss name to select the best model. Defaults to None"}
    )
    # Fields for eval and test run
    print_eval: bool = field(default=False, metadata={"help": "Print eval steps on the terminal. Defaults to False"})
    test_delay_epochs: int = field(default=0, metadata={"help": "Wait N epochs before running the test. Defaults to 0"})
    run_eval: bool = field(
        default=True, metadata={"help": "Run evalulation epoch after training epoch. Defaults to True"}
    )
    run_eval_steps: int = field(
        default=None,
        metadata={
            "help": "Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None"
        },
    )
    # Fields for distributed training
    distributed_backend: str = field(
        default="nccl", metadata={"help": "Distributed backend to use. Defaults to 'nccl'"}
    )
    distributed_url: str = field(
        default="tcp://localhost:54321",
        metadata={"help": "Distributed url to use. Defaults to 'tcp://localhost:54321'"},
    )
    # Fields for training specs
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision training. Defaults to False"})
    precision: str = field(
        default="fp16",
        metadata={
            "help": "Precision to use in mixed precision training. `fp16` for float16 and `bf16` for bfloat16. Defaults to 'f16'"
        },
    )
    epochs: int = field(default=1000, metadata={"help": "Number of epochs to train. Defaults to 1000"})
    batch_size: int = field(default=32, metadata={"help": "Batch size to use. Defaults to 32"})
    eval_batch_size: int = field(default=16, metadata={"help": "Batch size to use for eval. Defaults to 16"})
    grad_clip: float = field(
        default=0.0, metadata={"help": "Gradient clipping value. Disabled if <= 0. Defaults to 0.0"}
    )
    scheduler_after_epoch: bool = field(
        default=True,
        metadata={"help": "Step the scheduler after each epoch else step after each iteration. Defaults to True"},
    )
    # Fields for optimzation
    lr: Union[float, List[float]] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: Union[str, List[str]] = field(default=None, metadata={"help": "Optimizer(s) to use. Defaults to None"})
    optimizer_params: Union[Dict, List[Dict]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: Union[str, List[str]] = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: Dict = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    use_grad_scaler: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable gradient scaler explicitly. It is enabled by default with AMP training. Defaults to False"
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={
            "help": "A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs. Default to False."
        },
    )
    cudnn_enable: bool = field(default=True, metadata={"help": "Enable/disable cudnn explicitly. Defaults to True"})
    cudnn_deterministic: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable deterministic cudnn operations. Set this True for reproducibility but it slows down training significantly.  Defaults to False."
        },
    )
    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Enable/disable cudnn benchmark explicitly. Set this False if your input size change constantly. Defaults to False"
        },
    )
    training_seed: int = field(
        default=54321,
        metadata={"help": "Global seed for torch, random and numpy random number generator. Defaults to 54321"},
    )


@dataclass
class TrainerArgs(Coqpit):
    """Trainer arguments that can be accessed from the command line.

    Examples::
        >>> python train.py --restore_path /path/to/checkpoint.pth
    """

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting the best loss. If not specified, the latest best model in continue path is used"
        },
    )
    use_ddp: bool = field(
        default=False,
        metadata={"help": "Use DDP in distributed training. It is to set in `distribute.py`. Do not set manually."},
    )
    use_accelerate: bool = field(default=False, metadata={"help": "Use HF Accelerate as the back end for training."})
    grad_accum_steps: int = field(
        default=1,
        metadata={
            "help": "Number of gradient accumulation steps. It is used to accumulate gradients over multiple batches."
        },
    )
    overfit_batch: bool = field(default=False, metadata={"help": "Overfit a single batch for debugging."})
    skip_train_epoch: bool = field(
        default=False,
        metadata={"help": "Skip training and only run evaluation and test."},
    )
    start_with_eval: bool = field(
        default=False,
        metadata={"help": "Start with evaluation and test."},
    )
    small_run: int = field(
        default=None,
        metadata={
            "help": "Only use a subset of the samples for debugging. Set the number of samples to use. Defaults to None. "
        },
    )
    gpu: int = field(
        default=None, metadata={"help": "GPU ID to use if ```CUDA_VISIBLE_DEVICES``` is not set. Defaults to None."}
    )
    # only for DDP
    rank: int = field(default=0, metadata={"help": "Process rank in a distributed training. Don't set manually."})
    group_id: str = field(
        default="", metadata={"help": "Process group id in a distributed training. Don't set manually."}
    )


class Trainer:
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        args: TrainerArgs,
        config: Coqpit,
        output_path: str,
        c_logger: ConsoleLogger = None,
        dashboard_logger: "Logger" = None,
        model: nn.Module = None,
        get_model: Callable = None,
        get_data_samples: Callable = None,
        train_samples: List = None,
        eval_samples: List = None,
        test_samples: List = None,
        train_loader: DataLoader = None,
        eval_loader: DataLoader = None,
        training_assets: Dict = {},
        parse_command_line_args: bool = True,
        callbacks: Dict[str, Callable] = {},
        gpu: int = None,
    ) -> None:
        """Simple yet powerful 🐸💬 TTS trainer for PyTorch. It can train all the available `tts` and `vocoder` models
        or easily be customized.

        Notes:

            Supports Automatic Mixed Precision training. If `Apex` is availabe, it automatically picks that, else
            it uses PyTorch's native `amp` module. `Apex` may provide more stable training in some cases.

        Args:

            args (Union[Coqpit, Namespace]): Training arguments parsed either from console by `argparse` or `TrainerArgs`
                config object.

            config (Coqpit): Model config object. It includes all the values necessary for initializing, training, evaluating
                and testing the model.

            output_path (str): Path to the output training folder. All the files are saved under thi path.

            c_logger (ConsoleLogger, optional): Console logger for printing training status. If not provided, the default
                console logger is used. Defaults to None.

            dashboard_logger Union[TensorboardLogger, WandbLogger]: Dashboard logger. If not provided, the tensorboard logger is used.
                Defaults to None.

            model (nn.Module, optional): Initialized and ready-to-train model. If it is not defined, `Trainer`
                initializes a model from the provided config. Defaults to None.

            get_model (Callable):
                A function that returns a model. It is used to initialize the model when `model` is not provided.
                It either takes the config as the only argument or does not take any argument.
                Defaults to None

            get_data_samples (Callable):
                A function that returns a list of training and evaluation samples. Used if `train_samples` and
                `eval_samples` are None. Defaults to None.

            train_samples (List):
                A list of training samples used by the model's `get_train_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            eval_samples (List):
                A list of evaluation samples used by the model's `get_eval_data_loader` to init the `dataset` and the
                `data_loader`. Defaults to None.

            train_loader (DataLoader):
                A pytorch data loader object for training epochs. Leave as None if you want it to be made during training. Defaults to None.

            eval_loader (DataLoader):
                A pytorch data loader object for evaluation epochs. Leave as None to be generated during training. Defaults to None.

            test_samples (List):
                A list of test samples used by the model's `get_test_data_loader` to init the `dataset` and the
                `data_loader`. If None, the ```model.test_run()``` is expected to load the data. Defaults to None.

            training_assets (Dict):
                A dictionary of assets to be used at training and passed to the model's ```train_log(), eval_log(), get_data_loader()```
                during training. It can include  `AudioProcessor` or/and `Tokenizer`. Defaults to {}.

            parse_command_line_args (bool):
                If true, parse command-line arguments and update `TrainerArgs` and model `config` values. Set it
                to false if you parse the arguments yourself. Defaults to True.

            callbacks (Dict[str, Callable]):
                A dictionary of callbacks to be used during training. The keys are the callback names and the values

            gpu (int):
                GPU ID to use for training If "CUDA_VISIBLE_DEVICES" is not set. Defaults to None.

        Example::

            Running trainer with a model.

            >>> args = TrainerArgs(...)
            >>> config = ModelConfig(...)
            >>> model = Model(config)
            >>> trainer = Trainer(args, config, output_path, model=model)
            >>> trainer.fit()

            TODO:
                - Wrap model for not calling .module in DDP.
                - Deepspeed integration
                - Profiler integration.
                - Overfitting to a batch.
                - TPU training
        """
        # print("init正在执行")
        if parse_command_line_args:
            # parse command-line arguments to override TrainerArgs()
            args, coqpit_overrides = self.parse_argv(args)

            # get ready for training and parse command-line arguments to override the model config
            config, new_fields = self.init_training(args, coqpit_overrides, config)
        elif args.continue_path or args.restore_path:
            config, new_fields = self.init_training(args, {}, config)
        else:
            new_fields = {}

        # set the output path
        if args.continue_path:
            # use the same path as the continuing run
            output_path = args.continue_path
        else:
            # override the output path if it is provided
            output_path = config.output_path if output_path is None else output_path
            # create a new output folder name
            output_path = get_experiment_folder_path(config.output_path, config.run_name)
            os.makedirs(output_path, exist_ok=True)

        # copy training assets to the output folder
        copy_model_files(config, output_path, new_fields)

        # init class members
        self.args = args
        self.config = config
        self.output_path = output_path
        self.training_assets = training_assets
        self.grad_accum_steps = args.grad_accum_steps
        self.overfit_batch = args.overfit_batch
        self.skip_train_epoch = args.skip_train_epoch
        self.start_with_eval = args.start_with_eval

        assert self.grad_accum_steps > 0, " [!] grad_accum_steps must be greater than 0."

        # setup logging
        log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
        self._setup_logger_config(log_file)

        # setup training environment
        self.use_cuda, self.num_gpus = self.setup_training_environment(args=args, config=config, gpu=gpu)

        # init loggers
        self.dashboard_logger, self.c_logger = self.init_loggers(self.config, output_path, dashboard_logger, c_logger)
        # self.c_logger.logger = logger

        if not self.config.log_model_step:
            self.config.log_model_step = self.config.save_step

        # make sure that start_with_eval is disabled if eval is disabled
        if not self.config.run_eval and self.start_with_eval:
            self.start_with_eval = False

        self.total_steps_done = 0
        self.epochs_done = 0
        self.restore_step = 0
        self.restore_epoch = 0
        self.best_loss = {"train_loss": float("inf"), "eval_loss": float("inf") if self.config.run_eval else None}
        self.train_loader = None
        self.test_loader = None
        self.eval_loader = None

        self.keep_avg_train = None
        self.keep_avg_eval = None

        self.use_amp_scaler = (
            self.use_cuda
            if self.config.mixed_precision and self.config.precision == "fp16"
            else self.config.use_grad_scaler
        )

        if train_samples is not None:
            # use the provided samples
            self.train_samples = train_samples
            self.eval_samples = eval_samples
            self.test_samples = test_samples
        elif get_data_samples is not None:
            # run `get_data_samples` to init the data samples
            (  # pylint: disable=unbalanced-tuple-unpacking
                self.train_samples,
                self.eval_samples,
                self.test_samples,
            ) = self.run_get_data_samples(config, get_data_samples)
        else:
            # expecting to load the samples in `model.get_data_loader()`
            self.train_samples = None
            self.eval_samples = None
            self.test_samples = None

        # define custom train and eval loader
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # only use a subset of the samples if small_run is set
        self.setup_small_run(args.small_run)

        # init the model
        if model is None and get_model is None:
            raise ValueError("[!] `model` and `get_model` cannot both be None.")
        if model is not None:
            self.model = model
        else:
            self.run_get_model(self.config, get_model)

        # init model's training assets
        if isimplemented(self.model, "init_for_training"):
            self.model.init_for_training()

        # setup criterion
        self.criterion = self.get_criterion(self.model)

        # DISTRUBUTED
        if self.use_pt_ddp:
            rank_zero_logger_info(" > Using PyTorch DDP", logger)
            init_distributed(
                args.rank,
                self.num_gpus,
                args.group_id,
                self.config.distributed_backend,
                self.config.distributed_url,
            )

        if self.use_cuda:
            self.model.cuda()
            if isinstance(self.criterion, list):
                for criterion in self.criterion:
                    if isinstance(criterion, torch.nn.Module):
                        criterion.cuda()
            else:
                if isinstance(self.criterion, torch.nn.Module):
                    self.criterion.cuda()

        
        # --------- Emotion-only fine-tune setup --------------
        # 放在 model 已经移动到 device，并且在构建 optimizer 之前执行
        EMOTION_TOKEN_IDS = [6681, 6682, 6683, 6684, 6685]

        def _setup_emotion_finetune(trainer):
            try:
                gpt = trainer.model.xtts.gpt
                # 注意：text_embedding 是 nn.Embedding，.weight 是 Parameter
                emb = gpt.text_embedding.weight       # shape [vocab, hidden]
                head_w = gpt.text_head.weight         # shape [vocab, hidden]
                head_b = gpt.text_head.bias           # shape [vocab]

                # 1) 冻结全模型参数
                for n, p in trainer.model.named_parameters():
                    p.requires_grad = False

                # 2) 解冻整个 embedding 与 head 张量（hook 会屏蔽非目标行的 grad）
                emb.requires_grad = True
                head_w.requires_grad = True
                head_b.requires_grad = True

                # 3) 预构建 mask（放在对应 device / dtype）
                device = emb.device
                dtype = emb.dtype
                vocab_size, emb_dim = emb.shape

                mask_2d = torch.zeros((vocab_size, emb_dim), device=device, dtype=dtype)
                mask_1d = torch.zeros((vocab_size,), device=device, dtype=dtype)
                emotion_ids_tensor = torch.tensor(EMOTION_TOKEN_IDS, device=device, dtype=torch.long)
                mask_2d[emotion_ids_tensor] = 1.0
                mask_1d[emotion_ids_tensor] = 1.0

                # === 关键：保存 refs 到 trainer（否则后续检查/监控找不到） ===
                trainer._emotion_emb = emb
                trainer._emotion_head_w = head_w
                trainer._emotion_head_b = head_b

                # store masks and ids for later monitoring / saving
                trainer._emotion_ids = emotion_ids_tensor
                trainer._emotion_mask_2d = mask_2d
                trainer._emotion_mask_1d = mask_1d

                # 4) 注册 hook（保存 handle 以便后续 remove）
                def emb_hook(grad):
                    # grad has same dtype/device as mask_2d
                    return grad * trainer._emotion_mask_2d

                def head_w_hook(grad):
                    return grad * trainer._emotion_mask_2d

                def head_b_hook(grad):
                    return grad * trainer._emotion_mask_1d

                # 保存 handles
                trainer._emb_hook_handle = emb.register_hook(emb_hook)
                trainer._head_w_hook_handle = head_w.register_hook(head_w_hook)
                trainer._head_b_hook_handle = head_b.register_hook(head_b_hook)

                # 5) 保存初始快照 (在 no_grad 下)
                with torch.no_grad():
                    trainer._emotion_init_emb = emb.index_select(0, trainer._emotion_ids).detach().clone()
                    trainer._emotion_prev_emb = trainer._emotion_init_emb.clone()
                    trainer._emotion_init_head_w = head_w.index_select(0, trainer._emotion_ids).detach().clone()
                    trainer._emotion_prev_head_w = trainer._emotion_init_head_w.clone()
                    trainer._emotion_init_head_b = head_b.index_select(0, trainer._emotion_ids).detach().clone()
                    trainer._emotion_prev_head_b = trainer._emotion_init_head_b.clone()

                # --- MOD: 定位并解冻 GPT 最后一层（整层解冻，包括 LayerNorm 等） ---
                # 说明：我们尝试多种路径来兼容不同实现（gpt.gpt.h, gpt.h, gpt.blocks 等）
                try:
                    last_block = None
                    last_name = None
                    # 优先常见路径 gpt.gpt.h (HuggingFace style inside nested 'gpt')
                    if hasattr(gpt, "gpt") and hasattr(gpt.gpt, "h"):
                        num_blocks = len(gpt.gpt.h)
                        last_idx = num_blocks - 1
                        last_block = gpt.gpt.h[last_idx]
                        last_name = f"gpt.gpt.h.{last_idx}"
                    # 其次直接 gpt.h
                    elif hasattr(gpt, "h"):
                        num_blocks = len(gpt.h)
                        last_idx = num_blocks - 1
                        last_block = gpt.h[last_idx]
                        last_name = f"gpt.h.{last_idx}"
                    # 其它备选（blocks / layers / transformer_blocks）
                    else:
                        for attr in ("blocks", "layers", "transformer_blocks"):
                            if hasattr(gpt, attr):
                                ll = getattr(gpt, attr)
                                if len(ll) > 0:
                                    last_block = ll[len(ll)-1]
                                    last_name = f"gpt.{attr}.{len(ll)-1}"
                                    break

                    if last_block is None:
                        # 如果仍未找到，抛出异常以打印 warning，但不终止流程
                        raise RuntimeError("Cannot locate GPT transformer blocks to unfreeze last layer.")

                    # 解冻最后一层的所有参数（整层）
                    for name, param in last_block.named_parameters():
                        param.requires_grad = True

                    # 保存到 trainer，后续 optimizer 构造会使用 trainer._gpt_last_block
                    trainer._gpt_last_block = last_block
                    trainer._gpt_last_name = last_name

                    if getattr(trainer, "c_logger", None):
                        print(f"[Emotion-Finetune] Unfroze GPT last block: {trainer._gpt_last_name}")

                    # --- MOD: Save GPT last initial snapshot (CPU) for strict delta checks ---
                    try:
                        trainer._gpt_last_init = {name: p.detach().cpu().clone() for name, p in trainer._gpt_last_block.named_parameters()}
                        trainer._gpt_last_param_names = list(trainer._gpt_last_init.keys())
                        print(f"[Emotion-Finetune] Saved GPT last block initial snapshot ({len(trainer._gpt_last_init)} tensors).")
                    except Exception as e:
                        print(f"[Emotion-Finetune] Warning: failed to save GPT-last init snapshot: {e}")

                    # --- MOD: Save small sample of frozen params initial snapshot for strict frozen-check (optional) ---
                    try:
                        frozen_names = [n for n, p in trainer.model.named_parameters() if not p.requires_grad]
                        import random
                        sample_n = min(10, len(frozen_names))
                        sampled = random.sample(frozen_names, sample_n) if sample_n > 0 else []
                        trainer._frozen_sample_init = {}
                        named_map = dict(trainer.model.named_parameters())
                        for n in sampled:
                            trainer._frozen_sample_init[n] = named_map[n].detach().cpu().clone()
                        print(f"[Emotion-Finetune] Saved {len(trainer._frozen_sample_init)} frozen param snapshots for verification.")
                    except Exception as e:
                        print(f"[Emotion-Finetune] Warning: failed to save frozen sample snapshots: {e}")

                except Exception as e:
                    # 仅打印 warning，不中断整个设置过程（便于回退测试）
                    print(f"[Emotion-Finetune] Warning: could not unfreeze GPT last layer: {e}")
                # --- END MOD ----------------------------------------------------
 
                """ # --- MOD: 定位并解冻 GPT 最后两层（整层解冻，包括 LayerNorm 等） ---
                try:
                    last_blocks = []
                    last_names = []

                    # 优先常见路径 gpt.gpt.h (HuggingFace style inside nested 'gpt')
                    if hasattr(gpt, "gpt") and hasattr(gpt.gpt, "h"):
                        num_blocks = len(gpt.gpt.h)
                        if num_blocks == 0:
                            raise RuntimeError("No transformer blocks found in gpt.gpt.h")
                        start_idx = max(0, num_blocks - 2)
                        for idx in range(start_idx, num_blocks):
                            last_blocks.append(gpt.gpt.h[idx])
                            last_names.append(f"gpt.gpt.h.{idx}")

                    # 其次直接 gpt.h
                    elif hasattr(gpt, "h"):
                        num_blocks = len(gpt.h)
                        if num_blocks == 0:
                            raise RuntimeError("No transformer blocks found in gpt.h")
                        start_idx = max(0, num_blocks - 2)
                        for idx in range(start_idx, num_blocks):
                            last_blocks.append(gpt.h[idx])
                            last_names.append(f"gpt.h.{idx}")

                    # 其它备选（blocks / layers / transformer_blocks）
                    else:
                        found = False
                        for attr in ("blocks", "layers", "transformer_blocks"):
                            if hasattr(gpt, attr):
                                ll = getattr(gpt, attr)
                                if len(ll) > 0:
                                    num_blocks = len(ll)
                                    start_idx = max(0, num_blocks - 2)
                                    for idx in range(start_idx, num_blocks):
                                        last_blocks.append(ll[idx])
                                        last_names.append(f"gpt.{attr}.{idx}")
                                    found = True
                                    break
                        if not found:
                            raise RuntimeError("Cannot locate GPT transformer blocks to unfreeze last layers.")

                    # 解冻找到的最后一/二层的所有参数（整层）
                    for blk in last_blocks:
                        for name, param in blk.named_parameters():
                            param.requires_grad = True

                    # 保存到 trainer：为兼容旧代码同时赋值单个 last_block（最后一个）及 blocks 列表
                    trainer._gpt_last_blocks = last_blocks
                    trainer._gpt_last_names = last_names
                    # 向后兼容：保留旧变量（指向最后一个 block）
                    if len(last_blocks) > 0:
                        trainer._gpt_last_block = last_blocks[-1]
                        trainer._gpt_last_name = last_names[-1]
                    else:
                        trainer._gpt_last_block = None
                        trainer._gpt_last_name = None

                    if getattr(trainer, "c_logger", None):
                        print(f"[Emotion-Finetune] Unfroze GPT last blocks: {trainer._gpt_last_names}")

                    # --- Save GPT last blocks initial snapshot (CPU) for strict delta checks ---
                    try:
                        trainer._gpt_last_init = {}
                        trainer._gpt_last_param_names = []
                        for i, blk in enumerate(trainer._gpt_last_blocks):
                            prefix = trainer._gpt_last_names[i]
                            for name, p in blk.named_parameters():
                                full_name = f"{prefix}.{name}"
                                trainer._gpt_last_init[full_name] = p.detach().cpu().clone()
                                trainer._gpt_last_param_names.append(full_name)
                        print(f"[Emotion-Finetune] Saved GPT last blocks initial snapshot ({len(trainer._gpt_last_init)} tensors).")
                    except Exception as e:
                        print(f"[Emotion-Finetune] Warning: failed to save GPT-last init snapshot: {e}")

                    # --- Save small sample of frozen params initial snapshot for verification (keeps your original logic) ---
                    try:
                        frozen_names = [n for n, p in trainer.model.named_parameters() if not p.requires_grad]
                        import random
                        sample_n = min(10, len(frozen_names))
                        sampled = random.sample(frozen_names, sample_n) if sample_n > 0 else []
                        trainer._frozen_sample_init = {}
                        named_map = dict(trainer.model.named_parameters())
                        for n in sampled:
                            trainer._frozen_sample_init[n] = named_map[n].detach().cpu().clone()
                        print(f"[Emotion-Finetune] Saved {len(trainer._frozen_sample_init)} frozen param snapshots for verification.")
                    except Exception as e:
                        print(f"[Emotion-Finetune] Warning: failed to save frozen sample snapshots: {e}")

                except Exception as e:
                    # 仅打印 warning，不中断整个设置过程（便于回退测试）
                    print(f"[Emotion-Finetune] Warning: could not unfreeze GPT last layers: {e}")
                # --- END MOD --- """


                if trainer.c_logger:
                    print(f"[Emotion-Finetune] configured. emotion token ids: {EMOTION_TOKEN_IDS}")

            except Exception as e:
                print(f"[Emotion-Finetune] setup failed: {e}")

        # 调用
        _setup_emotion_finetune(self)
        # ----------end------------- 


        # setup optimizer
        self.optimizer = self.get_optimizer(self.model, self.config)

        
        # （只解冻情感嵌入）放在 self.optimizer = self.get_optimizer(...) 之后，且在 amp/restore/scheduler 之前
        """ if hasattr(self, "_emotion_emb"):
            try:
                emotion_lr = getattr(self.config, "emotion_lr", None) or getattr(self.config, "lr", 5e-05)
                params_groups = [
                    # {"params": [self._emotion_emb], "weight_decay": 0.0, "lr": emotion_lr},
                    # {"params": [self._emotion_head_w, self._emotion_head_b], "weight_decay": 0.0, "lr": emotion_lr},

                    {"params": [self._emotion_emb], "weight_decay": 0.0, "lr": 5e-5},
                    {"params": [self._emotion_head_w, self._emotion_head_b], "weight_decay": 0.0, "lr": 5e-5},
                ]

                from trainer.trainer_utils import get_optimizer as bottom_getopt
                # bottom_getopt should accept parameter groups; 如果不接受也可以直接用 torch.optim.AdamW
                self.optimizer = bottom_getopt(self.config.optimizer, self.config.optimizer_params, emotion_lr, parameters=params_groups)

                # 创建完 optimizer 之后，打印 param groups，用于确认param-group 里的 lr没有被覆盖
                for i, g in enumerate(self.optimizer.param_groups):
                    print(f"param_group[{i}] lr = {g.get('lr')}, params_count = {len(g['params'])}")

                # optional: tag group names for logging/debug
                try:
                    self.optimizer._group_names = [["_emotion_emb"], ["_emotion_head_w", "_emotion_head_b"]]
                except Exception:
                    pass

                if hasattr(self, 'c_logger'):
                    print("[Emotion-Finetune] Trainer replaced optimizer to only emotion params.")
            except Exception as e:
                print(f"[Emotion-Finetune] optimizer replacement failed: {e}") """


        # ---------- REPLACE ORIGINAL OPTIMIZER BUILD WITH BACKUP + CUSTOM REPLACE ----------
        # 如果做了 emotion setup（即存在 self._emotion_emb），则按方案 A 重建 optimizer（emotion emb/head + GPT last）
        if hasattr(self, "_emotion_emb"):
            try:
                from trainer.trainer_utils import get_optimizer as bottom_getopt

                # 基准 lr
                base_lr = getattr(self.config, "lr", 5e-06)

                # 情感 embedding/head：放大 10 倍
                emotion_lr = getattr(self.config, "emotion_lr", 5e-05)

                # 构造 param_groups：embedding / head / （可选）GPT last block
                params_groups = [
                    {"params": [self._emotion_emb], "weight_decay": 0.0, "lr": emotion_lr},
                    {"params": [self._emotion_head_w, self._emotion_head_b], "weight_decay": 0.0, "lr": emotion_lr},
                ]

                # （GPT最后一层）若在 _setup_emotion_finetune 中保存了最后一层引用，则加入其参数
                if hasattr(self, "_gpt_last_block"):
                    gpt_last_lr = getattr(self.config, "gpt_last_lr",2e-06)# 建议 2.5e-6 或 3e-6
                    gpt_last_wd = getattr(self.config, "gpt_last_weight_decay", getattr(self.config, "optimizer_params", {}).get("weight_decay", 1e-2))
                    # 只收集 requires_grad=True 的参数（在 setup 中我们已把该层设为 requires_grad=True）
                    gpt_last_params = [p for p in self._gpt_last_block.parameters() if p.requires_grad]
                    if len(gpt_last_params) > 0:
                        params_groups.append({"params": gpt_last_params, "weight_decay": gpt_last_wd, "lr": gpt_last_lr})
                        if getattr(self, "c_logger", None):
                            print(f"[Emotion-Finetune] Added GPT last block params to custom param_groups, params_count = {sum(p.numel() for p in gpt_last_params)}")
                    else:
                        print("[Emotion-Finetune] Warning: _gpt_last_block exists but no requires_grad params found.")

                # （GPT最后两层）若在 _setup_emotion_finetune 中保存了最后一层/几层引用，则加入其参数（支持 _gpt_last_blocks 或 兼容 _gpt_last_block）
                """ gpt_blocks = None
                if hasattr(self, "_gpt_last_blocks"):
                    gpt_blocks = getattr(self, "_gpt_last_blocks")
                elif hasattr(self, "_gpt_last_block"):
                    gpt_blocks = [getattr(self, "_gpt_last_block")]

                if gpt_blocks:
                    gpt_last_lr = getattr(self.config, "gpt_last_lr", 2e-06)
                    gpt_last_wd = getattr(self.config, "gpt_last_weight_decay", getattr(self.config, "optimizer_params", {}).get("weight_decay", 1e-2))
                    # 收集所有解冻 block 中 requires_grad=True 的参数
                    gpt_last_params = []
                    for blk in gpt_blocks:
                        gpt_last_params.extend([p for p in blk.parameters() if p.requires_grad])
                    if len(gpt_last_params) > 0:
                        params_groups.append({"params": gpt_last_params, "weight_decay": gpt_last_wd, "lr": gpt_last_lr})
                        if getattr(self, "c_logger", None):
                            try:
                                total_numel = sum(p.numel() for p in gpt_last_params)
                            except Exception:
                                total_numel = "?"
                            print(f"[Emotion-Finetune] Added GPT last block(s) params to custom param_groups, blocks={len(gpt_blocks)}, params_count = {total_numel}")
                    else:
                        print("[Emotion-Finetune] Warning: _gpt_last_blocks/_gpt_last_block exists but no requires_grad params found.")
 """

                # 调用 trainer_utils.get_optimizer 重建 optimizer（将 param_groups 传给 torch.optim）
                # bottom_getopt 的签名：get_optimizer(name, optimizer_params, lr, model=None, parameters=None)
                # 我们传入 emotion_lr 作为 base lr（不过 param_groups 中的 lr 会覆盖）
                try:
                    self.optimizer = bottom_getopt(self.config.optimizer, self.config.optimizer_params, emotion_lr, parameters=params_groups)
                    # 将原始 optimizer 作为备份保留（已保存到 self._orig_optimizer）
                except Exception as e:
                    print(f"[Emotion-Finetune] Failed to build custom optimizer via bottom_getopt: {e}")
                    # 回退到原始 optimizer（self._orig_optimizer 已赋值）
                    self.optimizer = self._orig_optimizer
                
                # —— 在 bottom_getopt(...) 成功之后立刻添加 —— 
                try:
                    # 1) 记录每个 param_group 的目标 lr（后续手动 warmup 用）
                    self._init_group_lrs = [g.get("lr", getattr(self.config, "lr", 5e-06)) for g in self.optimizer.param_groups]
                    if getattr(self, "c_logger", None):
                        print(f"[Emotion-Finetune] Saved init param_group lrs: {self._init_group_lrs}")
                except Exception as e:
                    print("[Emotion-Finetune] Warning: failed to save init_group_lrs:", e)
                    self._init_group_lrs = None

                # （GPT最后一层）标记只对 GPT-last 分组做 warmup（布尔掩码，长度与 param_groups 一致）
                self._warmup_mask = None
                try:
                    if hasattr(self, "_gpt_last_block"):
                        gpt_params = set(p for p in self._gpt_last_block.parameters() if p.requires_grad)
                        mask = []
                        for g in self.optimizer.param_groups:
                            pg_params = set(g["params"]) if isinstance(g["params"], (list, tuple)) else {g["params"]}
                            # 任何一个参数命中 _gpt_last_block，就视为 GPT-last 组
                            hit = any((p in gpt_params) for p in pg_params)
                            mask.append(bool(hit))
                        self._warmup_mask = mask
                        if getattr(self, "c_logger", None):
                            print(f"[Emotion-Finetune] Warmup mask per group (True=GPT-last): {self._warmup_mask}")
                except Exception as e:
                    print("[Emotion-Finetune] Warning: failed to build warmup_mask:", e)
                    self._warmup_mask = None

                #（GPT最后两层）标记只对 GPT-last 分组做 warmup（布尔掩码，长度与 param_groups 一致）
                """ self._warmup_mask = None
                try:
                    # 优先支持 _gpt_last_blocks（list），向后兼容 _gpt_last_block（单个）
                    gpt_blocks = None
                    if hasattr(self, "_gpt_last_blocks"):
                        gpt_blocks = getattr(self, "_gpt_last_blocks")
                    elif hasattr(self, "_gpt_last_block"):
                        gpt_blocks = [getattr(self, "_gpt_last_block")]

                    if gpt_blocks:
                        # 把所有 gpt_last block 的 params 放进 set 以便快速 membership 测试
                        gpt_params = set()
                        for blk in gpt_blocks:
                            for p in blk.parameters():
                                if p.requires_grad:
                                    gpt_params.add(p)
                        mask = []
                        for g in self.optimizer.param_groups:
                            pg_params = g.get("params", [])
                            # 统一把 param_group 中的 params 转为 set 以便检查（处理 list/tuple 或单个 param）
                            try:
                                if isinstance(pg_params, (list, tuple)):
                                    pg_set = set(pg_params)
                                else:
                                    pg_set = {pg_params}
                            except Exception:
                                # 如果遇到不可 hash 的包装（极少见），就尝试遍历并收集 object id
                                try:
                                    pg_list = list(pg_params)
                                    pg_set = set(pg_list)
                                except Exception:
                                    pg_set = set()
                            # 若 param_group 中任意一个参数在 gpt_params 中，则该组属于 GPT-last
                            hit = any((p in gpt_params) for p in pg_set)
                            mask.append(bool(hit))
                        self._warmup_mask = mask
                        if getattr(self, "c_logger", None):
                            print(f"[Emotion-Finetune] Warmup mask per group (True=GPT-last): {self._warmup_mask}")
                except Exception as e:
                    print("[Emotion-Finetune] Warning: failed to build warmup_mask:", e)
                    self._warmup_mask = None """


            except Exception as e:
                print(f"[Emotion-Finetune] optimizer replacement failed: {e}")
                # 回退到原始 optimizer
                self.optimizer = self._orig_optimizer

        # ---------- 验证与检查（GPT最后两层） ----------
        """ try:
            print("[Emotion-Finetune] Running post-optimizer checks...")

            # 1) 列出 param_groups 的 lr / wd / params_count，并列出每组前几个 param 的 id 和 shape（便于定位）
            for i, g in enumerate(self.optimizer.param_groups):
                lr = g.get("lr", None)
                wd = g.get("weight_decay", None)
                params = g.get("params", [])
                # 安全计算 param 数量
                try:
                    if isinstance(params, (list, tuple)):
                        nparams = sum(p.numel() for p in params if hasattr(p, "numel"))
                    else:
                        nparams = params.numel() if hasattr(params, "numel") else len(params)
                except Exception:
                    nparams = "?"
                # 采样打印前几个 param 的 id / shape / requires_grad（最多 3 个）
                sample_info = []
                try:
                    if isinstance(params, (list, tuple)):
                        sample_params = params[:3]
                    else:
                        # 有些实现会把 params 设为迭代器或 generator（罕见），我们尝试转换为 list
                        sample_params = list(params)[:3] if hasattr(params, "__iter__") else []
                    for p in sample_params:
                        if hasattr(p, "shape"):
                            sample_info.append(f"id={id(p)} shape={tuple(p.shape)} req_grad={p.requires_grad}")
                        else:
                            sample_info.append(f"id={id(p)} type={type(p)}")
                except Exception:
                    sample_info = ["<sample-info-failed>"]
                print(f"[Check] param_group[{i}] lr={lr}, weight_decay={wd}, params_count={nparams}, samples={sample_info}")

            # 2) 列出当前可训练参数名（便于确认哪些层被解冻）
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            print("[Check] Trainable params after optimizer replace (showing up to 200 names):")
            for n in trainable[:200]:
                print("   ", n)
            print("[Check] Total trainable params (numel):", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

            # 3) 检查你的 hook 是否注册（hook handle 存在性）
            print("[Check] embedding hook handle exists:", hasattr(self, "_emb_hook_handle"))
            print("[Check] head_w hook handle exists:", hasattr(self, "_head_w_hook_handle"))
            print("[Check] head_b hook handle exists:", hasattr(self, "_head_b_hook_handle"))

            # 4) 如果有 _gpt_last_block 或 _gpt_last_blocks，则打印其参数统计与前几个 param 的 id 以便对比
            if hasattr(self, "_gpt_last_blocks"):
                blocks = getattr(self, "_gpt_last_blocks")
                print(f"[Check] _gpt_last_blocks count = {len(blocks)}")
                for bi, blk in enumerate(blocks):
                    blk_params = [p for p in blk.parameters()]
                    try:
                        blk_numel = sum(p.numel() for p in blk_params)
                    except Exception:
                        blk_numel = "?"
                    blk_sample_ids = [id(p) for p in blk_params[:3]]
                    print(f"   block[{bi}] params={len(blk_params)} numel={blk_numel} sample_param_ids={blk_sample_ids}")
            elif hasattr(self, "_gpt_last_block"):
                blk = getattr(self, "_gpt_last_block")
                blk_params = [p for p in blk.parameters()]
                try:
                    blk_numel = sum(p.numel() for p in blk_params)
                except Exception:
                    blk_numel = "?"
                blk_sample_ids = [id(p) for p in blk_params[:3]]
                print(f"[Check] _gpt_last_block params={len(blk_params)} numel={blk_numel} sample_param_ids={blk_sample_ids}")

            # 5) 打印 warmup mask（如果存在），与 param_groups 长度对比
            if hasattr(self, "_warmup_mask"):
                print("[Check] warmup_mask:", self._warmup_mask, "len(param_groups)=", len(self.optimizer.param_groups))
            else:
                print("[Check] warmup_mask not set.")

            # 6) 另：快速映射每个 param_group 是否包含任意 gpt_last params（通过 id 比较），用于二次验证
            try:
                # 收集所有 gpt_last param 的 id（如果存在）
                gpt_param_ids = set()
                if hasattr(self, "_gpt_last_blocks"):
                    for blk in self._gpt_last_blocks:
                        for p in blk.parameters():
                            gpt_param_ids.add(id(p))
                elif hasattr(self, "_gpt_last_block"):
                    for p in self._gpt_last_block.parameters():
                        gpt_param_ids.add(id(p))

                for i, g in enumerate(self.optimizer.param_groups):
                    params = g.get("params", [])
                    found = False
                    try:
                        iterable = params if isinstance(params, (list, tuple)) else list(params) if hasattr(params, "__iter__") else []
                        for p in iterable:
                            if id(p) in gpt_param_ids:
                                found = True
                                break
                    except Exception:
                        found = "<check-failed>"
                    print(f"[Check] param_group[{i}] contains_gpt_last_param = {found}")
            except Exception:
                pass

        except Exception as e:
            print(f"[Emotion-Finetune] Pre-train checks failed: {e}") """
        # ---------- end checks ----------


        # 解冻GPT最后一层的检查代码
        #打印 param groups（检查）
        """ try:
            for i, g in enumerate(self.optimizer.param_groups):
                try:
                    nparams = sum(p.numel() for p in g['params'])
                except Exception:
                    nparams = len(g['params'])
                print(f"param_group[{i}] lr = {g.get('lr')}, weight_decay = {g.get('weight_decay')}, params_count = {nparams}")
        except Exception:
            pass """

        # TRAINING PRE-CHECKS（强烈建议执行）
        """ try:
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            print("[Check] Trainable params after optimizer replace:")
            for n in trainable:
                print("  ", n)
            print("[Check] Total trainable params (numel):", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

            # 确认 hook 是否已注册（你的 hook 存在于 trainer 对象）
            print("[Check] embedding hook handle exists:", hasattr(self, "_emb_hook_handle"))
            print("[Check] head_w hook handle exists:", hasattr(self, "_head_w_hook_handle"))
            print("[Check] head_b hook handle exists:", hasattr(self, "_head_b_hook_handle"))

            # 确认 optimizer param groups
            for i, g in enumerate(self.optimizer.param_groups):
                nparams = sum(p.numel() for p in g['params'])
                print(f"[Check] optimizer param_group[{i}] lr = {g.get('lr')}, weight_decay = {g.get('weight_decay')}, params_count = {nparams}")
        except Exception as e:
            print(f"[Emotion-Finetune] Pre-train checks failed: {e}") """
        # ---------- END REPLACE ----------

        
        # （只微调embedding）打印主进程可训练参数信息
        """ if getattr(self, "args", None) is None or getattr(self.args, "rank", 0) == 0:
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            print("[Emotion-Finetune] Trainable params after optimizer replacement:")
            for n in trainable:
                print("  ", n) """



        # If multiple-optimizer setup with grad accumulation and without custom optimize method raise an error
        if (
            self.grad_accum_steps != 1
            and isinstance(self.optimizer, list)
            and not isimplemented(self.model, "optimize")
        ):
            raise ValueError(
                " [!] Coqui Trainer does not support grad_accum_steps for multiple-optimizer setup, please set grad_accum_steps to 1 or implement in your model a custom method called ´optimize` that need to deal with dangling gradients in multiple-optimizer setup!"
            )

        # CALLBACK
        self.callbacks = TrainerCallback()
        self.callbacks.parse_callbacks_dict(callbacks)
        self.callbacks.on_init_start(self)

        # init AMP
        if self.use_amp_scaler:
            if self.use_apex:
                self.scaler = None
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # 断点续训
        # self.args.restore_path = "/tmp/xtts_ft/run/training/GPT_XTTS_FT-September-05-2025_09+55AM-dbf1a08a/checkpoint_47000.pth"
        # print(f"self.args.restore_path:{self.args.restore_path}")
        # args.restore_path = "/tmp/xtts_ft/run/training/GPT_XTTS_FT-September-05-2025_09+55AM-dbf1a08a/checkpoint_47000.pth"
        # print(f"args.restore_path:{args.restore_path}")

        # restore model
        if self.args.restore_path:
            (self.model, self.optimizer, self.scaler, self.restore_step, self.restore_epoch) = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer, self.scaler
            )
            self.scaler = torch.cuda.amp.GradScaler()
            # print("执行到restore model了")

        # setup scheduler
        self.scheduler = self.get_scheduler(self.model, self.config, self.optimizer)
        self.scheduler = self.restore_scheduler(
            self.scheduler, self.args, self.config, self.restore_epoch, self.restore_step
        )

        # DISTRIBUTED
        if self.use_pt_ddp:
            self.model = DDP_th(self.model, device_ids=[args.rank], output_device=args.rank)

        # setup accelerator
        self.setup_accelerate()

        # count model size
        num_params = count_parameters(self.model)
        rank_zero_logger_info(f"\n > Model has {num_params} parameters", logger)

        # 打印 GPT 模块参数量（如果存在）
        # if hasattr(self.model, "xtts") and hasattr(self.model.xtts, "gpt"):
        #     gpt_total_params = sum(p.numel() for p in self.model.xtts.gpt.parameters())
        #     gpt_trainable_params = count_parameters(self.model.xtts.gpt)
        #     rank_zero_logger_info(
        #         f"[GPT 模块] 总参数量：{gpt_total_params:,}；可训练参数量：{gpt_trainable_params:,}", logger
        #     )

        self.callbacks.on_init_end(self)
        self.dashboard_logger.add_config(config)
        self.save_training_script()
        ping_training_run()
        # print(f"Trainer 初始化完成，模型设备：{next(self.model.parameters()).device}")

    @property
    def use_apex(self):
        """Return True if using APEX."""
        return not self.args.use_accelerate and self._is_apex_available()

    @property
    def use_pt_ddp(self):
        """Return True if using PyTorch DDP."""
        return self.num_gpus > 1 and not self.use_accelerate

    @property
    def use_accelerate(self):
        """Return True if using HF Accelerate."""
        return self.args.use_accelerate

    def setup_accelerate(self):
        if self.use_accelerate:
            # print("正在执行setup_accelerate函数")
            self.model, self.optimizer, self.train_loader, self.scheduler, self.accelerator = self.init_accelerate(
                model=self.model,
                optimizer=self.optimizer,
                training_dataloader=self.train_loader,
                scheduler=self.scheduler,
                grad_accum_steps=self.grad_accum_steps,
                mixed_precision=self.config.mixed_precision,
                precision=self.config.precision,
            )

    def prepare_accelerate_loader(self, data_loader):
        """Prepare the accelerator for the training."""
        if self.use_accelerate:
            return self.accelerator.prepare_data_loader(data_loader)
        return data_loader

    @staticmethod
    def init_accelerate(model, optimizer, training_dataloader, scheduler, grad_accum_steps, mixed_precision, precision):
        """Setup HF Accelerate for the training."""

        # print("正在执行init_accelerate函数")

        # check if accelerate is installed
        try:
            from accelerate import Accelerator  # pylint:disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Please install accelerate to use this feature.") from e

        """ _precision = precision if precision is not None else "f16" if mixed_precision else None
        if _precision == "float16":
            _precision = "f16"
        elif _precision == "float8":
            _precision = "f8"
        elif _precision == "bfloat16":
            _precision = "bf16" """
        # 1 映射所有常见写法到 Accelerate 支持的模式
        if precision is not None:
            if precision in ("float16", "fp16"):
                _precision = "fp16"
            elif precision in ("bfloat16", "bf16"):
                _precision = "bf16"
            elif precision in ("float8", "fp8"):
                _precision = "fp8"
            elif precision == "no":
                _precision = "no"
            else:
                raise ValueError(f"Unsupported precision value: {precision}")
        else:
            # mixed_precision=True 时自动走 fp16，否则 none
            _precision = "fp16" if mixed_precision else "no"

        # 2 直接把 _precision 传给 Accelerate，不再做额外白名单检查
        accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision=_precision
        )
        if isinstance(model, torch.nn.Module):
            model = accelerator.prepare_model(model)

        if isinstance(optimizer, dict):
            for key, optim in optimizer.items():
                optimizer[key] = accelerator.prepare_optimizer(optim)
        elif isinstance(optimizer, list):
            for i, optim in enumerate(optimizer):
                optimizer[i] = accelerator.prepare_optimizer(optim)
        elif optimizer is not None:
            optimizer = accelerator.prepare_optimizer(optimizer)

        if isinstance(training_dataloader, torch.utils.data.DataLoader):
            training_dataloader = accelerator.prepare_data_loader(training_dataloader)

        if isinstance(scheduler, dict):
            for key, sched in scheduler.items():
                scheduler[key] = accelerator.prepare_scheduler(sched)
        elif isinstance(scheduler, list):
            for i, sched in enumerate(scheduler):
                scheduler[i] = accelerator.prepare_scheduler(sched)
        elif scheduler is not None:
            scheduler = accelerator.prepare_scheduler(scheduler)

        return model, optimizer, training_dataloader, scheduler, accelerator

    def save_training_script(self):
        """Save the training script to tracking dashboard and output path."""
        file_path = sys.argv[0]
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            self.dashboard_logger.add_artifact(file_or_dir=file_path, name=file_name, artifact_type="file")
            with open(file_path, "r", encoding="utf8") as f:
                self.dashboard_logger.add_text("training-script", f"{f.read()}", 0)
            shutil.copyfile(file_path, os.path.join(self.output_path, file_name))

    @staticmethod
    def parse_argv(args: Union[Coqpit, List]):
        """Parse command line arguments to init or override `TrainerArgs()`."""
        if isinstance(args, Coqpit):
            parser = args.init_argparse(arg_prefix="")
        else:
            train_config = TrainerArgs()
            parser = train_config.init_argparse(arg_prefix="")
        training_args, coqpit_overrides = parser.parse_known_args()
        args.parse_args(training_args)
        return args, coqpit_overrides

    @staticmethod
    def init_loggers(config: "Coqpit", output_path: str, dashboard_logger=None, c_logger=None):
        """Init console and dashboard loggers.
        Use the given logger if passed externally else use config values to pick the right logger.
        Return a dashboard logger only for the rank 0 process in DDP
        Define a console logger for each process in DDP

        Args:
            config (Coqpit): Model config.
            output_path (str): Output path to save the training artifacts.
            dashboard_logger (DashboardLogger): Object passed to the trainer from outside.
            c_logger (ConsoleLogger): Object passed to the trained from outside.

        Returns:
            Initialized dashboard_logger and console_logger objects.
        """
        c_logger = ConsoleLogger() if c_logger is None else c_logger

        # only allow dashboard logging for the main process in DDP mode
        if get_rank() > 0:
            return DummyLogger(), c_logger
        if dashboard_logger is None:
            dashboard_logger = logger_factory(config, output_path)
        return dashboard_logger, c_logger

    def setup_small_run(self, small_run: int = None):
        """Use a subset of samples for training, evaluation and testing."""
        if small_run is not None:
            logger.info("[!] Small Run, only using %i samples.", small_run)
            self.train_samples = None if self.train_samples is None else self.train_samples[:small_run]
            self.eval_samples = None if self.eval_samples is None else self.eval_samples[:small_run]
            self.test_samples = None if self.test_samples is None else self.test_samples[:small_run]

    @staticmethod
    def init_training(args: TrainerArgs, coqpit_overrides: Dict, config: Coqpit = None):
        """Initialize training and update model configs from command line arguments.

        Args:
            args (argparse.Namespace or dict like): Parsed trainer arguments.
            config_overrides (argparse.Namespace or dict like): Parsed config overriding arguments.
            config (Coqpit): Model config. If none, it is generated from `args`. Defaults to None.

        Returns:
            config (Coqpit): Config paramaters.
        """
        # set arguments for continuing training
        if args.continue_path:
            args.config_path = os.path.join(args.continue_path, "config.json")
            args.restore_path, best_model = get_last_checkpoint(args.continue_path)
            if not args.best_path:
                args.best_path = best_model
            # use the same config
            if config:
                config.load_json(args.config_path)
            else:
                coqpit = Coqpit()
                coqpit.load_json(args.config_path)

        # override config values from command-line args
        # TODO: Maybe it is better to do it outside
        if len(coqpit_overrides) > 0:
            config.parse_known_args(coqpit_overrides, relaxed_parser=True)

        # update the config.json fields and copy it to the output folder
        new_fields = {}
        if args.rank == 0:
            if args.restore_path:
                new_fields["restore_path"] = args.restore_path
            new_fields["github_branch"] = get_git_branch()
        return config, new_fields

    @staticmethod
    def setup_training_environment(args, config, gpu):
        if platform.system() != "Windows":
            # https://github.com/pytorch/pytorch/issues/973
            import resource  # pylint: disable=import-outside-toplevel

            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

        # set and initialize Pytorch runtime
        use_cuda, num_gpus = setup_torch_training_env(
            args=args,
            cudnn_enable=config.cudnn_enable,
            cudnn_deterministic=config.cudnn_deterministic,
            cudnn_benchmark=config.cudnn_benchmark,
            use_ddp=args.use_ddp,
            training_seed=config.training_seed,
            allow_tf32=config.allow_tf32,
            gpu=gpu if args.gpu is None else args.gpu,
        )

        print_training_env(args, config)
        return use_cuda, num_gpus

    @staticmethod
    def run_get_model(config: Coqpit, get_model: Callable) -> nn.Module:
        """Run the `get_model` function and return the model.

        Args:
            config (Coqpit): Model config.

        Returns:
            nn.Module: initialized model.
        """
        if len(signature(get_model).sig.parameters) == 1:
            model = get_model(config)
        else:
            model = get_model()
        return model

    @staticmethod
    def run_get_data_samples(config: Coqpit, get_data_samples: Callable) -> nn.Module:
        if callable(get_data_samples):
            if len(signature(get_data_samples).sig.parameters) == 1:
                train_samples, eval_samples = get_data_samples(config)
            else:
                train_samples, eval_samples = get_data_samples()
            return train_samples, eval_samples
        return None, None

    def restore_model(
        self,
        config: Coqpit,
        restore_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]:
        """Restore training from an old run. It restores model, optimizer, AMP scaler and training stats.

        Args:
            config (Coqpit): Model config.
            restore_path (str): Path to the restored training run.
            model (nn.Module): Model to restored.
            optimizer (torch.optim.Optimizer): Optimizer to restore.
            scaler (torch.cuda.amp.GradScaler, optional): AMP scaler to restore. Defaults to None.

        Returns:
            Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]: [description]
        """

        def _restore_list_objs(states, obj):
            if isinstance(obj, list):
                for idx, state in enumerate(states):
                    obj[idx].load_state_dict(state)
            elif isinstance(obj, dict):
                for key, state in states.items():
                    obj[key].load_state_dict(state)
            else:
                obj.load_state_dict(states)
            return obj

        logger.info(" > Restoring from %s ...", os.path.basename(restore_path))
        checkpoint = load_fsspec(restore_path, map_location="cpu")

        try:
            logger.info(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            logger.info(" > Restoring Optimizer...")
            try:
                optimizer = _restore_list_objs(checkpoint["optimizer"], optimizer)
            except (KeyError, TypeError, RuntimeError):
                logger.info(" > Optimizer is not compatible with the restored model.")
            if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
                logger.info(" > Restoring Scaler...")
                scaler = _restore_list_objs(checkpoint["scaler"], scaler)
        except (KeyError, RuntimeError, ValueError):
            logger.info(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_partial_state_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        optimizer = self.restore_lr(config, self.args, model, optimizer)

        logger.info(" > Model restored from step %i", checkpoint["step"])
        restore_step = checkpoint["step"] + 1  # +1 not to immediately checkpoint if the model is restored
        restore_epoch = checkpoint["epoch"]
        torch.cuda.empty_cache()
        return model, optimizer, scaler, restore_step, restore_epoch

    def restore_lr(self, config, args, model, optimizer):
        # use the same lr if continue training
        if not args.continue_path:
            if isinstance(optimizer, list):
                for idx, optim in enumerate(optimizer):
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[idx]
            elif isinstance(optimizer, dict):
                for optim_name, optim in optimizer.items():
                    for group in optim.param_groups:
                        group["lr"] = self.get_lr(model, config)[optim_name]
            else:
                for group in optimizer.param_groups:
                    group["lr"] = self.get_lr(model, config)
        return optimizer

    #########################
    # DATA LOADING FUNCTIONS
    #########################

    def _get_loader(
        self,
        model: nn.Module,
        config: Coqpit,
        assets: Dict,
        is_eval: str,
        samples: List,
        verbose: bool,
        num_gpus: int,
    ) -> DataLoader:
        if num_gpus > 1:
            if isimplemented(model.module, "get_data_loader"):
                loader = model.module.get_data_loader(
                    config,
                    assets,
                    is_eval,
                    samples,
                    verbose,
                    num_gpus,
                    self.args.rank,
                )
        else:
            if isimplemented(model, "get_data_loader"):
                loader = model.get_data_loader(
                    config=config, assets=assets, is_eval=is_eval, samples=samples, verbose=verbose, num_gpus=num_gpus
                )

        assert (
            len(loader) > 0
        ), " ❗ len(DataLoader) returns 0. Make sure your dataset is not empty or len(dataset) > 0. "
        return loader

    def get_train_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a training data loader.
        Call ```model.get_train_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=False```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if isimplemented(self.model.module, "get_train_data_loader"):
                loader = self.model.module.get_train_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if isimplemented(self.model, "get_train_data_loader"):
                loader = self.model.get_train_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            False,
            samples,
            verbose,
            self.num_gpus,
        )

    def get_eval_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a evaluation data loader.
        Call ```model.get_eval_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if isimplemented(self.model.module, "get_eval_data_loader"):
                loader = self.model.module.get_eval_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if isimplemented(self.model, "get_eval_data_loader"):
                loader = self.model.get_eval_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            True,
            samples,
            verbose,
            self.num_gpus,
        )

    def get_test_dataloader(self, training_assets: Dict, samples: List, verbose: bool) -> DataLoader:
        """Initialize and return a evaluation data loader.
        Call ```model.get_test_data_loader``` if it is implemented, else call ```model.get_data_loader```
        and set ```is_eval=True```.

        Args:
            ap (AudioProcessor): Audio processor.
            samples (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        if self.num_gpus > 1:
            if isimplemented(self.model.module, "get_test_data_loader"):
                loader = self.model.module.get_test_data_loader(
                    self.config,
                    self.training_assets,
                    samples,
                    verbose,
                    self.num_gpus,
                    self.args.rank,
                )
                return loader
        else:
            if isimplemented(self.model, "get_test_data_loader"):
                loader = self.model.get_test_data_loader(
                    self.config, self.training_assets, samples, verbose, self.num_gpus
                )
                return loader

        return self._get_loader(
            self.model,
            self.config,
            training_assets,
            True,
            samples,
            verbose,
            self.num_gpus,
        )

    def format_batch(self, batch: List) -> Dict:
        """Format the dataloader output and return a batch.

        1. Call ```model.format_batch```.
        2. Pass the batch to the Device.
        3. Call ```model.format_batch_on_device```.

        Args:
            batch (List): Batch returned by the dataloader.

        Returns:
            Dict: Formatted batch.
        """
        try:
            if self.num_gpus > 1:
                batch = self.model.module.format_batch(batch)
            else:
                batch = self.model.format_batch(batch)
        except NotImplementedError:
            pass

        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        elif isinstance(batch, list):
            batch = [to_cuda(v) for v in batch]

        try:
            if self.num_gpus > 1:
                batch = self.model.module.format_batch_on_device(batch)
            else:
                batch = self.model.format_batch_on_device(batch)
        except NotImplementedError:
            pass
        return batch

    ######################
    # TRAIN FUNCTIONS
    ######################

    @staticmethod
    def master_params(optimizer: torch.optim.Optimizer):
        """Generator over parameters owned by the optimizer.

        Used to select parameters used by the optimizer for gradient clipping.

        Args:
            optimizer: Target optimizer.
        """
        for group in optimizer.param_groups:
            for p in group["params"]:
                yield p

    @staticmethod
    def _model_train_step(
        batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a trainig forward step. Compute model outputs and losses.

        Args:
            batch (Dict): [description]
            model (nn.Module): [description]
            criterion (nn.Module): [description]
            optimizer_idx (int, optional): [description]. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: [description]
        """
        input_args = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        # unwrap model in DDP training
        if hasattr(model, "module"):
            return model.module.train_step(*input_args)
        return model.train_step(*input_args)

    def _get_autocast_args(self, mixed_precision: bool, precision: str):
        device = "cpu"
        dtype = torch.get_autocast_cpu_dtype()
        if self.use_cuda:
            device = "cuda"
            dtype = torch.float32
            if mixed_precision:
                if precision == "fp16":
                    dtype = torch.float16
                elif precision == "bf16":
                    dtype = torch.bfloat16
                else:
                    raise ValueError(f" ❗ Unknown precision {precision}")
        elif mixed_precision:
            dtype = torch.bfloat16
        return device, dtype

    def detach_loss_dict(
        self, loss_dict: Dict, step_optimizer: bool, optimizer_idx: int = None, grad_norm: float = None
    ):
        # detach losses for logging
        loss_dict_detached = self._detach_loss_dict(loss_dict)
        # loss_dict_detached["loss"] = loss_di`ct_detached["loss"] * float(self.grad_accum_steps)

        if optimizer_idx is not None:
            loss_dict_detached[f"loss_{optimizer_idx}"] = loss_dict_detached.pop("loss")
            if step_optimizer and grad_norm is not None:
                loss_dict_detached[f"grad_norm_{optimizer_idx}"] = grad_norm
        else:
            if step_optimizer and grad_norm is not None:
                loss_dict_detached["grad_norm"] = grad_norm
        return loss_dict_detached

    def _compute_loss(self, batch: Dict, model: nn.Module, criterion: nn.Module, config: Coqpit, optimizer_idx: int):
        device, dtype = self._get_autocast_args(config.mixed_precision, config.precision)
        with torch.autocast(device_type=device, dtype=dtype, enabled=config.mixed_precision):
            if optimizer_idx is not None:
                outputs, loss_dict = self._model_train_step(batch, model, criterion, optimizer_idx=optimizer_idx)
            else:
                outputs, loss_dict = self._model_train_step(batch, model, criterion)
        return outputs, loss_dict

    @staticmethod
    def _set_grad_clip_per_optimizer(config: Coqpit, optimizer_idx: int):
        # set gradient clipping threshold
        grad_clip = 0.0  # meaning no gradient clipping
        if "grad_clip" in config and config.grad_clip is not None:
            if optimizer_idx is not None:
                try:
                    grad_clip = config.grad_clip[optimizer_idx]
                except TypeError:
                    logger.info(" [!] You are using multiple optimizers but `grad_clip` is not a list.")
            else:
                grad_clip = config.grad_clip
        return grad_clip

    def _compute_grad_norm(self, optimizer: torch.optim.Optimizer):
        # 只计算有梯度的参数
        grads = [p.grad.view(-1) for p in self.master_params(optimizer) if p.grad is not None]
        if len(grads) == 0:
            return torch.tensor(0.0, device=self._device)  # 没有梯度时返回 0
        return torch.norm(torch.cat(grads), p=2)
    
        # 原实现
        # return torch.norm(torch.cat([param.grad.view(-1) for param in self.master_params(optimizer)], dim=0), p=2)

    def _grad_clipping(self, grad_clip: float, optimizer: torch.optim.Optimizer, scaler: "AMPScaler"):
        """Perform gradient clipping"""
        if grad_clip is not None and grad_clip > 0:
            if scaler:
                scaler.unscale_(optimizer)
            self.callbacks.before_gradient_clipping(self)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
        else:
            grad_norm = self._compute_grad_norm(optimizer)
        return grad_norm

    # 对原有实现进行改进，打印梯度裁剪触发频率
    # def _grad_clipping(self, grad_clip: float, optimizer: torch.optim.Optimizer, scaler: "AMPScaler"):
    #     """Perform gradient clipping with basic statistics and safe printing."""
    #     # init counters once
    #     if not hasattr(self, "_grad_clip_checks"):
    #         self._grad_clip_checks = 0
    #     if not hasattr(self, "_grad_clip_triggers"):
    #         self._grad_clip_triggers = 0

    #     if grad_clip is not None and grad_clip > 0:
    #         # 如果使用 AMP scaler，先取消 scale（你原实现有这一步）
    #         if scaler:
    #             try:
    #                 scaler.unscale_(optimizer)
    #             except Exception as e:
    #                 print("[GradClip] scaler.unscale_ failed:", e)

    #         # 保持原有回调
    #         self.callbacks.before_gradient_clipping(self)

    #         # clip_grad_norm_ 会 **就地修改** p.grad 并返回裁剪前的 total_norm (tensor)
    #         grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)

    #         # ------------- 统计与打印（稳健获取 scalar） -------------
    #         # grad_norm 可能是 Tensor（在 GPU），用 item() 取得 python float（会同步）
    #         try:
    #             pre_val = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    #         except Exception:
    #             # 兜底
    #             pre_val = float(grad_norm)

    #         clipped_flag = pre_val > float(grad_clip) + 1e-12
    #         self._grad_clip_checks += 1
    #         if clipped_flag:
    #             self._grad_clip_triggers += 1

    #         # 近似的裁剪后范数（无需额外遍历）：取 min
    #         post_norm = min(pre_val, float(grad_clip))

    #         # 控制打印频率（默认每 50 步打印一次）；可根据你需求改为 100/500
    #         step = getattr(self, "total_steps_done", 0)
    #         if step % 50 == 0:
    #             print(
    #                 f"[Grad-Clip] step={step}, pre_norm={pre_val:.6f}, max_norm={grad_clip}, "
    #                 f"clipped={clipped_flag}, post_norm≈{post_norm:.6f}, checks={self._grad_clip_checks}, "
    #                 f"triggers={self._grad_clip_triggers}"
    #             )
    #         # -------------------------------------------------------
    #     else:
    #         # 未启用裁剪，返回计算的范数（Tensor 或 float 视 _compute_grad_norm 实现）
    #         grad_norm = self._compute_grad_norm(optimizer)

    #     return grad_norm

    # （GPT最后一层）验证代码
    def _debug_grad_check(self, optimizer, scaler=None, freq=100):
        """
        调试用：检查 emotion embedding / head 的 grad 是否只在 emotion ids 上，
        并扩展支持 GPT last block 解冻的情况。同时打印每个 trainable 参数的 grad norm，
        以及各个 param_group 的平均 grad norm。
        freq: 每 freq 步打印一次，0 表示每步都检查。
        调用位置：在 backward() 完成后，grad clipping/optimizer.step 之前。
        """
        # 只在主进程打印
        if getattr(self, "args", None) and getattr(self.args, "rank", 0) != 0:
            return

        # 打印频率控制
        if freq > 0 and (self.total_steps_done % freq) != 0:
            return

        if not hasattr(self, "_emotion_emb"):
            print("[Grad-Check] no _emotion_emb set on trainer, skipping.")
            return

        try:
            # 如果用 GradScaler，先 unscale 得到真实 grad
            if self.use_amp_scaler and not getattr(self, "use_apex", False) and scaler is not None:
                try:
                    scaler.unscale_(optimizer)
                except Exception as e:
                    print("[Grad-Check] scaler.unscale_ failed:", e)

            emb_grad = self._emotion_emb.grad
            headw_grad = self._emotion_head_w.grad
            headb_grad = self._emotion_head_b.grad

            def nonzero_rows_2d(g):
                if g is None:
                    return []
                nz = (g.abs().sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(1)
                return nz.cpu().tolist() if nz.numel() > 0 else []

            def nonzero_idx_1d(g):
                if g is None:
                    return []
                nz = (g.abs() > 0).nonzero(as_tuple=False).squeeze(1)
                return nz.cpu().tolist() if nz.numel() > 0 else []

            nz_emb = nonzero_rows_2d(emb_grad)
            nz_headw = nonzero_rows_2d(headw_grad)
            nz_headb = nonzero_idx_1d(headb_grad)

            # 打印样本
            print(f"[Grad-Check] step={self.total_steps_done} emb nonzero rows (sample 20): {nz_emb[:20]}")
            print(f"[Grad-Check] step={self.total_steps_done} head_w nonzero rows (sample 20): {nz_headw[:20]}")
            print(f"[Grad-Check] step={self.total_steps_done} head_b nonzero idx (sample 20): {nz_headb[:20]}")

            # ============ 允许 GPT last block ==============
            allowed_params = {self._emotion_emb, self._emotion_head_w, self._emotion_head_b}
            if hasattr(self, "_gpt_last_block"):
                allowed_params.update(set(self._gpt_last_block.parameters()))

            bad = []
            for name, p in self.model.named_parameters():
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    if p not in allowed_params:
                        bad.append((name, p.grad.abs().sum().item()))

            if bad:
                print("[Grad-Check] WARNING: unexpected params have grads:", bad[:10])  # 打印前 10 个
            else:
                if hasattr(self, "_gpt_last_block"):
                    print("[Grad-Check] OK: only emotion emb/head + GPT last block have grads.")
                else:
                    print("[Grad-Check] OK: only emotion emb/head have grads.")
            # ===================================================

            # ============ 打印所有 trainable 参数 grad norm ============
            print("\n=== Gradient Norms (per parameter) ===")
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    print(f"[NO GRAD] {name}")
                else:
                    grad_norm = param.grad.detach().data.norm(2).item()
                    print(f"[GRAD] {name}: {grad_norm:.6f}")
            print("=== End per-parameter Gradient Norms ===\n")

            # ============ 新增：param_group 平均 grad norm ============
            print("=== Gradient Norms (per param_group) ===")
            for gi, group in enumerate(optimizer.param_groups):
                norms = []
                for p in group["params"]:
                    if p.grad is not None:
                        norms.append(p.grad.detach().data.norm(2).item())
                if norms:
                    avg_norm = sum(norms) / len(norms)
                    max_norm = max(norms)
                    min_norm = min(norms)
                    print(
                        f"[Group {gi}] lr={group.get('lr')}, "
                        f"params={len(group['params'])}, "
                        f"avg_norm={avg_norm:.6f}, max_norm={max_norm:.6f}, min_norm={min_norm:.6f}"
                    )
                else:
                    print(f"[Group {gi}] lr={group.get('lr')} has no grads.")
            print("=== End param_group Gradient Norms ===\n")
            # ===================================================

        except Exception as e:
            print(f"[Grad-Check] failed:", e)

    def _check_param_value_deltas(self, topk=6, tol=1e-12, check_frozen_sample=5):
        """
        严格检查 GPT-last 参数值是否有更新（与初始快照比较），并抽样检测冻结参数是否被改动。
        - topk: 打印 max delta 的前 topk 参数
        - tol: 判定“未变化”的阈值（L2）
        - check_frozen_sample: 随机抽取多少个被冻结参数检查（优先检查保存的 frozen snapshots）
        调用时机：在 optimizer.step() 执行后且在 optimizer.zero_grad() 之前
        """
        try:
            if not hasattr(self, "_gpt_last_init") or not hasattr(self, "_gpt_last_block"):
                print("[Delta-Check] No GPT-last init saved, skipping.")
                return

            deltas = {}
            for name, param in self._gpt_last_block.named_parameters():
                init = self._gpt_last_init.get(name, None)
                if init is None:
                    print(f"[Delta-Check] init not found for {name}, skipping")
                    continue
                curr = param.detach()
                init_on_device = init.to(curr.device)
                delta = (curr - init_on_device).norm().item()
                deltas[name] = delta

            if not deltas:
                print("[Delta-Check] No deltas computed, skipping.")
                return

            sorted_items = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
            topk_items = sorted_items[:topk]
            print(f"[Delta-Check] step={self.total_steps_done} GPT-last top{topk} deltas (L2):")
            for n, d in topk_items:
                print(f"  {n}: {d:.6e}")

            max_delta = max(deltas.values())
            mean_delta = sum(deltas.values()) / len(deltas)
            print(f"[Delta-Check] max_delta={max_delta:.6e}, mean_delta={mean_delta:.6e}")

            # Check sampled frozen params (prefer saved snapshots)
            frozen_issues = []
            if hasattr(self, "_frozen_sample_init") and self._frozen_sample_init:
                for n, init in self._frozen_sample_init.items():
                    # current param
                    curr = dict(self.model.named_parameters())[n].detach().cpu()
                    delta_f = (curr - init).norm().item()
                    if delta_f > max(tol, 1e-9):
                        frozen_issues.append((n, delta_f))
                print("[Delta-Check] frozen-sample deltas (should be near 0):", frozen_issues)
            else:
                # fallback: sample some frozen params and check if they have grad
                frozen_params = [(n,p) for n,p in self.model.named_parameters() if not p.requires_grad]
                sample_n = min(check_frozen_sample, len(frozen_params))
                if sample_n > 0:
                    import random
                    sampled = random.sample(frozen_params, sample_n)
                    frozen_changes = []
                    for n, p in sampled:
                        grad_exists = (p.grad is not None and p.grad.abs().sum().item() > 0)
                        frozen_changes.append((n, grad_exists))
                    print("[Delta-Check] frozen params grad-existence (should be False):", frozen_changes)

            # Warn if last block not updated
            if max_delta < tol:
                print("[Delta-Check] WARNING: GPT-last max delta very small (< tol). Maybe optimizer didn't update parameters or lr too small.")

            # Optionally log scalars
            writer = getattr(self.dashboard_logger, "writer", None)
            if writer:
                writer.add_scalar("delta/gpt_last_max", max_delta, self.total_steps_done)
                writer.add_scalar("delta/gpt_last_mean", mean_delta, self.total_steps_done)

        except Exception as e:
            print("[Delta-Check] failed:", e)

    # （GPT最后两层）验证代码
    # def _debug_grad_check(self, optimizer, scaler=None, freq=100, max_param_print=60):
    #     """
    #     调试用：检查 emotion embedding / head 的 grad 是否只在 emotion ids 上，
    #     支持 _gpt_last_block 或 _gpt_last_blocks（多块）。
    #     freq: 每 freq 步打印一次，0 表示每步都检查。
    #     max_param_print: per-parameter 打印的最大行数（防止日志刷屏）。
    #     """
    #     # 只在主进程打印（若有分布式）
    #     if getattr(self, "args", None) and getattr(self.args, "rank", 0) != 0:
    #         return

    #     if freq > 0 and (self.total_steps_done % freq) != 0:
    #         return

    #     if not hasattr(self, "_emotion_emb"):
    #         print("[Grad-Check] no _emotion_emb set on trainer, skipping.")
    #         return

    #     try:
    #         # AMP: unscale 得到真实 grad
    #         if self.use_amp_scaler and not getattr(self, "use_apex", False) and scaler is not None:
    #             try:
    #                 scaler.unscale_(optimizer)
    #             except Exception as e:
    #                 print("[Grad-Check] scaler.unscale_ failed:", e)

    #         emb_grad = self._emotion_emb.grad
    #         headw_grad = self._emotion_head_w.grad
    #         headb_grad = self._emotion_head_b.grad

    #         def nonzero_rows_2d(g):
    #             if g is None:
    #                 return []
    #             nz = (g.abs().sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(1)
    #             return nz.cpu().tolist() if nz.numel() > 0 else []

    #         def nonzero_idx_1d(g):
    #             if g is None:
    #                 return []
    #             nz = (g.abs() > 0).nonzero(as_tuple=False).squeeze(1)
    #             return nz.cpu().tolist() if nz.numel() > 0 else []

    #         nz_emb = nonzero_rows_2d(emb_grad)
    #         nz_headw = nonzero_rows_2d(headw_grad)
    #         nz_headb = nonzero_idx_1d(headb_grad)

    #         print(f"[Grad-Check] step={self.total_steps_done} emb nonzero rows (sample 20): {nz_emb[:20]}")
    #         print(f"[Grad-Check] step={self.total_steps_done} head_w nonzero rows (sample 20): {nz_headw[:20]}")
    #         print(f"[Grad-Check] step={self.total_steps_done} head_b nonzero idx (sample 20): {nz_headb[:20]}")

    #         # ============ allowed params: emotion emb/head + gpt-last block(s) ============
    #         allowed_params = {self._emotion_emb, self._emotion_head_w, self._emotion_head_b}

    #         last_blocks = []
    #         if hasattr(self, "_gpt_last_block"):
    #             last_blocks.append(self._gpt_last_block)
    #         if hasattr(self, "_gpt_last_blocks"):
    #             # 支持 list/tuple
    #             try:
    #                 last_blocks.extend(list(self._gpt_last_blocks))
    #             except Exception:
    #                 last_blocks.append(self._gpt_last_blocks)

    #         for blk in last_blocks:
    #             try:
    #                 for p in blk.parameters():
    #                     allowed_params.add(p)
    #             except Exception:
    #                 pass

    #         # 检查非允许参数是否有 grad
    #         bad = []
    #         for name, p in self.model.named_parameters():
    #             if p.grad is not None and p.grad.abs().sum().item() > 0:
    #                 if p not in allowed_params:
    #                     bad.append((name, p.grad.abs().sum().item()))
    #         if bad:
    #             print("[Grad-Check] WARNING: unexpected params have grads (show up to 10):", bad[:10])
    #         else:
    #             if last_blocks:
    #                 print("[Grad-Check] OK: only emotion emb/head + GPT-last block(s) have grads.")
    #             else:
    #                 print("[Grad-Check] OK: only emotion emb/head have grads.")
    #         # ===================================================

    #         # ============ per-parameter grad norms (limited) ============
    #         print("\n=== Gradient Norms (per parameter, up to {} items) ===".format(max_param_print))
    #         shown = 0
    #         for name, param in self.model.named_parameters():
    #             if not param.requires_grad:
    #                 continue
    #             if param.grad is None:
    #                 continue
    #             grad_norm = param.grad.detach().data.norm(2).item()
    #             # 仅打印有 grad 的参数，并限制行数
    #             print(f"[GRAD] {name}: {grad_norm:.6f}")
    #             shown += 1
    #             if shown >= max_param_print:
    #                 break
    #         if shown == 0:
    #             print("[Grad-Check] No trainable params have gradients (unexpected).")
    #         print("=== End per-parameter Gradient Norms ===\n")

    #         # ============ param_group level summary ============
    #         print("=== Gradient Norms (per param_group summary) ===")
    #         for gi, group in enumerate(optimizer.param_groups):
    #             norms = []
    #             params_list = group.get("params", [])
    #             # flatten param list
    #             if isinstance(params_list, (list, tuple)):
    #                 params_iter = params_list
    #             else:
    #                 params_iter = [params_list]
    #             for p in params_iter:
    #                 if getattr(p, "grad", None) is not None:
    #                     norms.append(p.grad.detach().data.norm(2).item())
    #             if norms:
    #                 avg_norm = sum(norms) / len(norms)
    #                 max_norm = max(norms)
    #                 min_norm = min(norms)
    #                 try:
    #                     param_len = len(params_list) if isinstance(params_list, (list, tuple)) else (1 if params_list is not None else 0)
    #                 except Exception:
    #                     param_len = "?"
    #                 print(f"[Group {gi}] lr={group.get('lr')}, params={param_len}, avg_norm={avg_norm:.6f}, max={max_norm:.6f}, min={min_norm:.6f}")
    #             else:
    #                 print(f"[Group {gi}] lr={group.get('lr')} has no grads.")
    #         print("=== End param_group Gradient Norms ===\n")

    #     except Exception as e:
    #         print(f"[Grad-Check] failed:", e)

    # def _check_param_value_deltas(self, topk=6, tol=1e-12, check_frozen_sample=5):
    #     """
    #     检查 GPT-last 参数值是否有更新（与初始快照比较），支持多个 block。
    #     """
    #     try:
    #         if not hasattr(self, "_gpt_last_init"):
    #             print("[Delta-Check] No GPT-last init saved, skipping.")
    #             return

    #         # Build iterable of (name, param, try_keys) to compare against saved init dict
    #         to_check = []
    #         # try single-block attribute first (compat)
    #         if hasattr(self, "_gpt_last_block"):
    #             for name, param in self._gpt_last_block.named_parameters():
    #                 to_check.append((name, param))
    #         # then multi-blocks if present
    #         if hasattr(self, "_gpt_last_blocks"):
    #             try:
    #                 for blk in self._gpt_last_blocks:
    #                     for name, param in blk.named_parameters():
    #                         # if name collision between blocks, we will try to resolve by searching in init keys
    #                         to_check.append((name, param))
    #             except Exception:
    #                 pass

    #         if not to_check:
    #             print("[Delta-Check] No GPT-last block parameters found to check, skipping.")
    #             return

    #         # compute deltas; lookup strategy: exact name -> prefixed name -> suffix match
    #         deltas = {}
    #         init_map = getattr(self, "_gpt_last_init", {})  # mapping saved earlier
    #         init_keys = list(init_map.keys())

    #         for name, param in to_check:
    #             curr = param.detach()
    #             # 1) direct match
    #             init = init_map.get(name, None)
    #             # 2) if not found, try to find any init key that endswith name
    #             if init is None:
    #                 matches = [k for k in init_keys if k.endswith(name)]
    #                 if len(matches) == 1:
    #                     init = init_map[matches[0]]
    #                 elif len(matches) > 1:
    #                     # ambiguous: try exact block-prefixed keys if present
    #                     # choose the one with same shape
    #                     chosen = None
    #                     for k in matches:
    #                         cand = init_map[k]
    #                         if hasattr(cand, "shape") and cand.shape == curr.cpu().shape:
    #                             chosen = cand
    #                             break
    #                     init = chosen if chosen is not None else init_map[matches[0]]
    #             if init is None:
    #                 # 没有对应快照，跳过
    #                 # print(f"[Delta-Check] init not found for {name}, skipping")
    #                 continue
    #             init_on_device = init.to(curr.device)
    #             delta = (curr - init_on_device).norm().item()
    #             deltas[name] = delta

    #         if not deltas:
    #             print("[Delta-Check] No deltas computed, skipping.")
    #             return

    #         sorted_items = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    #         topk_items = sorted_items[:topk]
    #         print(f"[Delta-Check] step={self.total_steps_done} GPT-last top{topk} deltas (L2):")
    #         for n, d in topk_items:
    #             print(f"  {n}: {d:.6e}")

    #         max_delta = max(deltas.values())
    #         mean_delta = sum(deltas.values()) / len(deltas)
    #         print(f"[Delta-Check] max_delta={max_delta:.6e}, mean_delta={mean_delta:.6e}")

    #         if max_delta < tol:
    #             print("[Delta-Check] WARNING: GPT-last max delta very small (< tol). Maybe optimizer didn't update parameters or lr too small.")

    #         # frozen params sample check (unchanged)
    #         if hasattr(self, "_frozen_sample_init") and self._frozen_sample_init:
    #             frozen_issues = []
    #             for n, init in self._frozen_sample_init.items():
    #                 curr = dict(self.model.named_parameters())[n].detach().cpu()
    #                 delta_f = (curr - init).norm().item()
    #                 if delta_f > max(tol, 1e-9):
    #                     frozen_issues.append((n, delta_f))
    #             print("[Delta-Check] frozen-sample deltas (should be near 0):", frozen_issues)
    #         else:
    #             frozen_params = [(n,p) for n,p in self.model.named_parameters() if not p.requires_grad]
    #             sample_n = min(check_frozen_sample, len(frozen_params))
    #             if sample_n > 0:
    #                 import random
    #                 sampled = random.sample(frozen_params, sample_n)
    #                 frozen_changes = []
    #                 for n, p in sampled:
    #                     grad_exists = (p.grad is not None and p.grad.abs().sum().item() > 0)
    #                     frozen_changes.append((n, grad_exists))
    #                 print("[Delta-Check] frozen params grad-existence (should be False):", frozen_changes)

    #     except Exception as e:
    #         print("[Delta-Check] failed:", e)


    # ------------------ 新增：打印所有 param_group lr ------------------
    def _print_param_group_lrs(self, optimizer=None, prefix="[LR-all]"):
        """打印 optimizer.param_groups 的 lr / weight_decay / params_count（只 print）。"""
        optimizer = optimizer if optimizer is not None else getattr(self, "optimizer", None)
        if optimizer is None or not hasattr(optimizer, "param_groups"):
            return
        step = getattr(self, "total_steps_done", 0)
        for i, g in enumerate(optimizer.param_groups):
            lr = g.get("lr", None)
            wd = g.get("weight_decay", 0.0)
            try:
                params = g.get("params", [])
                if isinstance(params, (list, tuple)):
                    nparams = sum(p.numel() for p in params if hasattr(p, "numel"))
                else:
                    nparams = params.numel() if hasattr(params, "numel") else len(params)
            except Exception:
                nparams = "?"
            print(f"{prefix} step={step} group[{i}] lr={lr}, weight_decay={wd}, params_count={nparams}")

    # ------------------ 修改：只做 warmup 的函数（取消单独打印 GPT-last lr） ------------------
    def _apply_manual_warmup_gpt_last(self, optimizer):
        """
        线性 warmup（只对 GPT-last 分组）：需要在 optimizer 创建之后保存 self._init_group_lrs 和 self._warmup_mask。
        在每步 optimizer.step() 之前调用。
        """
        # 必要前置检查
        if not hasattr(self, "_init_group_lrs") or not hasattr(self, "_warmup_mask"):
            return

        # warmup 配置与当前 step
        # warmup_steps = 2400
        warmup_steps = 4300
        step = max(0, getattr(self, "total_steps_done", 0))

        if warmup_steps <= 0:
            factor = 1.0
        else:
            # 使用 (step+1)/warmup 避免 step=0 时 lr=0
            factor = min(1.0, float(step + 1) / float(warmup_steps))

        # （GPT最后一层）仅修改 mask 为 True 的组的 lr，避免改写其它组（以免覆盖 scheduler）
        """ for i, (g, init_lr) in enumerate(zip(optimizer.param_groups, self._init_group_lrs)):
            if self._warmup_mask and i < len(self._warmup_mask) and self._warmup_mask[i]:
                g["lr"] = init_lr * factor
            else:
                # 保持其它组 lr 不变（不要覆盖 scheduler 的 lr）
                pass """

        # （GPT最后两层）遍历 param_groups（使用 zip 保持原有结构），仅对 mask 为 True 的组改变 lr
        for i, (g, init_lr) in enumerate(zip(optimizer.param_groups, self._init_group_lrs)):
            try:
                is_gpt_last = bool(self._warmup_mask and i < len(self._warmup_mask) and self._warmup_mask[i])
            except Exception:
                is_gpt_last = False

            if is_gpt_last:
                try:
                    init_lr_f = float(init_lr)
                except Exception:
                    # 如果无法转 float，跳过该组（以免抛异常）
                    continue
                g["lr"] = init_lr_f * factor
            else:
                # 不改写非 GPT-last 组（避免覆盖 scheduler）
                pass

        # 每 50 步打印所有组 lr，方便人工核对（且只在此处打印，不再单独打印 GPT-last）
        if step % 50 == 0:
            try:
                self._print_param_group_lrs(optimizer, prefix="[LR-all]")
            except Exception:
                pass


    def optimize(
        self,
        batch: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: "AMPScaler",
        criterion: nn.Module,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List, Dict],  # pylint: disable=protected-access
        config: Coqpit,
        optimizer_idx: int = None,
        step_optimizer: bool = True,
        num_optimizers: int = 1,
    ) -> Tuple[Dict, Dict, int]:
        """Perform a forward - backward pass and run the optimizer.

        Args:
            batch (Dict): Input batch. If
            model (nn.Module): Model for training. Defaults to None.
            optimizer (Union[nn.optim.Optimizer, List]): Model's optimizer. If it is a list then, `optimizer_idx` must be defined to indicate the optimizer in use.
            scaler (AMPScaler): AMP scaler.
            criterion (nn.Module): Model's criterion.
            scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler used by the optimizer.
            config (Coqpit): Model config.
            optimizer_idx (int, optional): Target optimizer being used. Defaults to None.
            step_optimizer (bool, optional): Whether step the optimizer. If False, gradients are accumulated and
                model parameters are not updated. Defaults to True.
            num_optimizers (int, optional): Number of optimizers. Defaults to 1.

        Raises:
            RuntimeError: When the loss is NaN.

        Returns:
            Tuple[Dict, Dict, int, torch.Tensor]: model outputs, losses, step time and gradient norm.
        """
        # print("正在执行optimize函数")
        step_start_time = time.time()

        # forward pass and loss computation
        outputs, loss_dict = self._compute_loss(
            batch=batch, model=model, criterion=criterion, config=config, optimizer_idx=optimizer_idx
        )

        # skip the rest if not outputs from the model
        if not loss_dict:
            step_time = time.time() - step_start_time
            return outputs, {}, step_time

        grad_clip = self._set_grad_clip_per_optimizer(config=config, optimizer_idx=optimizer_idx)
        # optimizer step
        grad_norm = 0
        update_lr_scheduler = True

        # callback
        self.callbacks.before_backward_pass(self, loss_dict)

        # accumulated gradients adjustment
        loss_dict["loss"] = loss_dict["loss"] / float(self.grad_accum_steps)

        if self.use_accelerate:
            with self.accelerator.accumulate(model):
                ctx_mgr = self.accelerator.autocast if config.mixed_precision else nullcontext
                with ctx_mgr():
                    self.accelerator.backward(loss_dict["loss"])
                    grad_norm = self._compute_grad_norm(optimizer)
                    if self.accelerator.sync_gradients and grad_clip is not None and grad_clip > 0:
                        self.accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    if not self.config.scheduler_after_epoch and not self.accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        else:
            if self.use_amp_scaler:
                if self.use_apex:
                    # TODO: verify AMP use for GAN training in TTS
                    # https://nvidia.github.io/apex/advanced.html?highlight=accumulate#backward-passes-with-multiple-optimizers
                    with amp.scale_loss(loss_dict["loss"], optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if step_optimizer:
                        grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=None)
                else:
                    # model optimizer step in mixed precision mode
                    scaler.scale(loss_dict["loss"]).backward()
                    # gradient accumulation
                    if step_optimizer:
                        grad_norm = self._grad_clipping(grad_clip=grad_clip, optimizer=optimizer, scaler=scaler)
                        scale_prev = scaler.get_scale()
                        scaler.step(optimizer)
                        # update the scaler at the end of all the optimizer steps
                        if optimizer_idx is None or (optimizer_idx + 1 == num_optimizers):
                            scaler.update()
                            loss_dict["amp_scaler"] = scaler.get_scale()  # for logging
                        update_lr_scheduler = scale_prev <= scaler.get_scale()
            else:
                # main model optimizer step
                loss_dict["loss"].backward()

                # 检查 emotion embedding / head 的 grad 是否只在 emotion ids 上，并扩
                # 展支持 GPT last block 解冻的情况。同时打印每个 trainable 参数的 grad norm，
                # 以及各个 param_group 的平均 grad norm。
                # freq: 每 freq 步打印一次，0 表示每步都检查。

                # self._debug_grad_check(optimizer, scaler=None, freq=10)

                

                # gradient accumulation
                # 调用_grad_clipping修正grad_norm，比下面方法好
                if step_optimizer:
                    grad_norm = self._grad_clipping(grad_clip, optimizer, scaler)
                    
                    # —— 微调GPT模块时warmup（先调 lr，再 step）——
                    self._apply_manual_warmup_gpt_last(optimizer)
                    
                    optimizer.step()
                """ if step_optimizer:
                    self.callbacks.before_gradient_clipping(self)
                    if grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params(optimizer), grad_clip)
                    else:
                        # 直接调用_compute_grad_norm修正grad_norm
                        grad_norm = self._compute_grad_norm(optimizer)  # <<< 新增这一行
                    optimizer.step() """

                # --- MOD: （GPT最后一层或两层检查代码）Delta check after optimizer.step() (non-AMP branch) ---
                # 每 delta_check_freq 步严格检查一次（避免日志刷屏）
                """ delta_check_freq = getattr(self.config, "delta_check_freq", 50)
                if delta_check_freq > 0 and (self.total_steps_done % delta_check_freq) == 0:
                    try:
                        self._check_param_value_deltas(topk=6, tol=1e-12, check_frozen_sample=5)
                    except Exception as e:
                        print("[Delta-Check] invocation failed:", e) """
                # --- END MOD ---

                # ←—— （只微调情感embedding）监控代码放在这里
                """ if hasattr(self, "_emotion_emb"):
                    with torch.no_grad():
                        curr_emb = self._emotion_emb.index_select(0, self._emotion_ids)
                        curr_head_w = self._emotion_head_w.index_select(0, self._emotion_ids)
                        curr_head_b = self._emotion_head_b.index_select(0, self._emotion_ids)

                        delta_emb = (curr_emb - self._emotion_init_emb).norm().item()
                        delta_head_w = (curr_head_w - self._emotion_init_head_w).norm().item()
                        delta_head_b = (curr_head_b - self._emotion_init_head_b).norm().item()

                        print(f"[Param-Monitor] step={self.total_steps_done}, "
                            f"Δ_emb={delta_emb:.6f}, Δ_head_w={delta_head_w:.6f}, Δ_head_b={delta_head_b:.6f}")

                        # 可选 tensorboard
                        writer = getattr(self.dashboard_logger, "writer", None)
                        if writer:
                            for i, tokid in enumerate(self._emotion_ids):
                                writer.add_scalar(f"emotion/Δ_emb_token_{tokid}", (curr_emb[i]-self._emotion_init_emb[i]).norm().item(), self.total_steps_done)
                                writer.add_scalar(f"emotion/Δ_head_w_token_{tokid}", (curr_head_w[i]-self._emotion_init_head_w[i]).norm().item(), self.total_steps_done)
                                writer.add_scalar(f"emotion/Δ_head_b_token_{tokid}", (curr_head_b[i]-self._emotion_init_head_b[i]).norm().item(), self.total_steps_done)

                        self._emotion_prev_emb = curr_emb.clone()
                        self._emotion_prev_head_w = curr_head_w.clone()
                        self._emotion_prev_head_b = curr_head_b.clone()
 """
            
            # setup lr
            if (
                scheduler is not None
                and update_lr_scheduler
                and not self.config.scheduler_after_epoch
                and step_optimizer
            ):
                scheduler.step()

            # zero-out optimizer
            if step_optimizer:
                optimizer.zero_grad(set_to_none=True)

        # pytorch skips the step when the norm is 0. So ignore the norm value when it is NaN
        if isinstance(grad_norm, torch.Tensor) and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            grad_norm = 0

        step_time = time.time() - step_start_time

        # detach loss dict
        loss_dict_detached = self.detach_loss_dict(loss_dict, step_optimizer, optimizer_idx, grad_norm)
        return outputs, loss_dict_detached, step_time

    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        """Perform a training step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            batch_n_steps (int): Number of steps needed to complete an epoch. Needed for logging.
            step (int): Current step number in this epoch.
            loader_start_time (float): The time when the data loading is started. Needed for logging.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        # 把总步数和当前步数写到模型中，方便gpt_trainer.py文件中的train_step方法获取
        # --- 将 trainer 的步数信息写入真实 model/module，供 model.train_step 使用 ---
        _target_model = self.model.module if hasattr(self.model, "module") else self.model
        try:
            # 全局已完成步数（从 Trainer 维护的 total_steps_done）
            setattr(_target_model, "_trainer_global_step", int(self.total_steps_done))
            # 本 epoch 的总步骤数（loader 的 batch 数）
            # setattr(_target_model, "_trainer_epoch_num_steps", int(batch_n_steps))
            # 本 epoch 内的当前步骤索引（0..batch_n_steps-1）
            # setattr(_target_model, "_trainer_step_in_epoch", int(step))
            # 可选：已经完成的 epoch 数
            # setattr(_target_model, "_trainer_epochs_done", int(self.epochs_done))
        except Exception:
            # 容错：不应阻塞训练
            pass
        # --- end injection ---


        # print("正在执行train_step函数")
        self.callbacks.on_train_step_start(self)
        # format data
        batch = self.format_batch(batch)
        # print(f"format后的batch为：{batch}")
        loader_time = time.time() - loader_start_time

        # conteainers to hold model outputs and losses for each optimizer.
        outputs_per_optimizer = None
        loss_dict = {}

   

        # OPTIMIZATION
        if isimplemented(self.model, "optimize"):  # pylint: disable=too-many-nested-blocks
            # custom optimize for the model
            step_time = time.time()
            device, dtype = self._get_autocast_args(self.config.mixed_precision, self.config.precision)
            with torch.autocast(device_type=device, dtype=dtype, enabled=self.config.mixed_precision):
                outputs, loss_dict_new = self.model.optimize(
                    batch,
                    self,
                )
            step_time = time.time() - step_time
            # If None, skip the step
            if outputs is None:
                return None, None
            # TODO: find a way to log grad_norm for custom optimize
            loss_dict_new = self.detach_loss_dict(loss_dict_new, True, None, None)
            loss_dict.update(loss_dict_new)
        else:
            # gradient accumulation
            # TODO: grad accumulation for each optimizer
            step_optimizer = True
            if ((step + 1) % self.grad_accum_steps != 0) and (step + 1 != batch_n_steps):
                step_optimizer = False

            if not isinstance(self.optimizer, list):
                # auto training with a single optimizer
                outputs, loss_dict_new, step_time = self.optimize(
                    batch,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.criterion,
                    self.scheduler,
                    self.config,
                    step_optimizer=step_optimizer,
                    num_optimizers=1,
                )
                loss_dict.update(loss_dict_new)
            else:
                # auto training with multiple optimizers (e.g. GAN)
                outputs_per_optimizer = [None] * len(self.optimizer)
                total_step_time = 0
                for idx, optimizer in enumerate(self.optimizer):
                    criterion = self.criterion
                    # scaler = self.scaler[idx] if self.use_amp_scaler else None
                    scaler = self.scaler
                    scheduler = None
                    if self.scheduler is not None:
                        scheduler = self.scheduler[idx]
                    outputs, loss_dict_new, step_time = self.optimize(
                        batch,
                        self.model,
                        optimizer,
                        scaler,
                        criterion,
                        scheduler,
                        self.config,
                        idx,
                        step_optimizer=step_optimizer,
                        num_optimizers=len(self.optimizer),
                    )
                    # skip the rest if the model returns None
                    total_step_time += step_time
                    outputs_per_optimizer[idx] = outputs
                    # merge loss_dicts from each optimizer
                    # rename duplicates with the optimizer idx
                    # if None, model skipped this optimizer
                    if loss_dict_new is not None:
                        for k, v in loss_dict_new.items():
                            if k in loss_dict:
                                loss_dict[f"{k}-{idx}"] = v
                            else:
                                loss_dict[k] = v
                    step_time = total_step_time

                outputs = outputs_per_optimizer

                # clear any pesky gradients after gradient accumulation
                if step_optimizer:
                    self.model.zero_grad(set_to_none=True)

        # update avg runtime stats
        keep_avg_update = {}
        keep_avg_update["avg_loader_time"] = loader_time
        keep_avg_update["avg_step_time"] = step_time
        self.keep_avg_train.update_values(keep_avg_update)

        # update avg loss stats
        update_eval_values = {}
        for key, value in loss_dict.items():
            update_eval_values["avg_" + key] = value
        self.keep_avg_train.update_values(update_eval_values)

        # print training progress
        if self.total_steps_done % self.config.print_step == 0:
            # log learning rates
            lrs = {}
            if isinstance(self.optimizer, list):
                for idx, optimizer in enumerate(self.optimizer):
                    current_lr = self.optimizer[idx].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{idx}": current_lr})
            elif isinstance(self.optimizer, dict):
                for key, optimizer in self.optimizer.items():
                    current_lr = self.optimizer[key].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{key}": current_lr})
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                lrs = {"current_lr": current_lr}

            # log run-time stats
            loss_dict.update(lrs)
            loss_dict.update(
                {
                    "step_time": round(step_time, 4),
                    "loader_time": round(loader_time, 4),
                }
            )
            self.c_logger.print_train_step(
                batch_n_steps,
                step,
                self.total_steps_done,
                loss_dict,
                self.keep_avg_train.avg_values,
            )

        if self.args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load and don't log every step
            if self.total_steps_done % self.config.plot_step == 0:
                self.dashboard_logger.train_step_stats(self.total_steps_done, loss_dict)
            if self.total_steps_done % self.config.save_step == 0 and self.total_steps_done != 0:
                if self.config.save_checkpoints:
                    # checkpoint the model
                    self.save_checkpoint()

            if self.total_steps_done % self.config.log_model_step == 0:
                # log checkpoint as artifact
                self.update_training_dashboard_logger(batch=batch, outputs=outputs)

            self.dashboard_logger.flush()




        self.total_steps_done += 1
        self.callbacks.on_train_step_end(self)
        return outputs, loss_dict

    def train_epoch(self) -> None:
        """Main entry point for the training loop. Run training on the all training samples."""
        # print("正在执行train_epoch函数")
        # initialize the data loader
        if self.train_loader is None:
            self.train_loader = self.get_train_dataloader(
                self.training_assets,
                self.train_samples,
                verbose=True,
            )
            self.train_loader = self.prepare_accelerate_loader(self.train_loader)


        # -------测试每个前几个batch中每种情感的数量------
        # def test_loader(loader, emotion_ids, n_batches=10):
        #     """快速测试 DataLoader 是否按预期返回每 batch 包含所有情感 token。
        #     loader: DataLoader
        #     emotion_ids: list[int]
        #     n_batches: 检查前 n_batches 个 batch
        #     """
        #     emotion_set = set(emotion_ids)
        #     batches_checked = 0
        #     report = {"ok_batches": 0, "bad_batches": 0, "per_emotion_count": Counter()}
        #     for i, batch in enumerate(loader):
        #         if i >= n_batches:
        #             break
        #         # 期望 collate_fn 产生 'padded_text'：形状 [B, T]
        #         if "padded_text" not in batch:
        #             print("[test_loader] Error: batch 没有 'padded_text' key，现有 keys:", list(batch.keys()))
        #             return
        #         padded = batch["padded_text"]  # Tensor [B, T] or np
        #         if isinstance(padded, torch.Tensor):
        #             padded = padded.cpu()
        #         B = padded.shape[0]
        #         found_all = True
        #         # 对每情感计数是否出现
        #         batch_emotions = set()
        #         for b in range(B):
        #             row = padded[b]
        #             for tid in emotion_ids:
        #                 if (row == tid).any():
        #                     batch_emotions.add(tid)
        #                     report["per_emotion_count"][tid] += 1
        #         # 检查是否齐全
        #         if batch_emotions >= emotion_set:
        #             report["ok_batches"] += 1
        #         else:
        #             report["bad_batches"] += 1
        #             print(f"[test_loader] batch {i} 缺失情感：present={sorted(batch_emotions)} missing={sorted(list(emotion_set - batch_emotions))}")
        #         batches_checked += 1

        #     print(f"[test_loader] Checked {batches_checked} batches. ok={report['ok_batches']}, bad={report['bad_batches']}")
        #     print("[test_loader] per_emotion_count (in checked batches):")
        #     for tid in emotion_ids:
        #         print(f"  token {tid}: {report['per_emotion_count'][tid]} occurrences")
        #     return report

        # 如果 prepare_accelerate_loader 不影响 self.train_loader，可以直接用 self.train_loader
        # test_loader(self.train_loader, [6681,6682,6683,6684,6685], n_batches=10)
        # --------------------end--------------


        # set model to training mode
        torch.set_grad_enabled(True)
        if self.num_gpus > 1:
            self.model.module.train()
        else:
            self.model.train()
        epoch_start_time = time.time()

        self.callbacks.on_train_epoch_start(self)

        self.c_logger.print_train_start()
        loader_start_time = time.time()
        
        # TRAINING EPOCH -> iterate over the training samples
        batch_num_steps = len(self.train_loader)
        for cur_step, batch in enumerate(self.train_loader):
            # filenames = batch["filenames"]
            # print(f"batch[filenames]:{filenames}")

            # print(f"cur_step:{cur_step}")
            # print(f"batch:{batch}")
            outputs, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
            if outputs is None:
                logger.info(" [!] `train_step()` retuned `None` outputs. Skipping training step.")
                continue
            del outputs
            loader_start_time = time.time()

            # RUN EVAL -> run evaluation epoch in the middle of training. Useful for big datasets.
            if self.config.run_eval_steps is not None and (self.total_steps_done % self.config.run_eval_steps == 0):
                self.eval_epoch()
                if self.num_gpus > 1:
                    self.model.module.train()
                else:
                    self.model.train()
                torch.set_grad_enabled(True)

        epoch_time = time.time() - epoch_start_time
        self.callbacks.on_train_epoch_end(self)

        # scheduler step
        if self.scheduler is not None and self.config.scheduler_after_epoch:
            if isinstance(self.scheduler, list):
                for scheduler in self.scheduler:
                    if scheduler is not None:
                        scheduler.step()
            elif isinstance(self.scheduler, dict):  # only with `model.optimize()``
                for scheduler in self.scheduler.values():
                    if scheduler is not None:
                        scheduler.step()
            else:
                self.scheduler.step()
        # plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.dashboard_logger.train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.model_param_stats:
                self.dashboard_logger.model_weights(self.model, self.total_steps_done)
        torch.cuda.empty_cache()

    #######################
    # EVAL FUNCTIONS
    #######################

    def _model_eval_step(
        self, batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a evaluation forward pass. Compute model outputs and losses with no gradients.

        Args:
            batch (Dict): IBatch of inputs.
            model (nn.Module): Model to call evaluation.
            criterion (nn.Module): Model criterion.
            optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: model outputs and losses.
        """
        input_args = [batch, criterion]

        if isimplemented(model, "optimize"):
            if hasattr(model, "module"):
                return model.module.eval_step(batch, self)
            return model.eval_step(batch, self)

        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        if hasattr(model, "module"):
            return model.module.eval_step(*input_args)

        return model.eval_step(*input_args)

    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        """Perform a evaluation step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            step (int): Current step number in this epoch.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        with torch.no_grad():
            outputs = []
            loss_dict = {}
            if not isinstance(self.optimizer, list) or isimplemented(self.model, "optimize"):
                outputs, loss_dict = self._model_eval_step(batch, self.model, self.criterion)
                if outputs is None:
                    return None, None
            else:
                outputs = [None] * len(self.optimizer)
                for idx, _ in enumerate(self.optimizer):
                    criterion = self.criterion
                    outputs_, loss_dict_new = self._model_eval_step(batch, self.model, criterion, idx)
                    if outputs_ is None:
                        return None, None
                    outputs[idx] = outputs_

                    if loss_dict_new:
                        loss_dict_new[f"loss_{idx}"] = loss_dict_new.pop("loss")
                        loss_dict.update(loss_dict_new)

            loss_dict = self._detach_loss_dict(loss_dict)

            # update avg stats
            update_eval_values = {}
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            self.keep_avg_eval.update_values(update_eval_values)

            if self.config.print_eval:
                self.c_logger.print_eval_step(step, loss_dict, self.keep_avg_eval.avg_values)

        return outputs, loss_dict

    def eval_epoch(self) -> None:
        """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""

        # initialize it when eval_epoch is called alone.
        self.keep_avg_eval = KeepAverage() if self.keep_avg_eval is None else self.keep_avg_eval

        if self.eval_loader is None:
            self.eval_loader = (
                self.get_eval_dataloader(
                    self.training_assets,
                    self.eval_samples,
                    verbose=True,
                )
                if self.config.run_eval
                else None
            )

        torch.set_grad_enabled(False)
        self.model.eval()
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        batch = None
        outputs = None
        for cur_step, batch in enumerate(self.eval_loader):
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
            outputs_, _ = self.eval_step(batch, cur_step)
            if outputs_ is None:
                logger.info(" [!] `eval_step()` retuned `None` outputs. Skipping evaluation step.")
                continue
            outputs = outputs_
            loader_start_time = time.time()
        # plot epoch stats, artifacts and figures
        if self.args.rank == 0 and outputs is not None:
            if hasattr(self.model, "module") and isimplemented(self.model.module, "eval_log"):
                self.model.module.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            elif isimplemented(self.model, "eval_log"):
                self.model.eval_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            self.dashboard_logger.eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)
        torch.cuda.empty_cache()

    ##################################
    # TESTING
    ##################################
    def test_run(self) -> None:
        """Run model test.

        Test run is expected to pass over test samples and produce logging artifacts.

        If ```model.test_run()``` is defined, it will be called and it is expected to set and execute everything
        in the model.

        Else if  ```mode.test()``` is defined, it will be called and it takes an test data loader as an argument
        and iterate over it.
        """
        self.model.eval()
        test_outputs = None
        if isimplemented(self.model, "test_run") or (
            self.num_gpus > 1 and isimplemented(self.model.module, "test_run")
        ):
            # handle everything in ```model.test_run()`
            if self.num_gpus > 1:
                test_outputs = self.model.module.test_run(self.training_assets)
            else:
                test_outputs = self.model.test_run(self.training_assets)
        elif isimplemented(self.model, "test") or (self.num_gpus > 1 and isimplemented(self.model.module, "test")):
            self.test_loader = self.get_test_dataloader(
                self.training_assets,
                self.test_samples if self.test_samples else self.eval_samples,
                verbose=True,
            )
            # use test_loader to load test samples
            if self.num_gpus > 1:
                test_outputs = self.model.module.test(self.training_assets, self.test_loader, None)
            else:
                test_outputs = self.model.test(self.training_assets, self.test_loader, None)
        if isimplemented(self.model, "test_log") or (
            self.num_gpus > 1 and isimplemented(self.model.module, "test_log")
        ):
            if self.num_gpus > 1:
                self.model.module.test_log(
                    test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done
                )
            else:
                self.model.test_log(test_outputs, self.dashboard_logger, self.training_assets, self.total_steps_done)

    def _restore_best_loss(self):
        """Restore the best loss from the args.best_path if provided else
        from the model (`args.continue_path`) used for resuming the training"""
        if self.args.continue_path and (self.restore_step != 0 or self.args.best_path):
            logger.info(" > Restoring best loss from %s ...", os.path.basename(self.args.best_path))
            ch = load_fsspec(self.args.restore_path, map_location="cpu")
            if "model_loss" in ch:
                if isinstance(ch["model_loss"], dict):
                    self.best_loss = ch["model_loss"]
                # For backwards-compatibility:
                elif isinstance(ch["model_loss"], float):
                    if self.config.run_eval:
                        self.best_loss = {"train_loss": None, "eval_loss": ch["model_loss"]}
                    else:
                        self.best_loss = {"train_loss": ch["model_loss"], "eval_loss": None}
            logger.info(" > Starting with loaded last best loss %s", self.best_loss)

    def test(self, model=None, test_samples=None) -> None:
        """Run evaluation steps on the test data split. You can either provide the model and the test samples
        explicitly or the trainer use values from the initialization.

        Args:
            model (nn.Module, optional): Model to use for testing. If None, use the model given in the initialization.
                Defaults to None.

            test_samples (List[str], optional): List of test samples to use for testing. If None, use the test samples
                given in the initialization. Defaults to None.
        """

        logger.info(" > USING TEST SET...")
        self.keep_avg_eval = KeepAverage()

        if model is not None:
            self.model = model

        eval_samples_cache = self.eval_samples
        if test_samples is not None:
            self.eval_samples = test_samples
        else:
            self.eval_samples = self.test_samples

        self.eval_epoch()
        self.c_logger.print_epoch_end(self.epochs_done, self.keep_avg_eval.avg_values)
        self.eval_samples = eval_samples_cache

    ###################################
    # FIT FUNCTIONS
    ###################################

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        # print("正在执行_fit函数")
        self._restore_best_loss()

        self.total_steps_done = self.restore_step

        #断点续训时，改变起始epoch为上次终止训练时的epoch
        # for epoch in range(self.restore_epoch, self.config.epochs): 
        for epoch in range(0, self.config.epochs):
            if self.num_gpus > 1:
                # let all processes sync up before starting with a new epoch of training
                dist.barrier()
            self.callbacks.on_epoch_start(self)
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs, self.output_path)
            if not self.skip_train_epoch and not self.start_with_eval:
                self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank <= 0:
                self.test_run()

            self.c_logger.print_epoch_end(
                epoch,
                self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values,
            )
            if self.args.rank in [None, 0]:
                self.save_best_model()
            self.callbacks.on_epoch_end(self)
            self.start_with_eval = False

    def fit_with_largest_batch_size(self, starting_batch_size=2048) -> None:
        cuda_meminfo()
        bs = starting_batch_size
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                gc.collect()
                torch.cuda.empty_cache()
                self.config.batch_size = bs
                logger.info(" > current batch size: %i", self.config.batch_size)
                self._fit()
            except RuntimeError as exception:
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as exception:  # pylint: disable=broad-except
                # catches the torch.cuda.OutOfMemoryError
                if bs > 1 and should_reduce_batch_size(exception):
                    bs //= 2
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            else:
                break

    def fit(self) -> None:
        """Where the ✨️magic✨️ happens..."""
        # print("正在执行fit函数") 
        try:
            # if self.use_accelerate:
            #     self.setup_accelerate()

            self._fit()
            if self.args.rank == 0:
                self.dashboard_logger.finish()
        except KeyboardInterrupt:
            logger.info(" > Keyboard interrupt detected.")
            if self.config.save_on_interrupt:
                logger.info(" > Saving model before exiting...")
                # save the model on keyboard interrupt
                self.save_checkpoint()
                # update the training dashboard logger
                self.update_training_dashboard_logger()
            # call the keyboard interrupt callback
            self.callbacks.on_keyboard_interrupt(self)
            # if the output folder is empty remove the run.
            remove_experiment_folder(self.output_path)
            # clear the DDP processes
            if self.num_gpus > 1:
                dist.destroy_process_group()
            # finish the wandb run and sync data
            if self.args.rank == 0:
                self.dashboard_logger.finish()
            # stop without error signal
            try:
                sys.exit(1)
            except SystemExit:
                os._exit(1)  # pylint: disable=protected-access
        except BaseException:  # pylint: disable=broad-except
            remove_experiment_folder(self.output_path)
            traceback.print_exc()
            sys.exit(1)

    def profile_fit(self, torch_profiler, epochs=None, small_run=None):
        """Run training under the torch profiler.

        Example::
            Run torch profiler to profile CPU, GPU and memory usage with Tensorboard logging.

            >>> import torch
            >>> profiler = torch.profiler.profile(
            >>>        activities=[
            >>>         torch.profiler.ProfilerActivity.CPU,
            >>>         torch.profiler.ProfilerActivity.CUDA,
            >>>     ],
            >>>     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            >>>     on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler/"),
            >>>     record_shapes=True,
            >>>     profile_memory=True,
            >>>     with_stack=True,
            >>> )
            >>> prof = trainer.profile_fit(profiler, epochs=1, small_run=64)
        """
        self.dashboard_logger = DummyLogger()
        # train the model for a custom number of epochs
        if epochs:
            self.config.epocshs = epochs
        # use a smaller set of training samples for profiling
        if small_run:
            self.setup_small_run(small_run)
        # run profiler
        self.config.run_eval = False
        self.config.test_delay_epochs = 9999999
        self.config.epochs = epochs
        # set a callback to progress the profiler
        self.callbacks_on_train_step_end = [  # pylint: disable=attribute-defined-outside-init
            lambda trainer: trainer.torch_profiler.step()
        ]
        # set the profiler to access in the Trainer
        self.torch_profiler = torch_profiler  # pylint: disable=attribute-defined-outside-init
        # set logger output for Tensorboard
        # self.torch_profiler.on_trace_ready = torch.profiler.tensorboard_trace_handler(self.output_path)
        self.torch_profiler.start()
        self.fit()
        self.torch_profiler.stop()
        return self.torch_profiler

    @rank_zero_only
    def save_best_model(self) -> None:
        """Save the best model. It only saves if the current target loss is smaller then the previous."""

        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        # save the model and update the best_loss
        self.best_loss = save_best_model(
            {"train_loss": train_loss, "eval_loss": eval_loss},
            self.best_loss,
            self.config,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            keep_all_best=True,
            keep_after=self.config.save_best_after,
            save_func=self.dashboard_logger.save_model,
        )

    @rank_zero_only
    def save_checkpoint(self) -> None:
        """Save the current model checkpoint."""
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)

        save_checkpoint(
            self.config,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            model_loss={"train_loss": train_loss, "eval_loss": eval_loss},
            save_n_checkpoints=self.config.save_n_checkpoints,
            save_func=self.dashboard_logger.save_model,
        )

    @rank_zero_only
    def update_training_dashboard_logger(self, batch=None, outputs=None):
        aliases = [
            f"epoch-{self.epochs_done}",
            f"step-{self.total_steps_done}",
        ]
        self.dashboard_logger.add_artifact(
            file_or_dir=self.output_path, name="checkpoint", artifact_type="model", aliases=aliases
        )

        # training visualizations
        if batch is not None and outputs is not None:
            if hasattr(self.model, "module") and isimplemented(self.model.module, "train_log"):
                self.model.module.train_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )
            elif isimplemented(self.model, "train_log"):
                self.model.train_log(
                    batch,
                    outputs,
                    self.dashboard_logger,
                    self.training_assets,
                    self.total_steps_done,
                )

    #####################
    # GET FUNCTIONS
    #####################

    @staticmethod
    def get_optimizer(model: nn.Module, config: Coqpit) -> Union[torch.optim.Optimizer, List]:
        """Receive the optimizer from the model if model implements `get_optimizer()` else
        check the optimizer parameters in the config and try initiating the optimizer.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A optimizer or a list of optimizers. GAN models define a list.
        """
        optimizer = None
        if isimplemented(model, "get_optimizer"):
            try:
                optimizer = model.get_optimizer()
            except NotImplementedError:
                optimizer = None
        if optimizer is None:
            optimizer_name = config.optimizer
            optimizer_params = {} if config.optimizer_params is None else config.optimizer_params
            return get_optimizer(optimizer_name, optimizer_params, config.lr, model)
        return optimizer

    @staticmethod
    def get_lr(model: nn.Module, config: Coqpit) -> Union[float, List[float]]:
        """Set the initial learning rate by the model if model implements `get_lr()` else try setting the learning rate
        fromthe config.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[float, List[float]]: A single learning rate or a list of learning rates, one for each optimzier.
        """
        lr = None
        if isimplemented(model, "get_lr"):
            try:
                lr = model.get_lr()
            except NotImplementedError:
                lr = None
        if lr is None:
            lr = config.lr
        return lr

    @staticmethod
    def get_scheduler(
        model: nn.Module, config: Coqpit, optimizer: Union[torch.optim.Optimizer, List, Dict]
    ) -> Union[torch.optim.lr_scheduler._LRScheduler, List]:  # pylint: disable=protected-access
        """Receive the scheduler from the model if model implements `get_scheduler()` else
        check the config and try initiating the scheduler.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List, Dict]: A scheduler or a list of schedulers, one for each optimizer.
        """
        scheduler = None
        if isimplemented(model, "get_scheduler"):
            try:
                scheduler = model.get_scheduler(optimizer)
            except NotImplementedError:
                scheduler = None
            if isinstance(scheduler, dict) and not isimplemented(model, "optimize"):
                raise ValueError(
                    " [!] Dictionary of schedulers are only supported with the manual optimization `model.optimize()`."
                )
        if scheduler is None:
            lr_scheduler = config.lr_scheduler
            lr_scheduler_params = config.lr_scheduler_params
            return get_scheduler(lr_scheduler, lr_scheduler_params, optimizer)
        return scheduler

    @staticmethod
    def restore_scheduler(
        scheduler: Union["Scheduler", List, Dict], args: Coqpit, config: Coqpit, restore_epoch: int, restore_step: int
    ) -> Union["Scheduler", List]:
        """Restore scheduler wrt restored model."""
        if scheduler is not None:  # pylint: disable=too-many-nested-blocks
            if args.continue_path:
                if isinstance(scheduler, list):
                    for s in scheduler:
                        if s is not None:
                            if config.scheduler_after_epoch:
                                s.last_epoch = restore_epoch
                            else:
                                s.last_epoch = restore_step
                elif isinstance(scheduler, dict):
                    for s in scheduler.values():
                        if s is not None:
                            if config.scheduler_after_epoch:
                                s.last_epoch = restore_epoch
                            else:
                                s.last_epoch = restore_step
                else:
                    if config.scheduler_after_epoch:
                        scheduler.last_epoch = restore_epoch
                    else:
                        scheduler.last_epoch = restore_step
        return scheduler

    @staticmethod
    def get_criterion(model: nn.Module) -> nn.Module:
        """Receive the criterion from the model. Model must implement `get_criterion()`.

        Args:
            model (nn.Module): Training model.

        Returns:
            nn.Module: Criterion layer.
        """
        criterion = None
        criterion = model.get_criterion()
        return criterion

    ####################
    # HELPER FUNCTIONS
    ####################

    @staticmethod
    def _detach_loss_dict(loss_dict: Dict) -> Dict:
        """Detach loss values from autograp.

        Args:
            loss_dict (Dict): losses.

        Returns:
            Dict: losses detached from autograph.
        """
        loss_dict_detached = {}
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_detached[key] = value
            else:
                loss_dict_detached[key] = value.detach().cpu().item()
        return loss_dict_detached

    def _pick_target_avg_loss(self, keep_avg_target: KeepAverage) -> Dict:
        """Pick the target loss to compare models"""

        # if the keep_avg_target is None or empty return None
        if keep_avg_target is None or len(list(keep_avg_target.avg_values.keys())) == 0:
            return None

        target_avg_loss = None
        # return if target loss defined in the model config
        # if not available in Dict use loss_1 as by default loss
        if "target_loss" in self.config and self.config.target_loss:
            if f"avg_{self.config.target_loss}" in keep_avg_target.avg_values.keys():
                return keep_avg_target[f"avg_{self.config.target_loss}"]

            raise ValueError(
                " [!] Target loss not found in the keep_avg_target. You might be exiting the training loop before it is computed or set the target_loss in the model config incorrectly."
            )

        # take the average of loss_{optimizer_idx} as the target loss when there are multiple optimizers
        if isinstance(self.optimizer, list):
            target_avg_loss = 0
            for idx in range(len(self.optimizer)):
                if f"avg_loss_{idx}" in keep_avg_target.avg_values:
                    target_avg_loss += keep_avg_target[f"avg_loss_{idx}"]
            target_avg_loss /= len(self.optimizer)
        else:
            target_avg_loss = keep_avg_target.avg_values.get("avg_loss", 0)
        return target_avg_loss

    def _setup_logger_config(self, log_file: str) -> None:
        """Set up the logger based on the process rank in DDP."""

        logger_new = logging.getLogger("trainer")
        handler = logging.FileHandler(log_file, mode="a")
        fmt = logging.Formatter("")
        handler.setFormatter(fmt)
        logger_new.addHandler(handler)

        # only log to a file if rank > 0 in DDP
        if self.args.rank > 0:
            logger_new.handlers = [h for h in logger_new.handlers if not isinstance(h, logging.StreamHandler)]

    @staticmethod
    def _is_apex_available() -> bool:
        """Check if Nvidia's APEX is available."""
        return importlib.util.find_spec("apex") is not None
