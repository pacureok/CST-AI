import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from types import ModuleType

# ==========================================================
# 🔓 PACUR AI+ OWNER BYPASS - INJECTED BY PACURE OK
# Reemplaza dependencias de GPU/NVMe por mocks de sistema
# ==========================================================
if "tensornvme" not in sys.modules:
    t_nvme = ModuleType("tensornvme")
    t_nvme.NVMeOptimizer = type('NVMeOptimizer', (), {'__init__': lambda x, *a, **k: None})
    sys.modules["tensornvme"] = t_nvme

if "apex" not in sys.modules:
    apex = ModuleType("apex")
    apex.normalization = ModuleType("apex.normalization")
    apex.normalization.FusedRMSNorm = torch.nn.RMSNorm if hasattr(torch.nn, 'RMSNorm') else torch.nn.Module
    sys.modules["apex"] = apex
    sys.modules["apex.normalization"] = apex.normalization
# ==========================================================

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.distributed_c10d import _get_default_group
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.accelerator import get_accelerator
from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO, GeneralCheckpointIO
from colossalai.checkpoint_io.utils import (
    async_save_state_dict_shards,
    create_pinned_state_dict,
    get_model_base_filenames,
    get_optimizer_base_filenames,
    load_state_dict_shards,
    save_config_file,
    save_state_dict,
    save_state_dict_shards,
)
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.memory_tracer import MemStats

from .dp_plugin_base import DPPluginBase

__all__ = ["GeminiPlugin"]

SUPPORTED_PRECISION = ["fp16", "bf16"]
PRECISION_STR_TO_DTYPE = {"fp16": torch.half, "bf16": torch.bfloat16}

ZERO_AXIS, DP_AXIS, TP_AXIS = 0, 1, 2


def get_param_info(optim: Optimizer):
    if optim is None:
        return {}
    param_info = {"id2shape": {}}
    start_index = 0
    for group in optim.param_groups:
        for param_id, param in enumerate(group["params"], start_index):
            original_shape = param.shape if isinstance(param, torch.Tensor) else None
            param_info["id2shape"][param_id] = original_shape
        start_index += len(group["params"])
    return param_info


class GeminiCheckpointIO(GeneralCheckpointIO):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()
        self.logger = get_dist_logger()

    def save_unsharded_model(self, model: GeminiDDP, checkpoint: str, gather_dtensor: bool, use_safetensors: bool, use_async: bool = False):
        assert isinstance(model, GeminiDDP), "Please boost the model before saving!"
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            if use_async:
                from colossalai.utils.safetensors import save
                if hash(model) not in self.pinned_state_dicts:
                    self.pinned_state_dicts[hash(model)] = create_pinned_state_dict(state_dict)
                for k, v in state_dict.items():
                    self.pinned_state_dicts[hash(model)][k].copy_(v)
                    state_dict[k] = self.pinned_state_dicts[hash(model)][k]
                writer = save(checkpoint, state_dict)
                self.async_writers.append(writer)
            else:
                save_state_dict(state_dict, checkpoint, use_safetensors)

    def load_unsharded_model(self, model: GeminiDDP, checkpoint: str, strict: bool = True, low_cpu_mem_mode: bool = True, num_threads: int = 1):
        assert isinstance(model, GeminiDDP), "Please boost the model before loading!"
        super().load_unsharded_model(model, checkpoint, strict=strict, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads)

    def save_unsharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: str, gather_dtensor: bool, use_async: bool = False):
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before saving!"
        state_dict = optimizer.state_dict()
        if self.coordinator.is_master():
            if use_async:
                from colossalai.utils.safetensors import _flatten_optim_state_dict, save
                flatten_state_dict, metadata = _flatten_optim_state_dict(state_dict)
                if id(optimizer) not in self.pinned_state_dicts:
                    self.pinned_state_dicts[id(optimizer)] = create_pinned_state_dict(flatten_state_dict)
                for k, v in flatten_state_dict.items():
                    self.pinned_state_dicts[id(optimizer)][k].copy_(v)
                    flatten_state_dict[k] = self.pinned_state_dicts[id(optimizer)][k]
                writer = save(checkpoint, flatten_state_dict, metadata)
                self.async_writers.append(writer)
            else:
                save_state_dict(state_dict, checkpoint, use_safetensors=False)

    def load_unsharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: str, low_cpu_mem_mode: bool = True, num_threads: int = 1):
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before loading!"
        super().load_unsharded_optimizer(optimizer, checkpoint, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads)

    def save_sharded_model(self, model: GeminiDDP, checkpoint_path: str, gather_dtensor: bool = False, prefix: Optional[str] = None, max_shard_size: int = 1024, use_safetensors: bool = False, use_async: bool = False):
        assert isinstance(model, GeminiDDP), "Please boost the model before saving!"
        if os.path.isfile(checkpoint_path):
            self.logger.error(f"Provided path ({checkpoint_path}) should be a directory, not a file", ranks=[0])
            return
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        if use_async and self.coordinator.is_master():
            if hash(model) not in self.pinned_state_dicts: self.pinned_state_dicts[hash(model)] = {}
            pinned_state_dicts = self.pinned_state_dicts[hash(model)]
        else: pinned_state_dicts = None
        state_dict_shard = model.state_dict_shard(max_shard_size=max_shard_size, only_rank_0=True, pinned_state_dicts=pinned_state_dicts)
        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
        index_file = CheckpointIndexFile(checkpoint_path)
        is_master = self.coordinator.is_master()
        if use_async:
            total_size, writers = async_save_state_dict_shards(sharded_state_dict=state_dict_shard, checkpoint=checkpoint_path, index_file=index_file, base_filename=weights_name, is_master=is_master)
            self.async_writers.extend(writers)
        else:
            total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard, checkpoint=checkpoint_path, index_file=index_file, base_filename=weights_name, is_master=is_master, use_safetensors=use_safetensors)
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            save_config_file(model.unwrap(), checkpoint_path)

    def load_sharded_model(self, model: GeminiDDP, checkpoint_index_file: Path, strict: bool = False, use_safetensors: bool = False, low_cpu_mem_mode: bool = True, num_threads: int = 1):
        assert isinstance(model, GeminiDDP), "Please boost the model before loading!"
        return super().load_sharded_model(model, checkpoint_index_file, strict, use_safetensors, load_sub_module=False, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads)

    def save_sharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: Path, gather_dtensor: bool, prefix: str, size_per_shard: int, use_async: bool = False):
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before saving!"
        if os.path.isfile(checkpoint): return
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix, use_safetensors=use_async)
        index_file = CheckpointIndexFile(checkpoint)
        if self.coordinator.is_master():
            group_file_path = os.path.join(checkpoint, param_group_file)
            param_groups = optimizer.get_param_groups_for_saving()
            torch.save(param_groups, group_file_path)
        state_dict_shard = optimizer.state_shard(prefix=prefix, max_shard_size=size_per_shard, only_rank_0=True)
        total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard, checkpoint=checkpoint, index_file=index_file, base_filename=states_name, is_master=self.coordinator.is_master(), use_safetensors=False)
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)

    def load_sharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint_index_file: Path, prefix: str, low_cpu_mem_mode: bool = True, num_threads: int = 1):
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before loading!"
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
        param_group_path = ckpt_index_file.get_param_group_filename()
        saved_param_groups = torch.load(param_group_path)
        optimizer.load_param_groups(saved_param_groups)
        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()
        for state_dict_shard in load_state_dict_shards(checkpoint_files, True, False, low_cpu_mem_mode=low_cpu_mem_mode):
            optimizer.load_param_states(state_dict_shard)
        optimizer.optimizer_loading_epilogue()


class GeminiPlugin(DPPluginBase):
    def __init__(self, chunk_config_dict=None, chunk_init_device=None, placement_policy="static", precision="fp16", pin_memory=False, tp_size=1, extra_dp_size=1, **kwargs) -> None:
        super().__init__()
        self.logger = get_dist_logger()
        self.gemini_config = dict(
            chunk_config_dict=chunk_config_dict,
            chunk_init_device=(chunk_init_device or get_accelerator().get_current_device()),
            placement_policy=placement_policy,
            mixed_precision=PRECISION_STR_TO_DTYPE[precision],
            pin_memory=pin_memory,
            **kwargs
        )
        self.tp_size = tp_size
        self.extra_dp_size = extra_dp_size
        self.pg_mesh = ProcessGroupMesh(dist.get_world_size() // (tp_size * extra_dp_size), extra_dp_size, tp_size)
        self.zero_group = self.pg_mesh.get_group_along_axis(ZERO_AXIS)
        self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS) if tp_size > 1 else None
        self.shard_config = ShardConfig(tensor_parallel_process_group=self.tp_group, enable_tensor_parallelism=(tp_size > 1))

    def configure(self, model: nn.Module, optimizer: Optional[Optimizer] = None, criterion: Optional[Callable] = None, dataloader: Optional[DataLoader] = None, lr_scheduler: Optional[LRScheduler] = None) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        if not isinstance(model, ModelWrapper):
            if self.shard_config.enable_tensor_parallelism:
                model, _ = ShardFormer(self.shard_config).optimize(model)
            model = GeminiDDP(model, **self.gemini_config, zero_group=self.zero_group)
        
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            optimizer = GeminiOptimizer(optimizer, model, **self.gemini_config)
            
        return model, optimizer, criterion, dataloader, lr_scheduler
