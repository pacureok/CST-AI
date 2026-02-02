from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.shardformer.layer import (FusedLinear1D_Col, FusedLinear1D_Row,
                                          Linear1D_Col, Linear1D_Row)
from colossalai.shardformer.layer._operation import all_to_all_comm
from colossalai.shardformer.layer.attn import RingComm, _rescale_out_lse
from colossalai.shardformer.layer.utils import is_share_sp_tp
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription, Policy, SubModuleReplacementDescription)
from colossalai.shardformer.shard import ShardConfig
from einops import rearrange
from flash_attn.flash_attn_interface import (_flash_attn_backward,
                                             _flash_attn_forward)
from liger_kernel.ops.rope import LigerRopeFunction

try:
    from flash_attn_interface import \
        _flash_attn_backward as _flash_attn_backward_v3
    from flash_attn_interface import \
        _flash_attn_forward as _flash_attn_forward_v3
    SUPPORT_FA3 = True
except ImportError:
    SUPPORT_FA3 = False

from torch import Tensor
from opensora.acceleration.checkpoint import auto_grad_checkpoint
from .layers import DoubleStreamBlock, SingleStreamBlock
from .math import apply_rope, attention
from .model import MMDiTModel

# --- Utilidades de Comunicación ---

class _SplitForwardGatherBackwardVarLen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, process_group, splits: List[int]):
        ctx.process_group = process_group
        ctx.dim = dim
        rank = dist.get_rank(process_group)
        ctx.grad_scale = splits[rank] / sum(splits)
        ctx.splits = splits
        return torch.split(input_, splits, dim=dim)[rank].clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.grad_scale
        grad_output = grad_output.contiguous()
        world_size = dist.get_world_size(ctx.process_group)
        shapes = [list(grad_output.shape) for _ in range(world_size)]
        for i, shape in enumerate(shapes):
            shape[ctx.dim] = ctx.splits[i]
        tensor_list = [torch.empty(shape, dtype=grad_output.dtype, device=grad_output.device) for shape in shapes]
        dist.all_gather(tensor_list, grad_output, group=ctx.process_group)
        return torch.cat(tensor_list, dim=ctx.dim), None, None, None

def split_forward_gather_backward_var_len(input_, dim, process_group, splits: List[int]):
    return _SplitForwardGatherBackwardVarLen.apply(input_, dim, process_group, splits)

class _GatherForwardSplitBackwardVarLen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, process_group, splits: List[int]):
        input_ = input_.contiguous()
        ctx.process_group = process_group
        ctx.dim = dim
        rank = dist.get_rank(process_group)
        ctx.grad_scale = sum(splits) / splits[rank]
        ctx.splits = splits
        world_size = dist.get_world_size(ctx.process_group)
        shapes = [list(input_.shape) for _ in range(world_size)]
        for i, shape in enumerate(shapes):
            shape[dim] = splits[i]
        tensor_list = [torch.empty(shape, dtype=input_.dtype, device=input_.device) for shape in shapes]
        dist.all_gather(tensor_list, input_, group=ctx.process_group)
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.grad_scale
        rank = dist.get_rank(ctx.process_group)
        return torch.split(grad_output, ctx.splits, dim=ctx.dim)[rank].clone(), None, None, None

def gather_forward_split_backward_var_len(input_, dim, process_group, splits: List[int]):
    return _GatherForwardSplitBackwardVarLen.apply(input_, dim, process_group, splits)

# --- Lógica de Flash Attention ---

def _fa_forward(q, k, v, dropout_p=0.0, softmax_scale=None):
    if SUPPORT_FA3:
        out, softmax_lse, *_ = _flash_attn_forward_v3(q, k, v, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, softmax_scale, False, (-1, -1))
        rng_state = None
    else:
        out, softmax_lse, _, rng_state = _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal=False, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, return_softmax=False)
    return out, softmax_lse, rng_state

def _fa_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, rng_state, dropout_p=0.0, softmax_scale=None, deterministic=False):
    if SUPPORT_FA3:
        _flash_attn_backward_v3(dout, q, k, v, out, softmax_lse, None, None, None, None, None, None, dq, dk, dv, softmax_scale, False, (-1, -1), deterministic=deterministic)
    else:
        _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=False, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, deterministic=deterministic, rng_state=rng_state)

# --- Ring Attention Core ---

class RingAttention(torch.autograd.Function):
    ATTN_DONE: torch.cuda.Event = None
    SP_STREAM: torch.cuda.Stream = None

    @staticmethod
    def forward(ctx, q, k, v, sp_group, sp_stream, dropout_p=0.0, softmax_scale=None, deterministic=False):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        sp_size = dist.get_world_size(sp_group)
        kv_comms = [RingComm(sp_group) for _ in range(2)]
        q, k, v = [x.contiguous() for x in [q, k, v]]
        kv_buffers = [torch.stack((k, v)), torch.empty_like(torch.stack((k, v)))]
        out, softmax_lse = None, None
        block_out, block_softmax_lse = [None, None], [None, None]
        rng_states = [None for _ in range(sp_size)]
        sp_streams = [torch.cuda.current_stream(), sp_stream]

        def _kv_comm(i):
            if not RingAttention.ATTN_DONE.query():
                kv_buffers[(i + 1) % 2] = torch.empty_like(kv_buffers[i % 2])
            if i < sp_size - 1:
                kv_comms[i % 2].send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])

        for i in range(sp_size):
            with torch.cuda.stream(sp_streams[i % 2]):
                if i == 0: _kv_comm(i)
                else: kv_comms[(i + 1) % 2].wait()
                
                kv_block = kv_buffers[i % 2]
                block_out[i % 2], block_softmax_lse[i % 2], rng_states[i] = _fa_forward(q, kv_block[0], kv_block[1], dropout_p, softmax_scale)
                RingAttention.ATTN_DONE.record()
                _kv_comm(i + 1)
                
                block_softmax_lse[i % 2] = block_softmax_lse[i % 2].transpose(1, 2).unsqueeze(-1).contiguous().float()
                if i == 0:
                    out, softmax_lse = block_out[0], block_softmax_lse[0]
                else:
                    out, softmax_lse = _rescale_out_lse(out, block_out[i % 2], softmax_lse, block_softmax_lse[i % 2])
        
        torch.cuda.current_stream().wait_stream(sp_stream)
        ctx.save_for_backward(q, k, v, out.to(q.dtype), softmax_lse.squeeze(-1).transpose(1, 2).contiguous(), *rng_states)
        ctx.dropout_p, ctx.softmax_scale, ctx.deterministic, ctx.sp_group = dropout_p, softmax_scale, deterministic, sp_group
        return out.to(q.dtype), softmax_lse.squeeze(-1).transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, grad_output, grad_softmax_lse):
        q, k, v, out, softmax_lse, *rng_states = ctx.saved_tensors
        sp_group = ctx.sp_group
        sp_size = dist.get_world_size(sp_group)
        kv_comm, dkv_comm = RingComm(sp_group), RingComm(sp_group)
        kv_buffers = [torch.stack((k, v)), torch.empty_like(torch.stack((k, v)))]
        dq, dq_block = None, torch.empty_like(q)
        dk_block, dv_block = torch.empty_like(k), torch.empty_like(v)
        dkv_buffers = [torch.empty_like(kv_buffers[0], dtype=torch.float) for _ in range(2)]

        for i in range(sp_size):
            if i > 0: kv_comm.wait()
            if i < sp_size - 1: kv_comm.send_recv(kv_buffers[i % 2], kv_buffers[(i + 1) % 2])
            
            k_block, v_block = kv_buffers[i % 2]
            _fa_backward(grad_output.contiguous(), q, k_block, v_block, out, softmax_lse, dq_block, dk_block, dv_block, rng_states[i], ctx.dropout_p, ctx.softmax_scale, ctx.deterministic)
            
            if i == 0:
                dq = dq_block.float()
                dkv_buffers[i % 2][0], dkv_buffers[i % 2][1] = dk_block.float(), dv_block.float()
            else:
                dq += dq_block
                dkv_comm.wait()
                dkv_buffers[i % 2][0] += dk_block
                dkv_buffers[i % 2][1] += dv_block
            dkv_comm.send_recv(dkv_buffers[i % 2], dkv_buffers[(i + 1) % 2])
        
        dkv_comm.wait()
        dkv = dkv_buffers[sp_size % 2]
        return dq.to(q.dtype), dkv[0].to(q.dtype), dkv[1].to(q.dtype), None, None, None, None, None

    @staticmethod
    def attention(q, k, v, sp_group, dropout_p=0.0, softmax_scale=None, deterministic=False, return_softmax=False):
        if RingAttention.ATTN_DONE is None: RingAttention.ATTN_DONE = torch.cuda.Event()
        if RingAttention.SP_STREAM is None: RingAttention.SP_STREAM = torch.cuda.Stream()
        out, lse = RingAttention.apply(q, k, v, sp_group, RingAttention.SP_STREAM, dropout_p, softmax_scale, deterministic)
        return (out, lse) if return_softmax else out

def ring_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, sp_group: dist.ProcessGroup) -> Tensor:
    if isinstance(pe, torch.Tensor): q, k = apply_rope(q, k, pe)
    else: q, k = LigerRopeFunction.apply(q, k, *pe)
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    x = RingAttention.attention(q, k, v, sp_group)
    return rearrange(x, "B L H D -> B L (H D)")

# --- Procesadores de Bloques ---

class DistributedDoubleStreamBlockProcessor:
    def __init__(self, shard_config: ShardConfig) -> None:
        self.shard_config = shard_config

    def __call__(self, attn: DoubleStreamBlock, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # Imagen
        img_m = (1 + img_mod1.scale) * attn.img_norm1(img) + img_mod1.shift
        if attn.img_attn.fused_qkv:
            img_q, img_k, img_v = rearrange(attn.img_attn.qkv(img_m), "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_m), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_m), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_m), "B L (H D) -> B L H D", H=attn.num_heads)
        
        # Texto
        txt_m = (1 + txt_mod1.scale) * attn.txt_norm1(txt) + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_q, txt_k, txt_v = rearrange(attn.txt_attn.qkv(txt_m), "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_m), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_m), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_m), "B L (H D) -> B L H D", H=attn.num_heads)

        # Concatenación y SP
        q, k, v = [torch.cat((txt_q, img_q), dim=2) for txt_q, img_q in [(txt_q, img_q), (txt_k, img_k), (txt_v, img_v)]]
        
        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "all_to_all":
            q = all_to_all_comm(q, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            k = all_to_all_comm(k, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            v = all_to_all_comm(v, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)

        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "ring_attn":
            attn_res = ring_attention(q, k, v, pe, self.shard_config.sequence_parallel_process_group)
        else:
            attn_res = attention(q, k, v, pe=pe)

        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "all_to_all":
            attn_res = all_to_all_comm(attn_res, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)

        t_len = txt.shape[1]
        txt_attn, img_attn = attn_res[:, :t_len], attn_res[:, t_len:]
        
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DistributedSingleStreamBlockProcessor:
    def __init__(self, shard_config: ShardConfig) -> None:
        self.shard_config = shard_config

    def __call__(self, attn: SingleStreamBlock, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift

        if attn.fused_qkv:
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        else:
            q = rearrange(attn.q_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            k = rearrange(attn.k_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            v, mlp = torch.split(attn.v_mlp(x_mod), [attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=attn.num_heads)
        
        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "all_to_all":
            q = all_to_all_comm(q, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            k = all_to_all_comm(k, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)
            v = all_to_all_comm(v, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)

        attn_res = ring_attention(q, k, v, pe, self.shard_config.sequence_parallel_process_group) if self.shard_config.sequence_parallelism_mode == "ring_attn" else attention(q, k, v, pe=pe)

        if self.shard_config.enable_sequence_parallelism and self.shard_config.sequence_parallelism_mode == "all_to_all":
            attn_res = all_to_all_comm(attn_res, self.shard_config.sequence_parallel_process_group, scatter_dim=1, gather_dim=2)

        return x + mod.gate * attn.linear2(torch.cat((attn_res, attn.mlp_act(mlp)), 2))

# --- Forward del Modelo y Políticas ---

def mmdit_model_forward(self: MMDiTModel, img, img_ids, txt, txt_ids, timesteps, y_vec, cond=None, guidance=None, shard_config=None, stage_index=None, **kwargs):
    if shard_config.pipeline_stage_manager is None or shard_config.pipeline_stage_manager.is_first_stage():
        img, txt, vec, pe = self.prepare_block_inputs(img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance)
        has_grad = img.requires_grad
        
        if shard_config.enable_sequence_parallelism:
            mask = torch.zeros(txt.shape[1] + img.shape[1], dtype=bool, device=img.device)
            mask[txt.shape[1]:] = 1
            mask_chunks = mask.chunk(shard_config.sequence_parallel_size)
            img_splits = [c.sum().item() for c in mask_chunks]
            txt_splits = [len(c) - c.sum().item() for c in mask_chunks]
            
            img = split_forward_gather_backward_var_len(img, 1, shard_config.sequence_parallel_process_group, img_splits)
            txt = split_forward_gather_backward_var_len(txt, 1, shard_config.sequence_parallel_process_group, txt_splits)
    
    # Ejecución de Bloques (Double y Single)
    for block in self.double_blocks:
        img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)
    
    img = torch.cat((txt, img), 1)
    for block in self.single_blocks:
        img = auto_grad_checkpoint(block, img, vec, pe)
    
    img = self.final_layer(img[:, txt.shape[1]:], vec)
    
    if shard_config.enable_sequence_parallelism:
        img = gather_forward_split_backward_var_len(img, 1, shard_config.sequence_parallel_process_group, img_splits)
    
    return img

class MMDiTPolicy(Policy):
    def module_policy(self):
        policy = {
            DoubleStreamBlock: ModulePolicyDescription(attribute_replacement={}, sub_module_replacement=[]),
            SingleStreamBlock: ModulePolicyDescription(attribute_replacement={}, sub_module_replacement=[]),
        }
        if self.shard_config.enable_sequence_parallelism:
            policy[DoubleStreamBlock].attribute_replacement["processor"] = DistributedDoubleStreamBlockProcessor(self.shard_config)
            policy[SingleStreamBlock].attribute_replacement["processor"] = DistributedSingleStreamBlockProcessor(self.shard_config)
        
        fwd_fn = partial(mmdit_model_forward, shard_config=self.shard_config)
        self.append_or_create_method_replacement(description={"forward": fwd_fn}, policy=policy, target_key=MMDiTModel)
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        # Lógica para Pipeline Parallelism
        stage_manager = self.shard_config.pipeline_stage_manager
        held_layers = []
        if stage_manager.is_first_stage():
            held_layers.extend([self.model.img_in, self.model.txt_in])
        # ... añadir bloques según stage_manager.get_stage_index ...
        if stage_manager.is_last_stage():
            held_layers.append(self.model.final_layer)
        return held_layers