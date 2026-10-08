import sys
import os
import ipdb
import warnings
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    fix_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

from dust3r.smpl_model import nms, apply_threshold
from einops import rearrange

from dust3r.utils.geometry import inverse_perspective_projection, get_camera_parameters
from dust3r.utils.image import unpad_uv, log_optimal_transport
from mhmr.blocks import Dinov2Backbone, FourierPositionEncoding, TransformerDecoder
from wilor.models.backbones.vit import ViT
from dust3r.fuse_mlp import HandPromptBuilder
from dust3r.hand_constants import (
    HAND_TOKENS_PER_HAND,
    NUM_HANDS,
    NUM_HAND_TOKENS,
)
from dust3r.heads.contact_head import MANOContactHead
from dust3r.roi_align import ROIAlign
printer = get_logger(__name__, log_level="DEBUG")

from dust3r.utils.device import to_cpu, to_gpu

@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

# todo : 改为wilor的load
def strip_module_mhmr(state_dict):
    """
    Load Multi-HMR pretrained model
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(("mlp_classif.", "mlp_offset.")):
            name = f"downstream_head.{k}"
        elif k.startswith(("x_attention_head.dec")):
            name = f"downstream_head.{k[17:]}"
        elif k.startswith(("x_attention_head.transformer.", "x_attention_head.cross_")):
            name = k[17:]
        elif k.startswith(("backbone")):
            name = k
        else:
            continue
        new_state_dict[name] = v
    return new_state_dict

def strip_module_wilor(state_dict, target_state_dict=None):
    """
    Load wilor pretrained model
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(
            ("backbone.decpose.", "backbone.decshape.", "backbone.deccam.")
        ):
            # MANO output heads are trained from scratch in Hand3R.
            continue
        if k.startswith(("backbone.")):
            name = f"hand_backbone.{k[9:]}"
        else:
            continue

        if target_state_dict is not None:
            target = target_state_dict.get(name)
            if target is None or v.shape != target.shape:
                continue
        new_state_dict[name] = v
    return new_state_dict



def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        contact_head=False,
        contact_feedback=False,
        contact_hidden_dim=256,
        contact_roi_size=5,
        detach_scene_for_hand=True,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.contact_head = contact_head
        self.contact_feedback = contact_feedback
        self.contact_hidden_dim = contact_hidden_dim
        self.contact_roi_size = contact_roi_size
        self.detach_scene_for_hand = bool(detach_scene_for_hand)
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        # CroCo's constructor replaces self.config with croco_cfg. Preserve this
        # HAND3R-specific switch on the model as well as in its input config.
        self.detach_scene_for_hand = bool(config.detach_scene_for_hand)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )


        #此处已添加新的token
        # self.masked_smpl_token = nn.Parameter(
        #     torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        # ) #此处learnable token

        self.masked_query_token = nn.Parameter(
            torch.randn(1, 768) * 0.02, requires_grad=True
        )
        self.mhmr_masked_img_token = nn.Parameter(
            torch.randn(1, 1280) * 0.02, requires_grad=True
        )#这个貌似没什么用

        self.hand_backbone = ViT(
                img_size=(256, 192),
                patch_size=16,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                ratio=1,
                use_checkpoint=False, #false
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.55,
            )
        self.roi = ROIAlign(output_size=(5,5),spatial_scale=1/16, sampling_ratio=2)
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        self.prompt_builder = HandPromptBuilder()
        """
        输入:
            f_scene: [B, C, H, W] (e.g., [B, 1024, 7, 7])
            f_prior: [B, 18, 1280]
            is_left_idx: [B] (0=Right, 1=Left)
        输出:
            prompt_query: [B, 18, embed_dim]
        """

        self.contact_head_enabled = bool(config.contact_head)
        if self.contact_head_enabled:
            self.mano_contact_head = MANOContactHead(
                decoder_dim=self.dec_embed_dim,
                prior_dim=1280,
                hidden_dim=config.contact_hidden_dim,
                roi_size=config.contact_roi_size,
                patch_size=self.croco_args["patch_size"],
                enable_feedback=config.contact_feedback,
            )

        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value

        if not kw.get("strict", True):
            model_state = self.state_dict()
            skipped = []
            compatible_ckpt = {}
            for key, value in new_ckpt.items():
                target = model_state.get(key)
                if target is not None and value.shape != target.shape:
                    skipped.append((key, tuple(value.shape), tuple(target.shape)))
                    continue
                compatible_ckpt[key] = value
            if skipped:
                preview = ", ".join(
                    f"{key}: {source_shape}->{target_shape}"
                    for key, source_shape, target_shape in skipped[:8]
                )
                if len(skipped) > 8:
                    preview += f", ... ({len(skipped)} total)"
                warnings.warn(
                    f"Skipped incompatible checkpoint parameters: {preview}",
                    RuntimeWarning,
                )
            new_ckpt = compatible_ckpt

        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
            "encoder_and_decoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
                self.downstream_head.dpt_self,
                self.downstream_head.final_transform,
                self.downstream_head.dpt_cross,
                self.downstream_head.dpt_rgb,
                self.downstream_head.pose_head,
            ],
            "hand": [
                self.hand_backbone,
                self.mhmr_masked_img_token,
            ],
        }

        if freeze == "encoder_and_decoder_and_head":
            # Frozen weights still propagate gradients to the input hand prompts.
            freeze_all_params(to_be_frozen["encoder_and_decoder_and_head"])
            freeze_all_params(to_be_frozen["hand"])
        else:
            freeze_all_params(to_be_frozen[freeze])

    def configure_wilor_finetuning(self, last_n_blocks=-1):
        """Enable full or partial fine-tuning of the WiLoR feature backbone."""
        if getattr(self, "_contact_head_only", False):
            raise ValueError("Contact-head-only training cannot fine-tune WiLoR")
        last_n_blocks = int(last_n_blocks)
        num_blocks = len(self.hand_backbone.blocks)
        if last_n_blocks == 0 or last_n_blocks < -1:
            raise ValueError(
                "wilor_unfreeze_last_n_blocks must be -1 (all blocks) or a "
                "positive integer"
            )
        if last_n_blocks > num_blocks:
            raise ValueError(
                "wilor_unfreeze_last_n_blocks cannot exceed the WiLoR depth "
                f"({last_n_blocks} vs {num_blocks})"
            )

        # set_freeze() may already have frozen this module. Reset it first so
        # repeated calls have deterministic behavior.
        for param in self.hand_backbone.parameters():
            param.requires_grad = False

        if last_n_blocks == -1:
            for param in self.hand_backbone.parameters():
                param.requires_grad = True
            mode = "all"
        else:
            for block in self.hand_backbone.blocks[-last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.hand_backbone.last_norm.parameters():
                param.requires_grad = True
            mode = f"last_{last_n_blocks}_blocks"

        self._wilor_finetune_last_n_blocks = last_n_blocks
        trainable_params = sum(
            param.numel()
            for param in self.hand_backbone.parameters()
            if param.requires_grad
        )
        return {
            "mode": mode,
            "trainable_params": trainable_params,
            "total_params": sum(
                param.numel() for param in self.hand_backbone.parameters()
            ),
        }

    def configure_contact_head_only(self):
        """Freeze the pretrained feature pipeline while fitting a new contact head."""
        if not self.contact_head_enabled:
            raise ValueError("Contact-head-only training requires contact_head=True")
        self.requires_grad_(False)
        self.mano_contact_head.requires_grad_(True)
        self._contact_head_only = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "_contact_head_only", False):
            for name, module in self.named_children():
                module.train(mode and name == "mano_contact_head")
            return self
        last_n_blocks = getattr(self, "_wilor_finetune_last_n_blocks", None)
        if mode and not any(p.requires_grad for p in self.hand_backbone.parameters()):
            self.hand_backbone.eval()
        elif mode and last_n_blocks not in (None, -1):
            # Frozen early blocks should not apply stochastic depth. Only the
            # trainable tail remains in training mode.
            self.hand_backbone.eval()
            for block in self.hand_backbone.blocks[-last_n_blocks:]:
                block.train(True)
            self.hand_backbone.last_norm.train(True)
        return self

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views_mhmr(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos

        # MHMR
        # MHMR for hand crops: resized_crop shape [num_views, B, 2, 3, H, W]
        imgs_hand = torch.stack([view["resized_crop"] for view in views], dim=0)  # (V, B, 2, 3, H, W)
        num_views = len(views)
        # merge view, batch, hand -> (V * B * 2, 3, H, W)
        imgs_hand = imgs_hand.view(-1, *imgs_hand.shape[3:])
        # mano_mask: (V, B, 2)
        mano_masks = torch.stack([
            view.get(
                "mano_mask",
                torch.zeros(batch_size, NUM_HANDS, dtype=torch.bool, device=device),
            )
            for view in views
        ], dim=0)
        mano_masks_flat = mano_masks.view(-1)
        # select present hand crops and encode with hand_backbone
        selected_imgs_hand = imgs_hand[mano_masks_flat]
        if selected_imgs_hand.size(0) > 0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
            selected_imgs_hand = (selected_imgs_hand * 0.5 + 0.5 - mean) / std
            hand_out = self.hand_backbone(selected_imgs_hand)  # [N_sel, n_patches, D]
        else:
            hand_out = None
        # build full per-hand container (V * B * 2, n_patches, D)
        hand_full_out = torch.zeros(
            len(views) * batch_size * NUM_HANDS,
            HAND_TOKENS_PER_HAND,
            1280,
            device=device,
            dtype=torch.float32,
        )

        if hand_out is not None:
            hand_full_out[mano_masks_flat] = hand_out
        hand_full_out[~mano_masks_flat] += self.mhmr_masked_img_token

        # reshape to (V, B, 2, n_patches, D) and merge two hands per view by masked mean
        hand_full_out = hand_full_out.view(
            num_views,
            batch_size,
            NUM_HANDS,
            hand_full_out.shape[1],
            hand_full_out.shape[2],
        )
        #到这里结束 #以上有很多不必要操作，因为img_mask一定是true
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
            hand_full_out,
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_smpl, pos_smpl, use_ttt3r=False):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            if f_smpl is not None:
                f_img = torch.cat([f_pose, f_img, f_smpl], dim=1)
                pos_img = torch.cat([pos_pose, pos_img, pos_smpl], dim=1)
            else:
                f_img = torch.cat([f_pose, f_img], dim=1) # used for naive CUT3R+MHMR
                pos_img = torch.cat([pos_pose, pos_img], dim=1) # used for naive CUT3R+MHMR
        final_output.append((f_state, f_img))
        cross_attn_states = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_ttt3r=use_ttt3r,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_ttt3r=False,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, use_ttt3r=use_ttt3r)
                f_img, _, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state, use_ttt3r=False)

            final_output.append((f_state, f_img))
            cross_attn_states.append(cross_attn_state)

        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )

        return zip(*final_output), cross_attn_states

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        smpl_feat,
        smpl_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        use_ttt3r=False,
    ):
        (new_state_feat, dec), cross_attn_states = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, smpl_feat, smpl_pos, use_ttt3r
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, cross_attn_states

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _predict_contact(
        self,
        decoder_output,
        hand_prior,
        bbox,
        image_shape,
        hand_mask,
    ):
        hand_tokens = decoder_output[:, -NUM_HAND_TOKENS:].float()
        if not self.contact_head_enabled:
            return hand_tokens, None

        scene_start = 1 if self.pose_head_flag else 0
        scene_tokens = decoder_output[
            :, scene_start:-NUM_HAND_TOKENS
        ].float()
        contact_logits, hand_delta = self.mano_contact_head(
            hand_tokens=hand_tokens,
            prior_tokens=hand_prior,
            scene_tokens=scene_tokens,
            bbox=bbox,
            image_size=image_shape,
            hand_mask=hand_mask,
        )
        if hand_delta is not None:
            hand_tokens = hand_tokens + hand_delta
        return hand_tokens, contact_logits


    def process_hand_features(self, feat_list, hand_feat, views):
        """
        极致优化版处理函数：利用 Mask 预先过滤空手，避免无效的计算开销。
        """
        V = len(feat_list)
        bs = feat_list[0].shape[0]
        patch_size = self.croco_args['patch_size']
        device = feat_list[0].device
        dtype = feat_list[0].dtype

        # ==========================================
        # 1. 堆叠并合并 V 和 bs 维度
        # ==========================================
        detach_scene = getattr(self, "detach_scene_for_hand", True)
        feat = torch.stack([f.detach() if detach_scene else f for f in feat_list], dim=0)
        num_view, batch_size = feat.shape[:2]
        feat = feat.view(-1, *feat.shape[2:]) #(num_view * bs, 576, 1024)
        # imgs_hand = torch.stack([view["resized_crop"] for view in views], dim=0)  # (V, B, 2, 3, H, W)
        bbox = torch.stack([view["bbox"] for view in views], dim=0)
        mask = torch.stack([view["mano_mask"] for view in views], dim=0)

        n_patches, D = hand_feat.shape[-2], hand_feat.shape[-1]
        hfeat = hand_feat.view(V * bs, NUM_HANDS, n_patches, D)

        # 将图片特征恢复为 2D 空间特征供 ROIAlign 使用
        nH,nW = views[0]["true_shape"][0] // patch_size
        feat_spatial = feat.transpose(1, 2).reshape(
            V * bs, self.enc_embed_dim, nH.item(), nW.item()
        )
        # ==========================================
        # 2. 预先构建包含 masked_query_token 的输出张量
        # ==========================================
        # 获取占位符并保证数据类型正确。假设其 token 维度大小为 embed_dim (如 768)
        masked_token = self.masked_query_token.to(device=device, dtype=dtype)
        embed_dim = masked_token.shape[-1]
        
        # 预先铺满所有的占位 token (V * bs * 2 只手)
        # 利用 expand 和 clone 避免循环赋值，这是最高效的初始填充方式
        full_prompts = masked_token.view(1, 1, embed_dim).expand(
            V * bs * NUM_HANDS, HAND_TOKENS_PER_HAND, embed_dim
        ).clone()

        # ==========================================
        # 3. 获取有效的 Mask 索引 (Filter)
        # ==========================================
        flat_mask = mask.view(-1) # [V * bs * 2]
        valid_idx = flat_mask.nonzero(as_tuple=True)[0] # 获取值为 True 的位置索引
        # 如果当前 batch 中存在有效的手，才执行重计算
        if valid_idx.numel() > 0:
            # 对应到 feat_spatial 的 batch index 是 valid_idx // 2
            # ROIAlign 要求的 rois 第一列是 float 类型的 batch 索引
            valid_batch_idx = (valid_idx // NUM_HANDS).unsqueeze(1).to(dtype)
            
            flat_bbox = bbox.view(V * bs * NUM_HANDS, 4)
            valid_bbox = flat_bbox[valid_idx]
            
            # [N_valid, 5]
            valid_rois = torch.cat([valid_batch_idx, valid_bbox], dim=1)
            
            # [N_valid, n_patches, D]
            flat_hfeat = hfeat.view(V * bs * NUM_HANDS, n_patches, D)
            valid_f_prior = flat_hfeat[valid_idx]
            
            # 巧妙利用奇偶性判断左右手 (0=Right, 1=Left)
            # 展平后的顺序恒为右、左、右、左...，因此直接模 2 即可
            valid_is_left = (valid_idx % NUM_HANDS).long()

            # ==========================================
            # 4. 仅对有效 Hand 执行重计算 (Heavy Compute)
            # ==========================================
            valid_roi_out = self.roi(feat_spatial, valid_rois)
            valid_prompt = self.prompt_builder(f_scene=valid_roi_out, 
                                            f_prior=valid_f_prior, 
                                            is_left_idx=valid_is_left)
            
            # ==========================================
            # 5. Scatter 填入预定义好的结果张量中
            # ==========================================
            full_prompts[valid_idx] = valid_prompt

        # ==========================================
        # 6. 计算 Patch UV 坐标 (hand_pos)
        # 简单算术计算非常轻量级，无需掩码直接全量计算即可
        # ==========================================
        cx = (bbox[..., 0] + bbox[..., 2]) / 2.0  # [V, bs, 2]
        cy = (bbox[..., 1] + bbox[..., 3]) / 2.0  # [V, bs, 2]
        
        
        hand_pos_flat = torch.stack([cy // patch_size, cx // patch_size], dim=-1).long() # [V, bs, 2, 2]
        

        hand_pos_flat = hand_pos_flat.unsqueeze(3).expand(
            -1, -1, -1, HAND_TOKENS_PER_HAND, -1
        )
        hand_poses = list(
            hand_pos_flat.reshape(V, bs, NUM_HAND_TOKENS, 2).unbind(dim=0)
        )
        # 直接 view 成 [V, bs, 36, embed_dim]
        prompt_queries = list(
            full_prompts.reshape(V, bs, NUM_HAND_TOKENS, embed_dim).unbind(dim=0)
        )
        
        return prompt_queries, hand_poses

    def _forward_impl(self, views, ret_state=False, inference=False):
        #ipdb.set_trace()
        shape, feat_ls, pos, hand_feat = self._encode_views_mhmr(views) 
        # shape v*[bs, 2], pos v*[bs, 576, 2], feat v*[bs, 576, 1024], hand_feat (V, B, 2, n_patches, D)
        feat = feat_ls[-1]
        hand_query, hand_pos = self.process_hand_features(feat, hand_feat, views)
        #v[bs, 36, embed_dim], v[bs, 36, 2]
        pos_central = hand_pos
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)  # [b, 256, 1536]
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            smpl_feat_i = hand_query[i]
            smpl_pos_i = pos_central[i]
            n_humans_i = smpl_feat_i.shape[1]

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)    # [b, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)   # coarse pose feat: [b, 1, 768]
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)   # coarse pose feat: [b, 1, 768]
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            new_state_feat, dec, _ = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                smpl_feat_i,
                smpl_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]   # After Cross-Attention, refined pose feat: [b, 1, 768]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )   # [b, 256, 1536]

            assert len(dec) == self.dec_depth + 1
            if n_humans_i > 0:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth][:, :-n_humans_i].float(),
                ]
                hand_origin = hand_feat[i].view(
                    hand_query[i].shape[0], NUM_HAND_TOKENS, 1280
                )
                smpl_token, contact_logits = self._predict_contact(
                    dec[self.dec_depth],
                    hand_origin,
                    views[i]["bbox"],
                    shape[i],
                    views[i]["mano_mask"],
                )
                smpl_token = torch.cat([smpl_token, hand_origin], dim=-1)
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
                smpl_token = None
                contact_logits = None
            res = self._downstream_head(
                head_input, shape[i], pos=pos_i, n_humans=n_humans_i, smpl_token=smpl_token, mano_mask=views[i]['mano_mask'])
            if contact_logits is not None:
                res["contact_logits"] = contact_logits
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views


    def forward(self, views, ret_state=False, inference=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state, inference=inference)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)


    def forward_recurrent_lighter(self, views, device, ret_state=False, use_ttt3r=False):
        ress = []
        all_state_args = []
        reset_mask = False
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None

            shape = shapes
            feat_i = img_out[-1]
            pos_i = img_pos
            # hand_enocde
            imgs_hand = view["resized_crop"]  # [1,2,3,256,192]
            mano_mask_flat = view["mano_mask"].view(-1)
            imgs_hand = imgs_hand.view(
                -1, *imgs_hand.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            selected_imgs_hand = imgs_hand[mano_mask_flat]
            hand_full_out = torch.zeros(
                batch_size * NUM_HANDS,
                HAND_TOKENS_PER_HAND,
                1280,
                device=device,
                dtype=torch.float32,
            )
            if selected_imgs_hand.size(0) > 0:
                mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
                selected_imgs_hand = (selected_imgs_hand * 0.5 + 0.5 - mean) / std
                hand_out = self.hand_backbone(selected_imgs_hand) # image[bs, 3, h, w] -> image feature [bs, 18, D]
                hand_full_out[mano_mask_flat] = hand_out
            hand_full_out[~mano_mask_flat] += self.mhmr_masked_img_token

            # hand tokenizer
            patch_size = self.croco_args['patch_size']
            nH, nW = map(int, shape[0].tolist())
            nH //= patch_size
            nW //= patch_size
            feat_spatial = feat_i.transpose(1, 2).reshape(
                batch_size, self.enc_embed_dim, nH, nW
            )
            masked_token = self.masked_query_token
            embed_dim = masked_token.shape[-1]
            full_prompts = masked_token.view(1, 1, embed_dim).expand(
                batch_size * NUM_HANDS, HAND_TOKENS_PER_HAND, embed_dim
            ).clone()

            n_patches, D = hand_full_out.shape[-2], hand_full_out.shape[-1]
            valid_idx = mano_mask_flat.nonzero(as_tuple=True)[0] # 获取值为 True 的位置索引
            bbox = view['bbox']
            # 如果当前 batch 中存在有效的手，才执行重计算
            if valid_idx.numel() > 0:
                # 对应到 feat_spatial 的 batch index 是 valid_idx // 2
                # ROIAlign 要求的 rois 第一列是 float 类型的 batch 索引
                valid_batch_idx = (valid_idx // NUM_HANDS).unsqueeze(1)
                
                flat_bbox = bbox.view(batch_size * NUM_HANDS, 4)
                valid_bbox = flat_bbox[valid_idx]
                valid_batch_idx = valid_batch_idx.to(valid_bbox.dtype)
                
                # [N_valid, 5]
                valid_rois = torch.cat([valid_batch_idx, valid_bbox], dim=1)
                
                # [N_valid, n_patches, D]
                flat_hfeat = hand_full_out.view(
                    batch_size * NUM_HANDS, n_patches, D
                )
                valid_f_prior = flat_hfeat[valid_idx]
                
                # 巧妙利用奇偶性判断左右手 (0=Right, 1=Left)
                # 展平后的顺序恒为右、左、右、左...，因此直接模 2 即可
                valid_is_left = (valid_idx % NUM_HANDS).long()

                # ==========================================
                # 4. 仅对有效 Hand 执行重计算 (Heavy Compute)
                # ==========================================
                valid_roi_out = self.roi(feat_spatial, valid_rois)
                valid_prompt = self.prompt_builder(f_scene=valid_roi_out, 
                                                f_prior=valid_f_prior, 
                                                is_left_idx=valid_is_left)
                
                # ==========================================
                # 5. Scatter 填入预定义好的结果张量中
                # ==========================================
                full_prompts[valid_idx] = valid_prompt

            # ==========================================
            # 6. 计算 Patch UV 坐标 (hand_pos)
            # 简单算术计算非常轻量级，无需掩码直接全量计算即可
            # ==========================================
            cx = (bbox[..., 0] + bbox[..., 2]) / 2.0  # [V, bs, 2]
            cy = (bbox[..., 1] + bbox[..., 3]) / 2.0  # [V, bs, 2]
            hand_pos_flat = torch.stack([cy // patch_size, cx // patch_size], dim=-1).long() # [1, 2, 2]
            hand_pos_flat = hand_pos_flat.unsqueeze(2).repeat(
                1, 1, HAND_TOKENS_PER_HAND, 1
            )
            smpl_pos_cut3r = hand_pos_flat.view(
                batch_size, NUM_HAND_TOKENS, 2
            )

            smpl_feat_i = full_prompts.view(
                batch_size, NUM_HAND_TOKENS, embed_dim
            )
            smpl_pos_i = smpl_pos_cut3r # (1,36, 2)
     
            n_humans_i = NUM_HAND_TOKENS
            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, cross_attn_states = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                smpl_feat_i,
                smpl_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                use_ttt3r=use_ttt3r,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            if n_humans_i > 0:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:-n_humans_i].float(),
                    dec[self.dec_depth][:, :-n_humans_i].float(),
                ]
                hand_origin = hand_full_out.view(
                    batch_size, NUM_HAND_TOKENS, 1280
                )
                smpl_token, contact_logits = self._predict_contact(
                    dec[self.dec_depth],
                    hand_origin,
                    bbox,
                    shape,
                    view["mano_mask"],
                )
                smpl_token_cat = torch.cat(
                    [smpl_token, hand_origin], dim=-1
                )
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
                smpl_token = None
                smpl_token_cat = None
                contact_logits = None
            res = self._downstream_head(
                head_input,
                shape,
                pos=pos_i,
                n_humans=n_humans_i,
                smpl_token=smpl_token_cat,
                mano_mask=view["mano_mask"],
            )
            if contact_logits is not None:
                res["contact_logits"] = contact_logits

            res_cpu = to_cpu({**res})
            ress.append(res_cpu)
            # ress.append(res)

            # updating the state and memory
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            if use_ttt3r and i != 0 and not reset_mask:
                cross_attn_states = rearrange(torch.cat(cross_attn_states, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)').mean(dim=(-1, -2))
                update_mask_state = update_mask * torch.sigmoid(cross_attn_states)[..., None]
            else:
                update_mask_state = update_mask

            state_feat = new_state_feat * update_mask_state + state_feat * (
                1 - update_mask_state
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
           
        if ret_state:
            return ress, views, all_state_args
        return ress, views




if __name__ == "__main__":
    print(ARCroco3DStereo.mro())


ARCroco3DStereo(ARCroco3DStereoConfig(freeze='encoder_and_decoder_and_head', 
  state_size=768, state_pe='2d', pos_embed='RoPE100',
  rgb_head=True, pose_head=True, patch_embed_cls='ManyAR_PatchEmbed', img_size=(512,
  512), head_type='dpt', output_mode='pts3d+pose+mano', depth_mode=('exp', -inf, inf),
  conf_mode=('exp', 1, inf), pose_mode=('exp', -inf, inf), enc_embed_dim=1024, enc_depth=24,
  enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, landscape_only=False))
