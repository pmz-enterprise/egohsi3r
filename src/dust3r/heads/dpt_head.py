# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

from einops import rearrange
from copy import deepcopy
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import (
    postprocess,
    postprocess_desc,
    postprocess_rgb,
    postprocess_pose_conf,
    postprocess_pose,
    reg_dense_conf,
    postprocess_smpl,
    postprocess_score,
    postprocess_mano
)
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter, Interpolate  # noqa
from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder
from dust3r.blocks import ConditionModulationBlock
from dust3r.hand_constants import (
    HAND_TOKENS_PER_HAND,
    LEFT_HAND_INDEX,
    MANO_NUM_JOINTS,
    NUM_HANDS,
    RIGHT_HAND_INDEX,
)
from dust3r.runtime import get_mano_mean_params_path

MANO_MEAN_PARAMS = str(get_mano_mean_params_path())
from torch.utils.checkpoint import checkpoint
from dust3r.smpl_model import SMPLDecoder, MEAN_PARAMS, regression_mlp
import numpy as np
import ipdb


class SideSpecificMANODecoder(nn.Module):
    """Route canonical right/left token slots through independent decoders."""

    def __init__(self, hidden_size, target_dim, num_layers=2, mlp_ratio=1.0):
        super().__init__()
        self.right = SMPLDecoder(
            hidden_size=hidden_size,
            target_dim=target_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
        )
        # Both sides start from the same function, then receive independent
        # gradients from their native MANO supervision.
        self.left = deepcopy(self.right)

    def forward(self, features):
        if features.ndim < 3 or features.shape[1] != NUM_HANDS:
            raise ValueError(
                "Expected MANO features with hand dimension [B, 2, ...], "
                f"got {tuple(features.shape)}"
            )

        right = self.right(features[:, RIGHT_HAND_INDEX])
        left = self.left(features[:, LEFT_HAND_INDEX])
        return torch.stack((right, left), dim=1)


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)

        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, ret_feat=False):
        assert (
            self.dim_tokens_enc is not None
        ), "Need to call init(dim_tokens_enc) function first"

        image_size = self.image_size if image_size is None else image_size
        H, W = image_size

        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        layers = [encoder_tokens[hook] for hook in self.hooks]

        layers = [self.adapt_tokens(l) for l in layers]

        layers = [
            rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers
        ]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]

        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        path_4 = self.scratch.refinenet4(layers[3])[
            :, :, : layers[2].shape[2], : layers[2].shape[3]
        ]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        out = self.head(path_1)
        
        if ret_feat:
            return out, path_1

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels=1,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        **kwargs
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(
            output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs
        )
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(
        num_channels=out_nchan + has_conf,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type="regression",
    )



class DPTPts3dPoseSMPL(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False, has_msk=False):
        super(DPTPts3dPoseSMPL, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode

        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose
        self.has_msk = has_msk

        pts_channels = 3 + has_conf
        rgb_channels = has_rgb * 3
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        hooks_idx = [0, 1, 2, 3]
        dim_tokens = [ed, dd, dd, dd]
        head_type = "regression"
        output_width_ratio = 1

        pts_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=pts_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        rgb_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=rgb_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        if hooks_idx is not None:
            pts_dpt_args.update(hooks=hooks_idx)
            rgb_dpt_args.update(hooks=hooks_idx)

        self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    net.dec_embed_dim,
                    net.dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=net.rope,
                )
                for _ in range(2)
            ]
        )

        self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_cross.init(**dpt_init_args)

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
            dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
            self.dpt_rgb.init(**dpt_init_args)

        in_dim = net.dec_embed_dim
        self.in_dim = in_dim
        if has_pose:
            self.pose_head = PoseDecoder(hidden_size=in_dim)

        # MHMR Heads - Detection
        backbone_dim = net.backbone_dim
        self.bb_patch_size = net.bb_patch_size
        self.mlp_classif = regression_mlp([backbone_dim, backbone_dim, 1]) # bg or human
        self.mlp_offset = regression_mlp([backbone_dim, backbone_dim, 2]) # offset
        if has_msk:
            self.mlp_msk = SMPLDecoder(
                hidden_size=backbone_dim, target_dim=self.bb_patch_size**2, num_layers=2, mlp_ratio=1)

        # feature fuse
        self.mlp_fuse = SMPLDecoder(hidden_size=ed+backbone_dim, target_dim=dd, num_layers=2, mlp_ratio=4)

        # SMPL
        self.joint_rep_type, self.joint_rep_dim = '6d', 6
        self.nrot = 53
        self.num_body_joints = self.nrot - 1

        npose = self.joint_rep_dim * (self.num_body_joints + 1)
        self.npose = npose

        self.input_is_mean_shape = True
        self.num_betas = 10
        assert self.num_betas in [10, 11]

        # SMPL param heads
        self.deccam = SMPLDecoder(
                        hidden_size=in_dim, 
                        target_dim=3, 
                        num_layers=2, 
                        mlp_ratio=4)
        self.decpose, self.decshape, self.decexpression = [
            SMPLDecoder(
                hidden_size=in_dim+backbone_dim, 
                target_dim=od, 
                num_layers=2, 
                mlp_ratio=4) for od in [self.npose, self.num_betas, 10]]

        self.set_smpl_init()


    def set_smpl_init(self):
        """ Fetch saved SMPL parameters and register buffers."""
        mean_params = np.load(MEAN_PARAMS)
        if self.nrot == 53:
            init_body_pose = torch.eye(3).reshape(1,3,3).repeat(self.nrot,1,1)[:,:,:2].flatten(1).reshape(1, -1)
            init_body_pose[:,:24*6] = torch.from_numpy(mean_params['pose'][:]).float() # global_orient+body_pose from SMPL
        else:
            init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)

        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        init_betas_kid = torch.cat([init_betas, torch.zeros_like(init_betas[:,[0]])],1)
        init_expression = 0. * torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)

        if self.num_betas == 11:
            init_betas = torch.cat([init_betas, torch.zeros_like(init_betas[:,:1])], 1)

        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_betas_kid', init_betas_kid)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_expression', init_expression)
        
    def detect_mhmr(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            scores = postprocess_score(self.mlp_classif(x)) # per token detection score.
        return scores

    def segment(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            msks = postprocess_score(self.mlp_msk(x))
        return msks
    
    def forward(self, x, img_info, **kwargs):
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            n_humans_i = kwargs.get("n_humans")
            token = x[-1][:, 1:]

            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
                if n_humans_i > 0:
                    smpl_token = kwargs.get("smpl_token")   # CUT3R smpl token (bs, 10, 768)
                    pred_body_pose = self.decpose(smpl_token) + self.init_body_pose
                    pred_betas = self.decshape(smpl_token) + self.init_betas
                    pred_cam = self.deccam(smpl_token[..., :self.in_dim])
                    pred_expression = self.decexpression(smpl_token) + self.init_expression
                    pred_smpl = [pred_body_pose, pred_betas, pred_cam, pred_expression]
                    
            token_cross = token.clone()
            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))
            x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        with torch.cuda.amp.autocast(enabled=False):
            self_out = checkpoint(
                self.dpt_self,
                x,
                image_size=(img_info[0], img_info[1]),
                # ret_feat=True,
                use_reentrant=False,
            )

            final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_out = checkpoint(
                    self.dpt_rgb,
                    x,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                rgb_output = postprocess_rgb(rgb_out)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                cross_out = checkpoint(
                    self.dpt_cross,
                    x_cross,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")
            if n_humans_i > 0:
                smpl_out = postprocess_smpl(pred_smpl, self.depth_mode)
                final_output.update(smpl_out)

        return final_output


class DPTPts3dPoseMANO(nn.Module):
    def __init__(self, net, has_conf=False, has_rgb=False, has_pose=False, has_msk=False):
        super(DPTPts3dPoseMANO, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.pose_mode = net.pose_mode

        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose
        self.has_msk = has_msk

        pts_channels = 3 + has_conf
        rgb_channels = has_rgb * 3
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        hooks_idx = [0, 1, 2, 3]
        dim_tokens = [ed, dd, dd, dd]
        head_type = "regression"
        output_width_ratio = 1

        pts_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=pts_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        rgb_dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=rgb_channels,
            feature_dim=feature_dim,
            last_dim=last_dim,
            dim_tokens=dim_tokens,
            hooks_idx=hooks_idx,
            head_type=head_type,
        )
        if hooks_idx is not None:
            pts_dpt_args.update(hooks=hooks_idx)
            rgb_dpt_args.update(hooks=hooks_idx)

        self.dpt_self = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_self.init(**dpt_init_args)

        self.final_transform = nn.ModuleList(
            [
                ConditionModulationBlock(
                    net.dec_embed_dim,
                    net.dec_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    rope=net.rope,
                )
                for _ in range(2)
            ]
        )

        self.dpt_cross = DPTOutputAdapter_fix(**pts_dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt_cross.init(**dpt_init_args)

        if has_rgb:
            self.dpt_rgb = DPTOutputAdapter_fix(**rgb_dpt_args)
            dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
            self.dpt_rgb.init(**dpt_init_args)

        if has_pose:
            in_dim = net.dec_embed_dim
            self.pose_head = PoseDecoder(hidden_size=in_dim)
            self.in_dim = in_dim

        # WiLoR contributes a 1280-dimensional feature to each MANO token.
        backbone_dim = 1280

        # SMPL
        self.joint_rep_type, self.joint_rep_dim = '6d', 6
        self.nrot = 16
        self.num_body_joints = self.nrot - 1

        npose = self.joint_rep_dim * (self.num_body_joints + 1)
        self.npose = npose

        self.input_is_mean_shape = True
        self.num_betas = 10

        mano_token_dim = in_dim + backbone_dim
        self.mano_token_dim = mano_token_dim
        self.deccam = SideSpecificMANODecoder(
            hidden_size=in_dim,
            target_dim=3,
            num_layers=2,
            mlp_ratio=4,
        )
        self.decpose = SideSpecificMANODecoder(
            hidden_size=mano_token_dim,
            target_dim=6,
            num_layers=2,
            mlp_ratio=0.25,
        )
        self.decshape = SideSpecificMANODecoder(
            hidden_size=mano_token_dim,
            target_dim=10,
            num_layers=2,
            mlp_ratio=0.25,
        )

        self.set_mano_init()


    def set_mano_init(self):
        """ Fetch saved SMPL parameters and register buffers."""


        mean_params = np.load(MANO_MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_cam', init_cam)
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)

    def decode_mano_tokens(self, mano_tokens):
        """Decode right/left MANO tokens without changing image handedness."""
        expected_tokens = NUM_HANDS * HAND_TOKENS_PER_HAND
        if (
            mano_tokens.ndim != 3
            or mano_tokens.shape[1] != expected_tokens
            or mano_tokens.shape[2] != self.mano_token_dim
        ):
            raise ValueError(
                "Expected MANO tokens "
                f"[B, {expected_tokens}, {self.mano_token_dim}], got "
                f"{tuple(mano_tokens.shape)}"
            )

        bs = mano_tokens.shape[0]
        tokens = mano_tokens.reshape(
            bs, NUM_HANDS, HAND_TOKENS_PER_HAND, self.mano_token_dim
        )
        pose_tokens = tokens[:, :, :MANO_NUM_JOINTS]
        beta_token = tokens[:, :, MANO_NUM_JOINTS]
        cam_token = tokens[:, :, MANO_NUM_JOINTS + 1]

        pose_6d = self.decpose(pose_tokens)
        pose_out = pose_6d.reshape(bs, NUM_HANDS, self.npose)
        pose_out = pose_out + self.init_hand_pose
        shape_out = self.decshape(beta_token) + self.init_betas
        # Keep the current decoder-token input while allowing native left-hand
        # translation to learn independently from the right-hand branch.
        cam_out = self.deccam(cam_token[..., :self.in_dim])

        return [pose_out, shape_out, cam_out]
    
    def forward(self, x, img_info, **kwargs):
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            n_humans_i = kwargs.get("n_humans")
            token = x[-1][:, 1:]

            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
                if n_humans_i > 0:
                    smpl_token = kwargs.get("smpl_token")   # [bs,36,2008]
                    pred_mano = self.decode_mano_tokens(smpl_token)
                    
            token_cross = token.clone()
            for blk in self.final_transform:
                token_cross = blk(token_cross, pose_token, kwargs.get("pos"))
            x = x[:-1] + [token]
            x_cross = x[:-1] + [token_cross]

        with torch.cuda.amp.autocast(enabled=False):
            self_out = checkpoint(
                self.dpt_self,
                x,
                image_size=(img_info[0], img_info[1]),
                # ret_feat=True,
                use_reentrant=False,
            )

            final_output = postprocess(self_out, self.depth_mode, self.conf_mode)
            final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
            final_output["conf_self"] = final_output.pop("conf")

            if self.has_rgb:
                rgb_out = checkpoint(
                    self.dpt_rgb,
                    x,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                rgb_output = postprocess_rgb(rgb_out)
                final_output.update(rgb_output)

            if self.has_pose:
                pose = postprocess_pose(pose, self.pose_mode)
                final_output["camera_pose"] = pose  # B,7
                cross_out = checkpoint(
                    self.dpt_cross,
                    x_cross,
                    image_size=(img_info[0], img_info[1]),
                    use_reentrant=False,
                )
                tmp = postprocess(cross_out, self.depth_mode, self.conf_mode)
                final_output["pts3d_in_other_view"] = tmp.pop("pts3d")
                final_output["conf"] = tmp.pop("conf")

            if n_humans_i > 0:
                mano_out = postprocess_mano(pred_mano, self.depth_mode)
                final_output.update(mano_out)

        return final_output
