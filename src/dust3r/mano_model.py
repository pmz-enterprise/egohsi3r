import torch
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
from dust3r.utils.geometry import (
    perspective_projection, 
    resize_camera_intrinsics,
    get_camera_parameters
)
from dust3r.utils.image import pad_image
import roma
import pickle
from wilor.models.mano_wrapper import MANO
import torchvision.ops as ops
import ipdb
from dust3r.runtime import get_mano_data_dir

MANO_DATA_DIR = str(get_mano_data_dir())

mano_cfg = {
            'model_path': MANO_DATA_DIR,
            'gender': 'neutral',
            'num_hand_joints': 15,
            'create_body_pose': False,
        }

HAND_ORDER = ('right', 'left')


def build_native_mano_layers(device=None):
    """Build MANO layers in the dataset hand order: right, then left."""
    layers = torch.nn.ModuleList([
        MANO(**mano_cfg, is_rhand=True),
        MANO(**mano_cfg, is_rhand=False),
    ])
    if device is not None:
        layers = layers.to(device)
    return layers


def forward_native_mano(mano_layers, pose, shape, transl, hand_indices):
    """Run each sample through its native MANO side without parameter mirroring."""
    if pose.ndim != 4 or pose.shape[1:] != (16, 3, 3):
        raise ValueError(f'Expected pose [N, 16, 3, 3], got {tuple(pose.shape)}')

    batch_size = pose.shape[0]
    if shape.shape[0] != batch_size or transl.shape[0] != batch_size:
        raise ValueError('pose, shape, and transl must have the same batch size')

    hand_indices = torch.as_tensor(
        hand_indices, device=pose.device, dtype=torch.long
    ).reshape(-1)
    if hand_indices.numel() != batch_size:
        raise ValueError('hand_indices must contain one value per MANO sample')
    if ((hand_indices < 0) | (hand_indices >= len(HAND_ORDER))).any():
        raise ValueError('hand_indices must use 0=right and 1=left')

    if batch_size == 0:
        layer = mano_layers[0]
        num_vertices = int(layer.v_template.shape[-2])
        num_joints = int(layer.joint_map.numel())
        return {
            'vertices': layer.v_template.new_empty((0, num_vertices, 3)),
            'joints': layer.v_template.new_empty((0, num_joints, 3)),
        }

    sample_indices = []
    vertices = []
    joints = []
    for hand_index, layer in enumerate(mano_layers):
        side_indices = torch.where(hand_indices == hand_index)[0]
        if side_indices.numel() == 0:
            continue

        output = layer(
            global_orient=pose[side_indices][:, [0]].float(),
            hand_pose=pose[side_indices, 1:].float(),
            betas=shape[side_indices].reshape(side_indices.numel(), -1).float(),
            transl=transl[side_indices].reshape(side_indices.numel(), -1).float(),
            pose2rot=False,
        )
        sample_indices.append(side_indices)
        vertices.append(output.vertices)
        joints.append(output.joints)

    restore_order = torch.argsort(torch.cat(sample_indices, dim=0))
    return {
        'vertices': torch.cat(vertices, dim=0)[restore_order],
        'joints': torch.cat(joints, dim=0)[restore_order],
    }


class MANOModel(object):
    def __init__(self, device):
        self.device = device
        self.mano_layers = build_native_mano_layers(self.device)

    @property
    def mano_layer(self):
        """Backward-compatible alias for the native right-hand layer."""
        return self.mano_layers[0]

    @property
    def left_mano_layer(self):
        return self.mano_layers[1]

    def process_hand_crops(self,
                        imgs: torch.Tensor, 
                        bbox: torch.Tensor, 
                        mano_mask: torch.Tensor) -> torch.Tensor:
        """
        根据 bbox 裁剪左右手并缩放到指定大小，保持两侧原始像素方向。
        处理了 mano_mask=False 时 bbox=[0,0,0,0] 的情况。
        
        参数:
            imgs: [N, 3, 384, 512] - 原始图像 (N = num_views * B)
            bbox: [N, 2, 4] - 边界框 [x1, y1, x2, y2]
            mano_mask: [N, 2] - 表示手是否有效

        # mano_mask [num_views X B, 2] 手是否有效
        # bbox [num_views X B, 2, 4]
        # imgs [num_views X B, 3, 384, 512]
        # output
        # resized_crop [num_views X B, 2, 3, 256, 192]
        # 每个 0 号为右手，1 号为左手
            
        返回:
            resized_crop: [N, 2, 3, 256, 192]
        """
        N = imgs.shape[0]
        device = imgs.device
        dtype = imgs.dtype
        
        # 1. 安全处理无效的 bbox
        # 确保 mano_mask 是布尔类型
        mask_bool = mano_mask.bool() 
        # 扩展 mask 形状为 [N, 2, 1] 以利用广播匹配 bbox 的 [N, 2, 4]
        mask_expanded = mask_bool.unsqueeze(-1) 
        
        # 构建一个安全的 Dummy Box [0, 0, 1, 1]
        dummy_box = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=bbox.dtype)
        
        # 如果有效，使用真实的 bbox；如果无效，使用 dummy_box 防止 roi_align 报错
        safe_bbox = torch.where(mask_expanded, bbox, dummy_box)
        
        # 2. 构造 rois 张量: 格式为 [batch_index, x1, y1, x2, y2]
        batch_idx = torch.arange(N, device=device, dtype=bbox.dtype).view(N, 1, 1).expand(N, 2, 1)
        rois = torch.cat([batch_idx, safe_bbox], dim=-1).view(-1, 5)
        rois = rois.to(torch.float32)
        # 3. 执行批量裁剪和缩放 (Crop and Resize)
        crops = ops.roi_align(imgs, rois, output_size=(256, 192), spatial_scale=1.0, aligned=True)
        
        # 4. 重新塑形为 [N, 2, 3, 256, 192]
        resized_crop = crops.view(N, 2, 3, 256, 192)
        
        # 5. 使用 mano_mask 彻底清理无效手的数据
        # 刚刚用 dummy_box 截取到的无意义图像像素，在这里被严格置为 0
        mask_final = mask_bool.view(N, 2, 1, 1, 1).to(dtype)
        resized_crop = resized_crop * mask_final

        return resized_crop

# crop
    def process_mano(self, pose, shape, transl, K, hand_indices):
        output = forward_native_mano(
            self.mano_layers, pose, shape, transl, hand_indices
        )
        verts = output['vertices']
        j3d = output['joints']
        # Projection in camera plane
        j2d = perspective_projection(j3d, K)
        v2d = perspective_projection(verts, K)
        out = {}
        out.update({
            'mano_v3d': verts, # in 3d camera space
            'mano_j3d': j3d, # in 3d camera space
            'mano_j2d': j2d, 
            'mano_v2d': v2d, 
            'mano_transl': transl, # translation of the primary keypoint in camera coordinates
            'mano_transl_pelvis': j3d[:, 0], # root=pelvis in camera coordinates
        })
        
        return out
                    # 输入
                    # mano_mask=mano_mask, [2]
                    # bbox = bbox, [2,4]
                    # theta = theta, [2,48]
                    # beta = beta, [2,10]
                    # transl = transl [2,3]



                    #目标： 
# "resized_crop" #
# mano_rotmat
# mano_shape
# mano_transl
# 

    def update_mano_gt(self, views):
        target = {}
        _target = {}
        

        bbox = torch.stack([view['bbox'] for view in views], dim=0)
        bbox = bbox.view(-1, *bbox.shape[2:])

        mano_rotvec = torch.stack([view['theta'] for view in views], dim=0)#[bv, 2, 48]
        mano_rotvec = mano_rotvec.view(-1, *mano_rotvec.shape[2:])
        mano_shape = torch.stack([view['mano_shape'] for view in views], dim=0)
        mano_shape = mano_shape.view(-1, *mano_shape.shape[2:])
        mano_transl = torch.stack([view['mano_transl'] for view in views], dim=0)
        mano_transl = mano_transl.view(-1, *mano_transl.shape[2:])

        mano_mask = torch.stack([view['mano_mask'] for view in views], dim=0)
        mano_mask = mano_mask.view(-1, *mano_mask.shape[2:])
        crop_mask = mano_mask
        parameter_valid = torch.stack([
            view.get('mano_param_valid', torch.ones_like(view['mano_mask']))
            for view in views
        ]).reshape_as(mano_mask)
        mano_mask = mano_mask & parameter_valid

        idx_h = torch.where(mano_mask) # vb_idx, human_idx
        K = torch.stack(
            [view['camera_intrinsics'] for view in views], dim=0
        )
        K = K.view(-1, *K.shape[2:])

        # Get MHMR input image (high-res, square)
        imgs = torch.stack([view["img"] for view in views], dim=0)
        imgs = imgs.view(-1, *imgs.shape[2:])

        _target['resized_crop'] = self.process_hand_crops(imgs, bbox, crop_mask)
        mano_rotmat = roma.rotvec_to_rotmat(
            mano_rotvec[mano_mask].reshape(-1, 16, 3)
        )
        target.update(self.process_mano(
            mano_rotmat,
            mano_shape[mano_mask],
            mano_transl[mano_mask],
            K=K[idx_h[0]],
            hand_indices=idx_h[1],
        ))
        target['mano_rotmat'] = mano_rotmat

        
        true_shapes = torch.stack([view["true_shape"] for view in views], dim=0)
        if len(torch.unique(true_shapes, dim=0)) != 1:
            raise NotImplementedError
        
        num_view = len(views)
        max_humans = 2
        batch_size = views[0]["img"].shape[0]
        for k, v in target.items():
            full_out = torch.zeros(
                num_view * batch_size, max_humans, *v.shape[1:], 
                device=v.device, dtype=v.dtype,
            )
            full_out[mano_mask] = v # discard unvisible humans due to olccusion
            _target[k] = full_out.chunk(num_view, dim=0) # .view(num_view, batch_size, *full_out.shape[1:])

        _target['resized_crop'] = _target['resized_crop'].chunk(num_view, dim=0) 
        for i, v in enumerate(zip(*_target.values())):
            views[i].update(dict(zip(_target.keys(), v)))
