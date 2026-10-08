import torch
import torch.nn as nn
from torchvision.ops import roi_align

from dust3r.hand_constants import (
    HAND_TOKENS_PER_HAND,
    MANO_NUM_JOINTS,
    MANO_NUM_VERTICES,
    NUM_HANDS,
)
from dust3r.mano_model import build_native_mano_layers


def _native_mano_skinning_weights():
    layers = build_native_mano_layers("cpu")
    weights = layers[0].lbs_weights.detach().float().clone()
    if not torch.equal(weights, layers[1].lbs_weights.detach().float()):
        raise ValueError("Left and right MANO vertex skinning weights must align")
    return weights


class MANOContactHead(nn.Module):
    """Predict per-MANO-vertex contact from hand and local scene tokens."""

    def __init__(
        self,
        decoder_dim,
        prior_dim=1280,
        hidden_dim=256,
        num_heads=4,
        roi_size=5,
        patch_size=16,
        enable_feedback=False,
        skinning_weights=None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        weights = (
            _native_mano_skinning_weights()
            if skinning_weights is None
            else torch.as_tensor(skinning_weights, dtype=torch.float32)
        )
        expected_shape = (MANO_NUM_VERTICES, MANO_NUM_JOINTS)
        if tuple(weights.shape) != expected_shape:
            raise ValueError(
                f"Expected skinning weights {expected_shape}, got {tuple(weights.shape)}"
            )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("skinning_weights", weights, persistent=True)

        self.decoder_dim = int(decoder_dim)
        self.prior_dim = int(prior_dim)
        self.hidden_dim = int(hidden_dim)
        self.roi_size = int(roi_size)
        self.patch_size = int(patch_size)
        self.enable_feedback = bool(enable_feedback)

        self.hand_proj = nn.Linear(self.decoder_dim, self.hidden_dim)
        self.prior_proj = nn.Linear(self.prior_dim, self.hidden_dim)
        self.scene_proj = nn.Linear(self.decoder_dim, self.hidden_dim)
        self.vertex_embedding = nn.Parameter(
            torch.randn(1, 1, MANO_NUM_VERTICES, self.hidden_dim) * 0.02
        )
        self.side_embedding = nn.Embedding(NUM_HANDS, self.hidden_dim)
        self.missing_scene_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * 0.02
        )

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.scene_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, 1)

        if self.enable_feedback:
            self.feedback = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(
                    self.hidden_dim,
                    HAND_TOKENS_PER_HAND * self.decoder_dim,
                ),
            )
            nn.init.zeros_(self.feedback[-1].weight)
            nn.init.zeros_(self.feedback[-1].bias)
            self.feedback_gate = nn.Parameter(torch.tensor(-4.0))

    def _scene_roi_tokens(self, scene_tokens, bbox, image_size, hand_mask):
        batch_size, num_scene_tokens, channels = scene_tokens.shape
        image_size = torch.as_tensor(image_size, device=scene_tokens.device)
        if image_size.ndim == 1:
            image_size = image_size.unsqueeze(0).expand(batch_size, -1)
        if image_size.shape != (batch_size, 2):
            raise ValueError(
                f"Expected image_size [B,2], got {tuple(image_size.shape)}"
            )
        if not torch.equal(image_size, image_size[:1].expand_as(image_size)):
            raise ValueError("All samples in a batch must have the same image size")

        image_height, image_width = map(int, image_size[0].tolist())
        patch_height = image_height // self.patch_size
        patch_width = image_width // self.patch_size
        if patch_height * patch_width != num_scene_tokens:
            raise ValueError(
                "Scene token count does not match image size: "
                f"{num_scene_tokens} vs {patch_height}x{patch_width}"
            )

        feature_map = scene_tokens.transpose(1, 2).reshape(
            batch_size, channels, patch_height, patch_width
        )
        flat_bbox = bbox.reshape(batch_size * NUM_HANDS, 4).to(
            device=scene_tokens.device, dtype=scene_tokens.dtype
        )
        flat_mask = hand_mask.reshape(-1).bool()
        dummy_bbox = flat_bbox.new_tensor(
            [0.0, 0.0, float(self.patch_size), float(self.patch_size)]
        )
        safe_bbox = torch.where(flat_mask[:, None], flat_bbox, dummy_bbox)
        batch_indices = (
            torch.arange(batch_size, device=scene_tokens.device)
            .repeat_interleave(NUM_HANDS)
            .to(scene_tokens.dtype)
        )
        rois = torch.cat([batch_indices[:, None], safe_bbox], dim=1)
        roi_features = roi_align(
            feature_map,
            rois.float(),
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0 / self.patch_size,
            sampling_ratio=2,
            aligned=True,
        )
        roi_tokens = roi_features.flatten(2).transpose(1, 2)
        roi_tokens = self.scene_proj(roi_tokens).reshape(
            batch_size, NUM_HANDS, self.roi_size**2, self.hidden_dim
        )
        missing = self.missing_scene_token.expand(
            batch_size, NUM_HANDS, self.roi_size**2, -1
        )
        return torch.where(hand_mask[:, :, None, None], roi_tokens, missing)

    def forward(
        self,
        hand_tokens,
        prior_tokens,
        scene_tokens,
        bbox,
        image_size,
        hand_mask=None,
    ):
        batch_size = hand_tokens.shape[0]
        hand_tokens = hand_tokens.reshape(
            batch_size, NUM_HANDS, HAND_TOKENS_PER_HAND, self.decoder_dim
        )
        prior_tokens = prior_tokens.reshape(
            batch_size, NUM_HANDS, HAND_TOKENS_PER_HAND, self.prior_dim
        )
        if hand_mask is None:
            hand_mask = torch.ones(
                batch_size, NUM_HANDS, dtype=torch.bool, device=hand_tokens.device
            )
        else:
            hand_mask = hand_mask.to(device=hand_tokens.device, dtype=torch.bool)

        hand_features = self.hand_proj(hand_tokens) + self.prior_proj(prior_tokens)
        pose_features = hand_features[:, :, :MANO_NUM_JOINTS]
        vertex_features = torch.einsum(
            "vj,bhjd->bhvd", self.skinning_weights, pose_features
        )
        global_features = hand_features[:, :, MANO_NUM_JOINTS:].mean(
            dim=2, keepdim=True
        )
        side_features = self.side_embedding.weight.reshape(
            1, NUM_HANDS, 1, self.hidden_dim
        )
        vertex_features = self.query_norm(
            vertex_features
            + global_features
            + self.vertex_embedding
            + side_features
        )

        scene_roi = self._scene_roi_tokens(
            scene_tokens, bbox, image_size, hand_mask
        )
        query = vertex_features.flatten(0, 1)
        context = scene_roi.flatten(0, 1)
        attended, _ = self.scene_attention(query, context, context, need_weights=False)
        features = self.attention_norm(query + attended)
        features = self.output_norm(features + self.ffn(features))
        features = features.reshape(
            batch_size, NUM_HANDS, MANO_NUM_VERTICES, self.hidden_dim
        )
        contact_logits = self.classifier(features).squeeze(-1)

        hand_delta = None
        if self.enable_feedback:
            contact_summary = features.mean(dim=2)
            hand_delta = self.feedback(contact_summary).reshape(
                batch_size,
                NUM_HANDS,
                HAND_TOKENS_PER_HAND,
                self.decoder_dim,
            )
            hand_delta = hand_delta * torch.sigmoid(self.feedback_gate)
            hand_delta = hand_delta * hand_mask[:, :, None, None]
            hand_delta = hand_delta.reshape(
                batch_size, NUM_HANDS * HAND_TOKENS_PER_HAND, self.decoder_dim
            )

        return contact_logits, hand_delta
