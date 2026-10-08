import torch
import torch.nn as nn
import torch.nn.functional as F

class HandPromptBuilder(nn.Module):
    def __init__(self, 
                 dim_scene=1024,  # ViT/ResNet 输出特征维度
                 dim_prior=1280,  # Hand Prior 特征维度
                 embed_dim=768,   # Decoder 的工作维度
                 dropout=0.1):    
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # --- 1. 投影层 (Projections) ---
        
        # A. 处理 Hand Prior (序列数据)
        # [B, 18, 1280] -> [B, 18, embed_dim]
        self.prior_proj = nn.Sequential(
            nn.Linear(dim_prior, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # B. 处理 Scene Feature (图像/序列数据)
        # [B, 1024, H, W] -> [B, embed_dim] (Global Context)
        # 修改点：使用 Linear + LN 替代 Conv + BN
        self.scene_proj = nn.Sequential(
            nn.Linear(dim_scene, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # C. 处理左右手 (Side Embedding)
        # [B] -> [B, embed_dim]
        self.side_embedding = nn.Embedding(2, embed_dim)
        
        # --- 2. 融合机制 (Fusion) ---
        
        # 门控单元：决定融入多少 Scene 信息
        # Input: [Prior + Scene] -> Output: Gate (0~1)
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim), # 加个 LN 更稳定
            nn.Sigmoid()
        )
        
        # 最终的混合 MLP (比单层 Linear 更强)
        self.mixer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), # 升维
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), # 降维回原状
            nn.LayerNorm(embed_dim)
        )

    def forward(self, f_scene, f_prior, is_left_idx):
        """
        输入:
            f_scene: [B, C, H, W] (e.g., [B, 1024, 7, 7])
            f_prior: [B, 18, 1280]
            is_left_idx: [B] (0=Right, 1=Left)
        输出:
            prompt_query: [B, 18, embed_dim]
        """
        B = f_scene.shape[0]
        
        # -----------------------------------------------------------
        # Step 1: 投影与对齐 (Align)
        # -----------------------------------------------------------
        
        # 1. Prior 投影: [B, 18, 1280] -> [B, 18, embed_dim]
        q_prior = self.prior_proj(f_prior) 
        
        # 2. Side Embedding: [B] -> [B, 1, embed_dim]
        q_side = self.side_embedding(is_left_idx).unsqueeze(1)
        
        # 3. Scene Context 处理
        # 输入可能是 [B, C, H, W]，我们需要处理成 [B, embed_dim]
        if f_scene.dim() == 4:
            # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            x_scene = f_scene.flatten(2).transpose(1, 2)
        else:
            x_scene = f_scene # 假设已经是 [B, L, C]
            
        # 先投影: [B, L, C] -> [B, L, embed_dim]
        x_scene = self.scene_proj(x_scene)
        
        # 再做 Global Average Pooling，提取全局上下文
        # [B, L, embed_dim] -> [B, embed_dim]
        q_scene_global = x_scene.mean(dim=1) 
        
        # 扩展维度以便广播: [B, 1, embed_dim]
        q_scene_global = q_scene_global.unsqueeze(1)
        
        # -----------------------------------------------------------
        # Step 2: 注入融合 (Inject & Gate)
        # -----------------------------------------------------------
        
        # A. 基础 Query = 先验 + 左右手信息
        # q_base: [B, 18, embed_dim]
        q_base = q_prior + q_side 
        
        # B. 计算门控系数
        # 将 q_scene_global 复制 18 份与 q_base 拼接
        # [B, 18, embed_dim * 2]
        gate_input = torch.cat([
            q_base, 
            q_scene_global.expand(-1, q_prior.size(1), -1)
        ], dim=-1)
        
        # gate: [B, 18, embed_dim]
        gate = self.context_gate(gate_input)
        
        # C. 注入 Scene 信息 (残差连接思想)
        # Query = Base + Gate * Scene
        q_fused = q_base + gate * q_scene_global
        
        # -----------------------------------------------------------
        # Step 3: 最终混合 (Mix)
        # -----------------------------------------------------------
        
        # 通过 MLP 深度融合特征分布
        prompt_query = self.mixer(q_fused)
        
        return prompt_query