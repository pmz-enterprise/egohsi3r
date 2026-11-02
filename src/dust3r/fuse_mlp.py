import torch
import torch.nn as nn
import torch.nn.functional as F

class HandPromptBuilder(nn.Module):
    def __init__(self, 
                 dim_scene=1024,
                 dim_prior=1280,
                 embed_dim=768,
                 dropout=0.1):    
        super().__init__()
        
        self.embed_dim = embed_dim
        

        

        # [B, 18, 1280] -> [B, 18, embed_dim]
        self.prior_proj = nn.Sequential(
            nn.Linear(dim_prior, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        

        # [B, 1024, H, W] -> [B, embed_dim] (Global Context)

        self.scene_proj = nn.Sequential(
            nn.Linear(dim_scene, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        

        # [B] -> [B, embed_dim]
        self.side_embedding = nn.Embedding(2, embed_dim)
        

        

        # Input: [Prior + Scene] -> Output: Gate (0~1)
        self.context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Sigmoid()
        )
        

        self.mixer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, f_scene, f_prior, is_left_idx):
        """Inputs:
            f_scene: [B, C, H, W] (e.g., [B, 1024, 7, 7])
            f_prior: [B, 18, 1280]
            is_left_idx: [B] (0=Right, 1=Left)
        Outputs:
            prompt_query: [B, 18, embed_dim]"""
        B = f_scene.shape[0]
        
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        

        q_prior = self.prior_proj(f_prior) 
        
        # 2. Side Embedding: [B] -> [B, 1, embed_dim]
        q_side = self.side_embedding(is_left_idx).unsqueeze(1)
        


        if f_scene.dim() == 4:
            # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            x_scene = f_scene.flatten(2).transpose(1, 2)
        else:
            x_scene = f_scene
            

        x_scene = self.scene_proj(x_scene)
        

        # [B, L, embed_dim] -> [B, embed_dim]
        q_scene_global = x_scene.mean(dim=1) 
        

        q_scene_global = q_scene_global.unsqueeze(1)
        
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        

        # q_base: [B, 18, embed_dim]
        q_base = q_prior + q_side 
        


        # [B, 18, embed_dim * 2]
        gate_input = torch.cat([
            q_base, 
            q_scene_global.expand(-1, q_prior.size(1), -1)
        ], dim=-1)
        
        # gate: [B, 18, embed_dim]
        gate = self.context_gate(gate_input)
        

        # Query = Base + Gate * Scene
        q_fused = q_base + gate * q_scene_global
        
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        

        prompt_query = self.mixer(q_fused)
        
        return prompt_query