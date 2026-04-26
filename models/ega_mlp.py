import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGating(nn.Module):
    def __init__(self, dim):
        super(FeatureGating, self).__init__()
        # This branch learns a scaling factor for each dimension
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)

class EGABlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(EGABlock, self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        self.gate = FeatureGating(dim)
        
    def forward(self, x):
        # A dual-path residual update: local transformation + global gating
        res = self.conv_branch(x)
        res = self.gate(res)
        return x + res

class EGAMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, num_blocks=2):
        super(EGAMLP, self).__init__()
        # Stacking multiple specialized blocks instead of a flat MLP
        self.blocks = nn.ModuleList([
            EGABlock(input_dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        # Final refinement layer
        self.refiner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        
        out = self.refiner(out)
        # Final hypersphere projection for optimized vector search
        return F.normalize(out, p=2, dim=1)