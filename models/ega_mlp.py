# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 26, 2026
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class EGAMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        nn.init.zeros_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)

    def forward(self, x):
        out = x + self.adapter(x)
        out = F.normalize(out, p=2, dim=1)
        return out