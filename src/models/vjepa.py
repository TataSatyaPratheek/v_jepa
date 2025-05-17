import torch
from torch import nn

class MemoryEfficientViT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, dim=384):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, dim))
        
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=6,
                dim_feedforward=dim*2,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(6)]
        )

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embed
        return self.blocks(x)

class VJEPA(nn.Module):
    def __init__(self, base_encoder=MemoryEfficientViT):
        super().__init__()
        self.context_encoder = base_encoder()
        self.target_encoder = base_encoder()
        self.predictor = nn.Sequential(
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, 384)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
