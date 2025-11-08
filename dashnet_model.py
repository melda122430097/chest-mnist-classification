import math
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B, C, n_h, n_w, p, p)
        x = x.contiguous().view(B, C, -1, p, p)        # (B, C, N, p, p)
        x = x.permute(0, 2, 1, 3, 4)                   # (B, N, C, p, p)
        x = x.reshape(B, self.num_patches, -1)         # (B, N, C*p*p)
        x = self.proj(x)                               # (B, N, embed_dim)
        return x

class DashNet(nn.Module):
    """
    Non-CNN DashNet: small Vision-Transformer style model (patch + Transformer encoder).
    - No convolutional layers used.
    - Compatible with train.py through model.py re-export (SimpleCNN alias).
    - For binary (num_classes==2) returns shape (N,) to match existing training code.
    """
    def __init__(self, in_channels=1, num_classes=2, img_size=28,
                 patch_size=7, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        # Patch embedding (implemented with linear, not conv)
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                          in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder (uses only linear + attention, no conv)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes)

        # init head
        nn.init.normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)
        x = self.patch_embed(x)             # (B, N, E)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,E)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+N, E)
        x = x + self.pos_embed                         # add pos embedding

        x = self.encoder(x)                            # (B, 1+N, E)
        cls_out = x[:, 0]                              # (B, E)
        cls_out = self.norm(cls_out)
        out = self.head(cls_out)                       # (B, 1) or (B, C)

        if self.num_classes == 2:
            return out.view(-1)   # (B,) to match existing train.py expectation
        return out