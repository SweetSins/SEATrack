import torch.nn as nn
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding with Dual Embedding Support """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True,
                 dual_embedding=False):
        """
        Args:
            img_size (int or tuple): Input image size.
            patch_size (int or tuple): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimension of embedding.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
            flatten (bool): Whether to flatten output to (B, N, C). Defaults to True.
            dual_embedding (bool): Whether to enable dual embedding (global and local). Defaults to False.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.dual_embedding = dual_embedding

        # Main projection layer for patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if self.dual_embedding:
            # Additional layers for global and local embedding
            self.global_proj = nn.Linear(embed_dim, embed_dim)
            self.local_proj = nn.Linear(embed_dim, embed_dim)
            self.global_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            self.local_pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

            # Initialize position embeddings
            nn.init.trunc_normal_(self.global_pos_emb, std=0.02)
            nn.init.trunc_normal_(self.local_pos_emb, std=0.02)

    def forward(self, x):
        """
        Forward pass for Patch Embedding.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Embedded patches of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # Project input to patch embeddings
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # Convert BCHW -> BNC
        x = self.norm(x)  # Normalize embeddings

        if self.dual_embedding:
            # Generate global and local embeddings
            global_emb = self.global_proj(x) + self.global_pos_emb
            local_emb = self.local_proj(x) + self.local_pos_emb
            x = global_emb + local_emb  # Combine embeddings

        return x