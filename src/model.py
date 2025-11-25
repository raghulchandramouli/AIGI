import torch
import torch.nn as nn
import timm

class MFMViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool=''
        )

        # Lightweight Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.patch_dim)
        )   
    
    def unpatchify(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.num_patches, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4)
        
        patches_per_side = self.img_size // self.patch_size
        x = x.reshape(B, 3, patches_per_side, self.patch_size, patches_per_side, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, 3, self.img_size, self.img_size)
        return x
    
    def forward(self, x):
        tokens = self.encoder(x)
        patches = self.decoder(tokens)
        reconstructed = self.unpatchify(patches)
        return reconstructed
