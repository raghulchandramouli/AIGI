import torch
import torch.nn as nn
import timm

class MFMViT(nn.Module):
    def __init__(self, img_size=224, model_name='vit_base_patch16_224'):
        super().__init__()
        self.img_size = img_size
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        self._initialized = False

    def init_from_sample(self, x):
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder(x)
        if tokens.dim() == 2:
            raise RuntimeError("Encoder returned pooled output (B, D). Make sure timm model returns token sequence "
                               "by using num_classes=0 and global_pool=''.")
        B, T, D = tokens.shape
        self.embed_dim = D
        self.num_patches = T - 1
        if int(self.num_patches ** 0.5) ** 2 != self.num_patches:
            raise RuntimeError(f"Non-square patch grid detected: num_patches={self.num_patches}")
        self.patches_per_side = int(self.num_patches ** 0.5)
        self.patch_size = self.img_size // self.patches_per_side
        self.patch_dim = 3 * self.patch_size * self.patch_size

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.patch_dim)
        ).to(x.device)  # Move decoder to same device as input
        
        self._initialized = True
        print(f"[MFMViT init] embed_dim={self.embed_dim}, num_patches={self.num_patches}, "
              f"patches_per_side={self.patches_per_side}, patch_size={self.patch_size}, patch_dim={self.patch_dim}")

    def unpatchify(self, patches):
        B, N, PD = patches.shape
        assert N == self.num_patches and PD == self.patch_dim, f"unpatchify mismatch: {patches.shape}"
        p = self.patch_size
        ps = self.patches_per_side
        patches = patches.reshape(B, ps, ps, 3, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        img = patches.reshape(B, 3, self.img_size, self.img_size)
        return img

    def forward(self, x):
        if not self._initialized:
            self.init_from_sample(x)

        tokens = self.encoder(x)
        if tokens.dim() != 3:
            raise RuntimeError("Unexpected encoder output dim. Expected (B, T, D).")

        patch_tokens = tokens[:, 1:, :]
        B, N, D = patch_tokens.shape
        patches_flat = self.decoder(patch_tokens.reshape(B * N, D))
        patches = patches_flat.reshape(B, N, self.patch_dim)
        reconstructed = self.unpatchify(patches)
        return reconstructed

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MFMViT(img_size=224).to(device)
    B = 4
    dummy = torch.randn(B, 3, 224, 224).to(device)
    out = model(dummy)
    print("output shape:", out.shape)
