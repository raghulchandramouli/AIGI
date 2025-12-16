import torch
import torch.nn as nn
import timm

class MFMViT_QF(nn.Module):
    """
    MFMViT with FBCNN QF predictor conditioning
    Integration of Modulation into encoder via affine transformations
    """

    def __init__(self, img_size=224, model_name='vit_base_patch16_224',
                 qf_dim=64, use_qf=True):
        
        super().__init__()
        self.img_size = img_size
        self.qf_dim = qf_dim
        self.use_qf = use_qf

        # VIT Encoder
        self.encoder = timm.create_model(model_name, pretrained=False,
                                         num_classes=0, global_pool='')
        
        self._initialized = False

    def init_from_sample(self, x):
        """
        Inits decoder and QF Module after first pass
        """

        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder(x)

        if tokens.dim() == 2:
            raise RuntimeError("Encoder returned pooled o/p (B, D)" \
                                "Make sure timm model returns token seq.")
        
        B, T, D = tokens.shape
        self.embed_dim = D
        self.num_patch = T - 1

        if int(self.num_patch**0.5)**2 != self.num_patch:
            raise RuntimeError("Number of patches should be a perfect square.")
        
        self.patches_per_side = int(self.num_patches ** 0.5)
        self.patch_size = self.img_size // self.patches_per_side
        self.patch_dim = 3 * self.patch_size * self.patch_size     

# PHASE 2: QF modulation layers (alpha and beta)
        if self.use_qf:
            self.qf_alpha = nn.Linear(self.qf_dim, self.embed_dim).to(x.device)
            self.qf_beta = nn.Linear(self.qf_dim, self.embed_dim).to(x.device)
            print(f"[MFMViT-QF] QF modulation enabled: qf_dim={self.qf_dim} -> embed_dim={self.embed_dim}")

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.patch_dim)
        ).to(x.device)
        
        self._initialized = True
        print(f"[MFMViT-QF init] embed_dim={self.embed_dim}, num_patches={self.num_patches}, "
              f"patches_per_side={self.patches_per_side}, patch_size={self.patch_size}, "
              f"patch_dim={self.patch_dim}")

    def unpatchify(self, patches):
        """Convert patches back to image"""
        B, N, PD = patches.shape
        assert N == self.num_patches and PD == self.patch_dim
        p = self.patch_size
        ps = self.patches_per_side
        patches = patches.reshape(B, ps, ps, 3, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        img = patches.reshape(B, 3, self.img_size, self.img_size)
        return img

    def forward(self, x, qf_vector=None):
        """
        Forward pass with optional QF conditioning.
        
        Args:
            x: Input image (B, 3, H, W)
            qf_vector: Quality factor embedding (B, qf_dim) from frozen FBCNN
        
        Returns:
            Reconstructed image (B, 3, H, W)
        """
        if not self._initialized:
            self.init_from_sample(x)

        # PHASE 3: Encode with QF conditioning
        tokens = self.encoder(x)  # (B, T, D)
        
        if tokens.dim() != 3:
            raise RuntimeError("Unexpected encoder output dim. Expected (B, T, D).")

        # PHASE 3 Step 7: Apply QF modulation
        if self.use_qf and qf_vector is not None:
            # Compute scale (alpha) and bias (beta) from QF
            scale = self.qf_alpha(qf_vector)  # (B, embed_dim)
            bias = self.qf_beta(qf_vector)    # (B, embed_dim)
            
            # Apply affine transformation: x = scale * x + bias
            # Broadcast over token dimension
            scale = scale.unsqueeze(1)  # (B, 1, embed_dim)
            bias = bias.unsqueeze(1)    # (B, 1, embed_dim)
            
            tokens = scale * tokens + bias  # (B, T, embed_dim)

        # PHASE 4: Decode
        patch_tokens = tokens[:, 1:, :]  # Remove CLS token
        B, N, D = patch_tokens.shape
        patches_flat = self.decoder(patch_tokens.reshape(B * N, D))
        patches = patches_flat.reshape(B, N, self.patch_dim)
        reconstructed = self.unpatchify(patches)
        
        return reconstructed


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test model
    model = MFMViT_QF(img_size=224, qf_dim=64, use_qf=True).to(device)
    
    B = 4
    dummy_img = torch.randn(B, 3, 224, 224).to(device)
    dummy_qf = torch.randn(B, 64).to(device)
    
    out = model(dummy_img, dummy_qf)
    print("Output shape:", out.shape)
    print("QF modulation test passed!")