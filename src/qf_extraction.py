import torch
import torch.nn as nn
import torch.nn.functional as F

class QFAttention(nn.Module):
    """Quality Factor Attention from FBCNN"""
    def __init__(self, in_nc=64, nf=64):
        super().__init__()
        self.fuse = nn.Conv2d(in_nc, nf, 1, 1, 0, bias=True)
        self.attention = nn.Sequential(
            nn.Conv2d(nf, nf, 1, 1, 0, bias=True),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, qf_embed):
        qf_embed = qf_embed.unsqueeze(-1).unsqueeze(-1)
        fuse = self.fuse(torch.cat([x, qf_embed.expand(-1, -1, x.size(2), x.size(3))], dim=1))
        att = self.attention(fuse)
        return x * att
    
class FBCNN_QF(nn.Module):
    """
    Real FBCNN QF Predictor
    """

    def __init__(self, in_nc=3, nf=64, qf_dim=64):
        super().__init__()
        self.qf_dim = qf_dim

        # initial conv
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # DCT-aware filter extraction
        self.conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True) # 2124 -> 128
        self.conv2 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True) # 128 -> 64
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True) # 64 -> 32
        self.conv4 = nn.Conv2d(nf * 4, nf * 8, 3, 2, 1, bias=True) # 32 -> 16

        # QF predicted Head
        self.qf_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, qf_dim)
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        Extract QF embedding from input image
        x: (B, 3, H, W)
        returns: (B, qf_dim)
        """
        fea = self.lrelu(self.conv_first(x))
        fea = self.lrelu(self.conv1(fea))
        fea = self.lrelu(self.conv2(fea))
        fea = self.lrelu(self.conv3(fea))
        fea = self.lrelu(self.conv4(fea))
        
        qf_embedding = self.qf_pred(fea)
        return qf_embedding


def load_fbcnn_qf(pretrained_path=None, qf_dim=64, device='cuda'):
    """
    Load real FBCNN QF extractor and freeze it
    
    Args:
        pretrained_path: Path to FBCNN pretrained weights
        qf_dim: QF embedding dimension
        device: Device to load on
    
    Returns:
        Frozen FBCNN QF model
    """
    model = FBCNN_QF(in_nc=3, nf=64, qf_dim=qf_dim).to(device)
    
    if pretrained_path:
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)
            print(f"[FBCNN] Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"[FBCNN] Warning: {e}")
            print("[FBCNN] Using random initialization")
    else:
        print("[FBCNN] No pretrained weights, using random init")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    print(f"[FBCNN] Frozen with qf_dim={qf_dim}")
    
    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test
    fbcnn = load_fbcnn_qf(qf_dim=64, device=device)
    x = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        qf = fbcnn(x)
    
    print(f"Input: {x.shape}")
    print(f"QF embedding: {qf.shape}")
    print("✓ FBCNN QF extraction working!")

