import torch
import sys
sys.path.insert(0, '../')
from model import MFMViT

# Create model
model = MFMViT(img_size=224, patch_size=16, embed_dim=768)
model.eval()

# Create fake data
fake_input = torch.randn(2, 3, 224, 224)

# Forward pass
with torch.no_grad():
    output = model(fake_input)

print(f"Input shape: {fake_input.shape}")
print(f"Output shape: {output.shape}")
