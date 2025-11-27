import torch
import torch.nn as nn


class MFMLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred_amp, target_amp, mask):
        """
        pred_amp:    predicted amplitude   (B, C, H, W)
        target_amp:  ground truth amplitude (B, C, H, W)
        mask:        frequency mask (1 = visible, 0 = masked)
        """
        # ----------------------------
        # 1. use only masked regions
        # ----------------------------
        masked_region = (1 - mask)  # 1 = masked frequencies

        # ----------------------------
        # 2. log amplitude (stabilizes gradients)
        # ----------------------------
        pred = torch.log1p(pred_amp + self.eps)
        targ = torch.log1p(target_amp + self.eps)

        # ----------------------------
        # 3. instance normalization
        # per-image mean/std normalization
        # ----------------------------
        def inst_norm(x):
            mean = x.mean(dim=(1,2,3), keepdim=True)
            std = x.std(dim=(1,2,3), keepdim=True)
            return (x - mean) / (std + self.eps)

        pred = inst_norm(pred)
        targ = inst_norm(targ)

        # ----------------------------
        # 4. compute masked reconstruction loss
        # ----------------------------
        pred_m = pred * masked_region
        targ_m = targ * masked_region

        loss_mse = self.mse(pred_m, targ_m)
        loss_l1  = self.l1(pred_m, targ_m)

        # ----------------------------
        # 5. final weighted loss
        # ----------------------------
        loss = loss_mse + 0.1 * loss_l1

        return loss
