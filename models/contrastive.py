"""
Adapted from https://github.com/omron-sinicx/medical-modality-dropout/blob/main/losses/contrastive_loss.py
"""
import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

        log_scale = torch.log(torch.tensor(10, dtype=torch.float32))
        self.log_scale = nn.Parameter(log_scale)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bias = nn.Parameter(torch.tensor(-10, dtype=torch.float32))

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            label=None
    )-> torch.Tensor:
        
        # query: (B, C), key: (B, D), labels: (B, )
        B, C = query.shape
        device = query.device
        dtype = query.dtype
        factory_kwargs = {'device': device, 'dtype': dtype}

        with torch.no_grad():
            if label is not None:
                label = label.view(-1, 1).contiguous()

            if label is None:
                positive_mask = torch.eye(B, **factory_kwargs)
            else:
                positive_mask = torch.eq(
                    label, label.T).to(dtype)

        # logits = torch.matmul(query, key.T)
        logits = torch.einsum('nc,kc->nk', query, key)
        logits = torch.mul(logits, torch.exp(self.log_scale.to(device))) + self.bias.to(device)
        loss = self.loss(logits, positive_mask)
        return loss.mean()
