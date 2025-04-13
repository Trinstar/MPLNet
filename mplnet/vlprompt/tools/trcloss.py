import torch
import torch.nn.functional as F
from torch import Tensor

def TRCLoss(gprompts_features: Tensor,
                      lprompts_features: Tensor,
                      ) -> float:
    
    mean_gprompts_features = torch.mean(gprompts_features, dim=0).clone().detach()
    
    norm_lprompts_features = lprompts_features / lprompts_features.norm(dim=-1, keepdim=True)
    mean_gprompts_features /= mean_gprompts_features.norm(dim=-1, keepdim=True)

    loss = torch.sum(1 - torch.einsum("md,d -> m", norm_lprompts_features, mean_gprompts_features))
    return loss
