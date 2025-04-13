from typing import Type, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
from mplnet.vlprompt.tools.trcloss import TRCLoss

NoneType = Type[None]

class TotalLoss(_WeightedLoss):

    def __init__(
        self,
        global_dropout_p: float = 0.75,
    ) -> NoneType:
        super().__init__()
        self.global_dropout_p = global_dropout_p

    def forward(
        self,
        global_logits: Tensor,
        regional_logits: Tensor,
        local_logits: Tensor,
        targets: Tensor,
        logit_scale: float,
        att_scale: float = 20.0,
        rprompts_features : Optional[Tensor] = None,
        gprompts_features : Optional[Tensor] = None,
        lprompts_features : Optional[Tensor] = None,
        regional_attention: Optional[Tensor] = None,
        local_attention: Optional[Tensor] = None,
        num_shots : Optional[int] = 1,

    ) -> Tensor:
        """
        global_logits is a Tensor of shape (b, k, 1) or (b, k, n)
        local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
        
        lprompts_features is a Tensor of shape (1, m, d)
        local_attention is a Tensor of shape (b, p)
        """
        global_loss = regional_loss = local_loss = 0.
        
        if regional_logits is not None:          

            ''' 12 img-img attention '''
            regional_attention = regional_attention.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, regional_logits.size(2), regional_logits.size(3))
            regional_logits = regional_logits * regional_attention
            regional_logits = regional_logits.sum(dim=1)

            ''' 12 img-txt attention '''
            # regional_att = F.softmax(regional_logits * att_scale, dim=1)
            # regional_logits = regional_att * regional_logits
            # regional_logits = regional_logits.sum(dim=1)
            
            ''' without attention'''
            # regional_logits = regional_logits.mean(dim=1)

            ''' dropout'''
            keep_number1 = 1
            index = torch.randint(regional_logits.size(-1), (regional_logits.size(0), 1, keep_number1), device=regional_logits.device).expand(-1, regional_logits.size(1), -1)
            regional_logits = regional_logits.gather(-1, index)
            regional_loss = F.cross_entropy(logit_scale * regional_logits, targets.unsqueeze(-1).expand(-1, regional_logits.size(-1)))

        if local_logits is not None:

            ''' 11 img-img attention '''
            # local_attention = local_attention.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, local_logits.size(2), local_logits.size(3))
            # local_logits = local_logits * local_attention
            # local_logits = local_logits.sum(dim=1)

            ''' 11 img-txt attention '''
            local_att =  F.softmax(local_logits * att_scale, dim=1)
            local_logits = local_att * local_logits
            local_logits = local_logits.sum(dim=1)

            ''' without attention'''
            # local_logits = local_logits.mean(dim=1)
            
            ''' dropout'''
            keep_number2 = 1
            index = torch.randint(local_logits.size(-1), (local_logits.size(0), 1, keep_number2), device=local_logits.device).expand(-1, local_logits.size(1), -1)
            local_logits = local_logits.gather(-1, index)
            local_loss = F.cross_entropy(logit_scale * local_logits, targets.unsqueeze(-1).expand(-1, local_logits.size(-1)))


        if global_logits is not None: 
            keep_number = max(global_logits.size(-1) - int(self.global_dropout_p * global_logits.size(-1)), 1)
            index = torch.randint(global_logits.size(-1), (global_logits.size(0), 1, keep_number), device=global_logits.device).expand(-1, global_logits.size(1), -1)
            global_logits = global_logits.gather(-1, index).mean(-1)
        
            if global_logits.ndim == 2:
                global_loss = F.cross_entropy(logit_scale * global_logits, targets)
            elif global_logits.ndim == 3:
                global_loss = F.cross_entropy(logit_scale * global_logits, targets.unsqueeze(-1).expand(-1, global_logits.size(-1)))
            else:
                raise ValueError(f"Global logits must have 2 or 3 dimensions, but got {global_logits.ndim}.")

        gprompts_features = gprompts_features.squeeze(0) # n, d
        rprompts_features = rprompts_features.squeeze(0) # m, d
        lprompts_features = lprompts_features.squeeze(0) # m, d

        if lprompts_features is not None and gprompts_features is not None:
            trc_loss = (TRCLoss(gprompts_features, rprompts_features)  + TRCLoss(gprompts_features, lprompts_features)) / (2 * num_shots)
            
 

        return global_loss  + regional_loss + local_loss  + trc_loss * 10
        







