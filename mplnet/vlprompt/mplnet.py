from typing import Type, Any, Dict, Optional, List, Tuple

import math
import numpy as np
import torch.nn.functional as F
import clip
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys

from clip import load as load_clip

import mplnet.lib as lib
import mplnet.vlprompt.tools as vlp_tools
from mplnet.vlprompt.prompted_transformers import PromptedTransformer
from mplnet.vlprompt.clip_local import ModifiedResNet, VisionTransformer, CLIP

NoneType = Type[None]
KwargType = Dict[str, Any]
CLIP_NAME = {"clip_vit_b32": "ViT-B/32", "clip_vit_b16": "ViT-B/16", "clip_resnet50": "RN50", "clip_resnet101": "RN101"}

class Linear(nn.Module):
    def __init__(self, in_dim: int, identity_init: bool = True) -> NoneType:
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        if identity_init:
            nn.init.zeros_(self.linear.weight)
            self.linear.weight.data += torch.eye(in_dim)
        else:
            nn.init.normal_(self.linear.weight, std=in_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class MPLNet(CLIP):
    TRAINABLE_PARAMS: List[str] = []

    def __init__(
        self,
        clip_name: str,
        use_local_features: bool = True,
        checkpointing_segments: int = 8,
        template: str = "A photo of a {}",
        learn_local_proj: bool = True,
        learn_global_prompts: bool = True,
        learn_regional_prompts: bool = True,
        learn_local_prompts: bool = True,
        class_names: List[str] = None,
        n_global_prompts: int = 1,
        n_regional_prompts: int = 1,
        n_local_prompts: int = 1,
        prompts_batch_size: int = math.inf,
        ood_method: str = "GL-MCM",
        ood_temp_scale: float = 1.0,
        parallel_text_encoder: bool = False,
        parallel_vision_encoder: bool = False,
        num_shots: int = None,
    ) -> NoneType:
        self.model_name = "mplnet_" + clip_name[5:]
        clip_model, _ = load_clip(CLIP_NAME[clip_name], device="cuda")

        clip_state_dict = clip_model.state_dict()
        clip_kwargs = lib.get_clip_hyperparams(clip_state_dict)
        clip_kwargs["return_local_features"] = use_local_features

        super().__init__(**clip_kwargs)
        self.clip_name = clip_name
        self.use_local_features = use_local_features
        self.learn_local_proj = learn_local_proj
        self.class_names = class_names
        self.template = template[:-1] if template[-1] == "." else template

        self.learn_local_prompts = learn_local_prompts
        self.learn_global_prompts = learn_global_prompts
        self.learn_regional_prompts = learn_regional_prompts
        self.n_global_prompts = n_global_prompts
        self.n_regional_prompts = n_regional_prompts
        self.n_local_prompts = n_local_prompts

        self.prompts_batch_size = min(prompts_batch_size, self.n_global_prompts)
        self.ood_method = ood_method
        self.ood_temp_scale = ood_temp_scale
        self.parallel_text_encoder = parallel_text_encoder
        self.parallel_vision_encoder = parallel_vision_encoder
        self.num_shots = num_shots


        if isinstance(clip_kwargs["vision_layers"], (tuple, list)):
            self.visual = ModifiedResNet(
                layers=clip_kwargs["vision_layers"],
                output_dim=clip_kwargs["embed_dim"],
                heads=clip_kwargs["vision_width"] * 32 // 64,
                input_resolution=clip_kwargs["image_resolution"],
                width=clip_kwargs["vision_width"],
            )
            vision_dim = clip_kwargs["embed_dim"]
        else:
            self.visual = VisionTransformer(
                input_resolution=clip_kwargs["image_resolution"],
                patch_size=clip_kwargs["vision_patch_size"],
                width=clip_kwargs["vision_width"],
                layers=clip_kwargs["vision_layers"],
                heads=clip_kwargs["vision_width"] // 64,
                output_dim=clip_kwargs["embed_dim"],
            )
            vision_dim = clip_kwargs["vision_width"]

        self.transformer = PromptedTransformer(
            width=clip_kwargs["transformer_width"],
            layers=clip_kwargs["transformer_layers"],
            heads=clip_kwargs["transformer_heads"],
            attn_mask=self.build_attention_mask(),
            segments=checkpointing_segments,
        )


        self.att_scale = nn.Parameter(torch.ones([]) * np.log(20))
        self.relu = nn.ReLU()
        self.local_proj = Linear(vision_dim)
        if self.learn_local_proj:
            self.TRAINABLE_PARAMS.append("local_proj")
        
        if self.learn_global_prompts or self.learn_local_prompts or self.n_global_prompts > 1 or self.n_local_prompts > 1:
            
            template = self.template.replace("{}", " ").replace("_", " ").strip()
            tokenized_template = clip.tokenize(template)
            self.template_init_tokens = int(tokenized_template.argmax(dim=-1)) - 1
            self.n_token_context = self.template_init_tokens

            if self.learn_global_prompts or self.n_global_prompts > 1:
                if self.learn_global_prompts:
                    self.TRAINABLE_PARAMS.append("global_prompts")
                self.global_prompts = nn.Parameter(
                    torch.empty(self.n_global_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )
                
            if self.learn_regional_prompts or self.n_regional_prompts > 1:
                if self.learn_regional_prompts:
                    self.TRAINABLE_PARAMS.append("regional_prompts")
                self.regional_prompts = nn.Parameter(
                    torch.empty(self.n_regional_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )

            if self.learn_local_prompts or self.n_local_prompts > 1:
                if self.learn_local_prompts:
                    self.TRAINABLE_PARAMS.append("local_prompts")
                self.local_prompts = nn.Parameter(
                    torch.empty(self.n_local_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )
        
        
        ############################## load clip #################################
        self.initialize_parameters()
        key_issue_clip = self.load_state_dict(clip_state_dict, strict=False)

        # remoteclip_state_dict = torch.load('/root/autodl-tmp/Checkpoint/RemoteCLIP-RN50.pt')
        # key_issue_clip = self.load_state_dict(remoteclip_state_dict, strict=False)
        ##########################################################################
        

        if len(key_issue_clip.missing_keys) > 0:
            lib.LOGGER.warning(f"Missing keys in CLIP: {key_issue_clip.missing_keys}")

        self.transformer = self.transformer if not self.parallel_text_encoder else vlp_tools.DataParallel(self.transformer)
        self.visual = self.visual if not self.parallel_vision_encoder else vlp_tools.DataParallel(self.visual)

    @property
    def num_devices(self) -> int:
        if not hasattr(self, "__device"):
            self.__device = torch.cuda.device_count()
        return self.__device

    def pad_if_necessary(self, x: Tensor) -> Tensor:
        if not self.parallel_text_encoder:
            return x, 0

        n = x.size(0)
        if n % self.num_devices == 0:
            return x, 0

        pad = self.num_devices - n % self.num_devices
        return torch.cat([x, torch.zeros(pad, *x.shape[1:], device=x.device)], dim=0), pad

    def unpad_if_necessary(self, x: Tensor, pad: int) -> Tensor:
        if pad == 0:
            return x

        return x[:-pad]

    def _default_encode_text(self, class_names: List[str]) -> Tensor:
        prompts = [self.template.format(name) for name in class_names]
        tokenized_text = clip.tokenize(prompts).cuda(non_blocking=True)
        text_features = super().encode_text(tokenized_text, batch_first=True)
        return text_features.unsqueeze(1)   # k1d

    def _encode_text(self, prefix: Tensor, prompt: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        x = torch.cat([prefix, prompt, suffix], dim=1)

        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND  # This is not needed as we are using batch_first=True
        x, padding = self.pad_if_necessary(x)
        x, *_ = self.transformer(x, batch_first=True)
        x = self.unpad_if_necessary(x, padding)
        # x = x.permute(1, 0, 2)  # LND -> NLD  # This is not needed as we are using batch_first=True
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_tokens + self.n_token_context] @ self.text_projection
        return x

    def _single_forward_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        n_prompts = prompts.size(0)  # m/n
        n_classes = prefix.size(0)   # k

        text_features = self._encode_text(
            prefix.repeat_interleave(n_prompts, dim=0),
            prompts.repeat(n_classes, 1, 1),
            suffix.repeat_interleave(n_prompts, dim=0),
            eot_tokens.repeat_interleave(n_prompts),
        )
        text_features = text_features.unflatten(0, (n_classes, n_prompts))
        return text_features

    def _loop_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        text_features = []
        for i in range(prompts.size(0)):
            x = self._encode_text(prefix, prompts[i : i + 1].expand(prefix.size(0), -1, -1), suffix, eot_tokens)
            text_features.append(x)

        return torch.stack(text_features, dim=1)

    def _most_efficient_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        if self.parallel_text_encoder:
            return self._single_forward_encode_text(prefix, prompts, suffix, eot_tokens)
        return self._loop_encode_text(prefix, prompts, suffix, eot_tokens)

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        if not self.learn_global_prompts and not self.learn_local_prompts:
            text_features = self._default_encode_text(class_names)
            return text_features, text_features

        tokenized_text = clip.tokenize(class_names).cuda(non_blocking=True)   # sos len name(1 or 2 ...) eos
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -(self.n_token_context), :]


        if self.learn_global_prompts or self.n_global_prompts > 1:
            global_prompts = self.global_prompts
            if self.prompts_batch_size < self.n_global_prompts and self.training:
                idx_select = torch.randperm(self.n_global_prompts)[: self.prompts_batch_size]  # we don't want to do this for local prompts
                global_prompts = self.global_prompts[idx_select]
            text_features = self._most_efficient_encode_text(prefix, global_prompts, suffix, eot_tokens)

        else:
            text_features = self._default_encode_text(class_names)

        if self.learn_regional_prompts or self.n_regional_prompts > 1:
            regional_text_features = self._most_efficient_encode_text(prefix, self.regional_prompts, suffix, eot_tokens)
        else:
            regional_text_features = text_features

        if self.learn_local_prompts or self.n_local_prompts > 1:
            local_text_features = self._most_efficient_encode_text(prefix, self.local_prompts, suffix, eot_tokens)
        else:
            local_text_features = text_features

        

        return text_features, regional_text_features, local_text_features

    def encode_image_and_proj(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        image_features, regional_features, local_features = self.encode_image(image)
        regional_features = self.local_proj(regional_features)
        local_features = self.local_proj(local_features)

        if hasattr(self.visual, "proj"):
            image_features = image_features @ self.visual.proj
            if self.use_local_features:
                local_features = local_features @ self.visual.proj
                regional_features = regional_features @ self.visual.proj

        return image_features, regional_features, local_features

    def forward(
        self,
        image: Tensor,
        class_names: Optional[List[str]] = None,
        text_features: Optional[Tensor] = None,
        regional_text_features: Optional[Tensor] = None,
        local_text_features:  Optional[Tensor] = None,
    ) -> Tensor:
        if class_names is not None:
            assert isinstance(class_names, list), "class_names must be a list of strings"
        if text_features is not None:
            assert isinstance(text_features, torch.Tensor), "text_features must be a Tensor"
        assert class_names is not None or text_features is not None, "Please provide either class_names or text_features"

        if text_features is None:
            assert regional_text_features is None, "regional_text_features should be None if text_features is None"
            assert local_text_features is None, "local_text_features should be None if text_features is None"
            text_features, regional_text_features, local_text_features = self.encode_text(class_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True) if self.learn_local_prompts else text_features
            regional_text_features = regional_text_features / regional_text_features.norm(dim=-1, keepdim=True)
        image_features, regional_features, local_features = self.encode_image_and_proj(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        global_logits = torch.einsum("bd,kmd-> bkm", image_features, text_features)

        if self.use_local_features:
            regional_features = regional_features / regional_features.norm(dim=-1, keepdim=True)
            regional_logits = torch.einsum("bpd,knd-> bpkn", regional_features, regional_text_features)
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            local_logits = torch.einsum("bpd,knd-> bpkn", local_features, local_text_features)
            
            regional_attention = torch.einsum("bd,bpd-> bp", image_features, regional_features)
            regional_attention = self.relu(regional_attention)
            regional_attention = F.softmax(regional_attention * math.exp(self.att_scale), dim=1)

            local_attention = torch.einsum("bd,bpd-> bp", image_features, local_features)
            local_attention = self.relu(local_attention)
            local_attention = F.softmax(local_attention * math.exp(self.att_scale), dim=1)
        else:
            regional_logits = None
            local_logits = None
            regional_attention = None
            local_attention = None
        
        global_prompts_features, regional_prompts_features, local_prompts_features = self.prompt_features()

        return global_logits, regional_logits, local_logits, global_prompts_features, regional_prompts_features, local_prompts_features, regional_attention, local_attention

    def _prompt_features(self, promtps: Tensor) -> Tensor:
        tokenized_text = clip.tokenize("").cuda(non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]

        text_features = self._most_efficient_encode_text(prefix, promtps, suffix, eot_tokens)
        return text_features

    def prompt_features(
        self,
    ) -> Tensor:
        global_prompts_features = regional_prompts_features = local_prompts_features = None
        if self.learn_global_prompts:
            global_prompts_features = self._prompt_features(self.global_prompts)

        if self.learn_regional_prompts:
            regional_prompts_features = self._prompt_features(self.regional_prompts)

        if self.learn_local_prompts:
            local_prompts_features = self._prompt_features(self.local_prompts)

        return global_prompts_features, regional_prompts_features, local_prompts_features

    def template_features(self) -> Tensor:
        template_features =  self._default_encode_text(self.class_names) # k1d
        template_features = template_features / template_features.norm(dim=-1, keepdim=True)
        return template_features

    @property
    def device(self) -> torch.device:
        return self.text_projection.device

    def freeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = False

        for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            module.eval()
            module.train = lambda _: None

    def unfreeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = True

        for _ in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            print("Warning this module has Batchnorm that cannot be unfrozen.")
            break

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.state_dict().items() if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])}

    def load_trainable_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> _IncompatibleKeys:
        keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in keys.missing_keys if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])]
        if strict:
            error_msgs: List[str] = []
            if len(keys.unexpected_keys) > 0:
                error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in keys.unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys)))

            if len(error_msgs) > 0:
                raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs)))

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=keys.unexpected_keys)

    @torch.no_grad()
    def initialize_prompt(self) -> NoneType:
        if not self.learn_global_prompts and not self.learn_local_prompts:
            return

        template = self.template.replace("{}", " ").replace("_", " ").strip()
        tokenized_template = clip.tokenize(template)
        embedding = self.token_embedding(tokenized_template).type(self.dtype)
        global_prompts_init = embedding[:, 1 : 1 + self.template_init_tokens, :]

        if self.learn_global_prompts:
            self.global_prompts.data[:, : self.template_init_tokens].copy_(global_prompts_init.clone().expand(self.n_global_prompts, -1, -1))

        if self.learn_regional_prompts:
            self.regional_prompts.data[:, : self.template_init_tokens].copy_(global_prompts_init.clone().expand(self.n_local_prompts, -1, -1))
 
        if self.learn_local_prompts:
            self.local_prompts.data[:, : self.template_init_tokens].copy_(global_prompts_init.clone().expand(self.n_local_prompts, -1, -1))

    def compute_GRL_scores(
        self,
        global_logits: Tensor,
        regional_logits: Optional[Tensor],
        local_logits: Optional[Tensor],
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        scores = -np.max(global_probs, axis=-1)
        if regional_logits is not None:
            regional_probs = torch.softmax(regional_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            regional_score = -np.max(regional_probs, axis=(1, 2))
            scores += regional_score

        if local_logits is not None:
            local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            local_score = -np.max(local_probs, axis=(1, 2))
            scores += local_score

        return scores

    def compute_L_mcm_scores(
        self,
        local_logits: Tensor,
    ) -> NoneType:
        assert local_logits is not None
        local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
        local_score = -np.max(local_probs, axis=(1, 2))
        return local_score
    
    def compute_R_mcm_scores(
        self,
        regional_logits: Tensor,
    ) -> NoneType:
        assert regional_logits is not None
        regional_probs = torch.softmax(regional_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
        regional_score = -np.max(regional_probs, axis=(1, 2))
        return regional_score

    def compute_mcm_scores(
        self,
        global_logits: Tensor,
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        global_score = -np.max(global_probs, axis=-1)
        return global_score

    def compute_scores(
        self,
        global_logits: Tensor,
        regional_logits: Optional[Tensor],
        local_logits: Optional[Tensor],
        ood_method: Optional[str] = None,
    ) -> NoneType:
        if ood_method is None:
            ood_method = self.ood_method

        if ood_method == "GRL-MCM":
            return self.compute_GRL_scores(global_logits, regional_logits, local_logits)
        elif ood_method == "MCM":
            return self.compute_mcm_scores(global_logits)
        elif ood_method == "L-MCM":
            return self.compute_L_mcm_scores(local_logits)
        elif ood_method == "R-MCM":
            return self.compute_R_mcm_scores(regional_logits)
        else:
            raise ValueError(f"Method {self.ood_method} not implemented")


    @torch.no_grad()
    def create_prediction_scores(
        self,
        global_logits: Tensor,
        regional_logits: Optional[Tensor],
        local_logits: Optional[Tensor],
        regional_attention: Optional[Tensor],
        local_attention: Optional[Tensor],
    ) -> Tensor:
        
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None and regional_logits is None:
            regional_probs = local_probs = None
            multi_probs = global_probs

        else:
            ''' 12 img-img attention '''
            regional_attention = regional_attention.unsqueeze(-1).unsqueeze(-1)
            regional_logits = regional_logits * regional_attention
            regional_logits = regional_logits.sum(dim=1)
            regional_logits = regional_logits.mean(dim=-1)
            regional_probs = torch.softmax(logit_scale * regional_logits, dim=-1)

            ''' 12 img-txt attention '''
            # regional_att = F.softmax(regional_logits * self.att_scale.exp(), dim=1)
            # regional_logits = torch.einsum("bpkm,bpkm -> bkm", regional_att, regional_logits)
            # regional_logits = regional_logits.mean(dim=-1)
            # regional_probs = torch.softmax(logit_scale * regional_logits, dim=-1)
            
            ''' without attention'''
            # regional_logits = regional_logits.mean(dim=1)
            # regional_logits = regional_logits.mean(dim=-1)
            # regional_probs = torch.softmax(logit_scale * regional_logits, dim=-1)

            # ----------------------------------------------------------------------------------------------------------

            ''' 11 img-img attention '''
            # local_attention = local_attention.unsqueeze(-1).unsqueeze(-1)
            # local_logits = local_logits * local_attention
            # local_logits = local_logits.sum(dim=1)     #bkm
            # local_logits = local_logits.mean(dim=-1)
            # local_probs = torch.softmax(logit_scale * local_logits, dim=-1)

            ''' 11 img-txt attention '''
            local_att =  F.softmax(local_logits * self.att_scale.exp(), dim=1)
            local_logits = local_logits *  local_att
            local_logits = local_logits.sum(dim=1)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)

            '''without attention'''
            # local_logits = local_logits.mean(dim=1)
            # local_logits = local_logits.mean(dim=-1)
            # local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
           
       
        multi_logits = (regional_logits + global_logits + local_logits) / 3
        l12_logits = (regional_logits + global_logits ) / 2
        l13_logits = (global_logits + local_logits) / 2
        l23_logits = (regional_logits + local_logits) / 2

        l12 = torch.softmax(logit_scale * l12_logits, dim=-1)
        l13 = torch.softmax(logit_scale * l13_logits, dim=-1)
        l23 = torch.softmax(logit_scale * l23_logits, dim=-1)
        multi_probs = torch.softmax(logit_scale * multi_logits, dim=-1)

        return multi_probs, global_probs, regional_probs, local_probs, l12, l13, l23
