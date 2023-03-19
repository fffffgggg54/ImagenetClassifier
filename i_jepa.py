# similar to actual i-jepa from https://arxiv.org/abs/2301.08243
# but not faithful to paper, too lazy to do it properly
# sacrifices performance from vit-specific optimizations for generality

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block
from copy import deepcopy


# some hints from https://github.com/gaasher/I-JEPA/blob/main/model.py
class Predictor(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        depth=6,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_norm=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path_rate=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.proj = nn.Linear(in_dim, out_dim)
        
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=out_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
            
        self.norm = norm_layer(out_dim)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x
        

#works with 2d shape
#some performance to be had
def generate_random_mask(shape, target_size):
    mask = torch.randn(shape).flatten()
    _, idx = torch.topk(mask, target_size)
    mask = torch.zeros_like(mask)
    mask[idx] = 1
    mask = mask.reshape(shape)
    return mask
    

def get_masks( 
    target_shape,
    num_targets_per_sample=4,
    target_size=6,
    scale_range=(0.15,0.2),
    aspect_ratio_range=(3./4., 2./1.),
    out_size=None,
):
    batch_size = target_shape[0]
    if out_size:
        assert isinstance(out_size, (list, tuple))
    elif target_shape[1]**0.5%1==0:
        out_size = (int(target_shape[1]**0.5), int(target_shape[1]**0.5))
    else:
        raise Exception('masker got invalid input, go debug')
        
    if target_size:
        target_masks = []
        for sample in range(batch_size):
            sample_masks=[]
            # get some amount of masks for each sample
            for mask in range(num_targets_per_sample):
                sample_masks.append(generate_random_mask(out_size, target_size))
            target_masks.append(torch.stack(sample_masks))
        # target masks ends as [B, num_targets_per_sample, *out_size]
        target_masks = torch.stack(target_masks)
        context_mask = (target_masks.sum(dim=1) >= 1).logical_not().reshape(batch_size, 1, out_size[0], out_size[1])
        target_masks = target_masks.flatten(2)
        
    else:
        raise Exception('rectangular targets not supported rn D:')
        
    
    
    return context_mask, target_masks
    
    
class I_JEPA(nn.Module):
    def __init__(
        self,
        backbone,
        predictor_dim = 384,
        predictor_num_heads = 6,
        num_targets_per_sample=4,
        target_size=6,
    ):
        super().__init__()
        self.context_encoder = backbone
        self.target_encoder = deepcopy(backbone)
        self.predictor_dim = predictor_dim

        self.predictor = Predictor(backbone.num_features, predictor_dim, predictor_num_heads)
        self.encoder_dim = backbone.num_features
        
        self.num_targets_per_sample = num_targets_per_sample
        self.target_size = target_size
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.encoder_dim))
        self.mask_pe = None
        
    def forward(self, x):
        in_shape = x.shape
        target_unmasked = self.target_encoder(x)
        # only bnc for now
        B, N, C = target_unmasked.shape
        context_mask, target_masks = get_masks(
            target_unmasked.shape, 
            num_targets_per_sample=self.num_targets_per_sample,
            target_size = self.target_size
        )
        
        target_unmasked = target_unmasked.reshape(B, 1, N, C)
        target_masks = target_masks.reshape(B, self.num_targets_per_sample, N, 1).to(x.device)
        targets = target_unmasked * target_masks
        
        context_mask = F.interpolate(context_mask.float().to(x.device), (in_shape[-2], in_shape[-1]))

        contexts = []
        for target_mask in target_masks.transpose(0,1):
            mask_shape = target_mask.shape
            new_mask = target_mask.reshape(mask_shape[0], mask_shape[1], 1)
            
            current_context = x * context_mask
            context = self.context_encoder(current_context)
            if self.mask_pe == None:
                self.mask_pe = nn.Parameter((torch.randn(1, mask_shape[1], self.encoder_dim).to(x.device) * .02))
            context = context + new_mask * (self.mask_token + self.mask_pe)
            context = new_mask * self.predictor(context)
            contexts.append(context)
        
        contexts = torch.stack(contexts).transpose(0,1)
        targets = self.predictor.proj(targets)
            
        return targets, contexts
            
            
