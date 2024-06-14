from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout

from .vision_transformer import *
from .operators import *
from .promptEncoders import *

from pdb import set_trace as st

import timm



class XPrompt(nn.Module):
  
    def __init__(self,args):
        super(XPrompt, self).__init__()

        self.args = args
        self.vit = vit_small(patch_size=args.patch_size,pretrained_weights=args.pretrained_weights)
        self.classifier= nn.Linear(args.hidden_size, args.num_classes)

        self.proj = nn.Linear(self.args.sample_number,self.args.proj_hidd_dim)
        self.promptG=PromptEncoderDecoder(self.args.embed_num,self.args.hidden_size,self.args.prompt_s_num,self.args.prompt_m_num,self.args.proj_hidd_dim,self.args.prompt_token_length,nhead=args.nhead,layers=args.layers,activation=args.activation)
       
        self.prompt_dropout = Dropout(args.prompt_dropout)
        self.last_selfattn_weights=None

        # if project the prompt embeddings
        if self.args.prompt_project > -1:
            # only for prepend / add
            prompt_dim = self.args.prompt_project
            self.prompt_proj = nn.Linear(
                prompt_dim, args.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = args.hidden_size
            self.prompt_proj = nn.Identity()
       
    def incorporate_prompt(self,x,y,operator='cat'):
        if operator=='add':
            x = torch.cat((
                x[:, :1, :],
                x[:, 1:, :] + y,
            ), dim=1)
            # (batch_size, cls_token + n_patches, hidden_dim)
        elif operator=='cat':
            x = torch.cat((
                x[:, :1, :],
                y,
                x[:, 1:, :],
            ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x
    
    def forward_deep_prompt(self, x, prompt1=None, prompt2=None, prompt3=None):
        hidden_states = None
        deep_prompt_embeddings = prompt1
        num_layers = self.args.vit_num_layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.vit.blocks[i](x)
            else:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    deep_prompt_embeddings[:,i]))

                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    deep_prompt_emb,
                    hidden_states[:, (1+self.args.prompt_token_length):, :]
                ), dim=1)
                hidden_states, self.last_selfattn_weights = self.vit.blocks[i](hidden_states,return_attention=True)

        return hidden_states
    
    def forward(self, x, prompt1=None, prompt2=None, prompt3=None):
        nTiles,c,h,w=x.shape
        image = x[:,:3,:,:]
        emb_Q=self.proj(prompt1)
        self.deep_prompt_embeddings = self.promptG(emb_Q)

        tokenS_ = self.vit.prepare_tokens(image)
        tokenS = self.incorporate_prompt(tokenS_,self.deep_prompt_embeddings[:,0],self.args.prompt_combine_operator)

        if self.args.prompt_deep:
            tokenS=self.forward_deep_prompt(tokenS,self.deep_prompt_embeddings)
        else:
            tokenS = self.vit.encoder(tokenS)

        tokenS = self.vit.norm(tokenS)
        hidden = tokenS[:,0]
        logits = self.classifier(hidden)

        return logits
    




      



