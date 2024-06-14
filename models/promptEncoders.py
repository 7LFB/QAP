

import os
import logging
import time
from pathlib import Path

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from pdb import set_trace as st

from .operators import *

def init_implicit_prompts(length,dim,patch_size=16):
    prompt_embeddings = nn.Parameter(torch.zeros(
                1, length, dim))
        # xavier_uniform initialization
    val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size), 1) + dim))
    nn.init.uniform_(prompt_embeddings.data, -val, val)

    return prompt_embeddings


class ImgPromptEncoder(nn.Module):
    def __init__(self,img_c,img_h,img_w,hidden_dim,scale=4):
        super(ImgPromptEncoder,self).__int__()

        self.resize=transforms.Resize((int(img_h/scale),int(img_h/scale)))

        self.encoder=nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=2, stride=2),
            LayerNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, hidden_dim, kernel_size=1),
        )

    def forward(self,x):
        scale_x = self.resize(x)
        y=self.encoder(scale_x).flatten(2).transpose(1,2)
    
        return y
    



class DensityEstimationLayer(nn.Module):
    def __init__(self,in_channels,out_dim,out_h,out_w):
        super(DensityEstimationLayer, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AdaptiveAvgPool2d((out_h,out_w))
        
        # Define the fully connected layers
        self.fc1 = nn.Conv2d(64, 512,1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(512, out_dim,1)
        
    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Pass the output through the fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        x = x.reshape(x.shape[0],x.shape[1],-1).transpose(1,2)
        # Return the estimated density
        return x
    


class PromptG(nn.Module):
    def __init__(self,embed_num, embed_dim, proj_input_dim, proj_hidd_dim, promptL, promptLayers=12):
        super(PromptG, self).__init__()

        self.embed_num = embed_num
        self.embed_dim = embed_dim

        # projector to generate cooeficients
        self.proj_input_dim = proj_input_dim
        self.proj_hidd_dim = proj_hidd_dim
        self.prompt_len = promptL
        self.prompt_layers = promptLayers

        rawEmbed = torch.empty(
            self.embed_num,
            self.embed_dim
        )
        torch.nn.init.normal_(rawEmbed, std=0.02)
        self.attributes = torch.nn.Parameter(rawEmbed)


        self.lin1 = nn.Linear(self.proj_input_dim, self.proj_hidd_dim)
        self.lin2 = nn.Linear(self.proj_hidd_dim, self.prompt_len*self.prompt_layers*self.embed_num)
        self.act1 = nn.GELU()
        self.act2 = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.act1(self.lin1(x))
        coef = self.act2(self.lin2(x).view(-1,self.prompt_len*self.prompt_layers,self.embed_num))
        prompts = torch.einsum('bom,mv->bov',[coef,self.attributes])
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len,self.embed_dim)
        return prompts
    

class PromptSG(nn.Module):
    def __init__(self,embed_num, embed_dim, proj_input_dim, proj_hidd_dim, promptL, promptLayers=12):
        super(PromptSG, self).__init__()

        self.embed_num = embed_num
        self.embed_dim = embed_dim

        # projector to generate cooeficients
        self.proj_input_dim = proj_input_dim
        self.proj_hidd_dim = proj_hidd_dim
        self.prompt_len = promptL
        self.prompt_layers = promptLayers

        rawEmbed = torch.empty(
            self.embed_num,
            self.embed_dim
        )
        torch.nn.init.normal_(rawEmbed, std=0.02)
        self.attributes = torch.nn.Parameter(rawEmbed)

        self.gate_logit=nn.Parameter((-torch.ones(proj_input_dim) * 0.5))

        self.lin1 = nn.Linear(self.proj_input_dim, self.proj_hidd_dim)
        self.lin2 = nn.Linear(self.proj_hidd_dim, self.prompt_len*self.prompt_layers*self.embed_num)
        self.act1 = nn.GELU()
        self.act2 = nn.Softmax(dim=-1)
        
    def forward(self, x):

        x = x*self.gate_logit.sigmoid()
        x = self.act1(self.lin1(x))
        coef = self.act2(self.lin2(x).view(-1,self.prompt_len*self.prompt_layers,self.embed_num))
        prompts = torch.einsum('bom,mv->bov',[coef,self.attributes])
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len,self.embed_dim)
        return prompts
    


class PromptProj(nn.Module):
    def __init__(self,embed_dim, proj_input_dim, proj_hidd_dim, promptL, promptLayers=12):
        super(PromptProj, self).__init__()

        # projector to generate cooeficients
        self.proj_input_dim = proj_input_dim
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len = promptL
        self.prompt_layers = promptLayers

     
        self.lin1 = nn.Linear(self.proj_input_dim, self.proj_hidd_dim)
        self.lin2 = nn.Linear(self.proj_hidd_dim, self.prompt_len*self.prompt_layers*self.proj_output_dim)
        self.act1 = nn.GELU()
        self.act2 = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.act1(self.lin1(x))
        prompts = self.lin2(x).view(-1,self.prompt_layers,self.prompt_len,self.proj_output_dim)
        return prompts
    

class PromptSAT(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(PromptSAT, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.attributes = nn.Parameter(torch.empty(
            self.embed_num,
            self.embed_dim
        ))

        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m+1,
            self.proj_hidd_dim
        ))

        torch.nn.init.normal_(self.attributes, std=0.02)
        torch.nn.init.normal_(self.cls_token,std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.prompt_len_per_layer*self.prompt_layers*self.embed_num)

        
    def forward(self, x):

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.props_pe
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        coef = F.softmax(self.proj(x[:,0,:]),dim=-1).view(-1,self.prompt_len_per_layer*self.prompt_layers,self.embed_num)#-> B x proj_output_dim x (promptL x prompt_layers)
        prompts = torch.einsum('bom,mv->bov',[coef,self.attributes])
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts
        
    

class PromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(PromptEncoderDecoder, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))
        self.props_pe = nn.Parameter(torch.empty(1,
            self.prompt_num_s + self.prompt_num_m,
            self.proj_hidd_dim
        ))
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)
        torch.nn.init.normal_(self.props_pe,std=0.02)

        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

        
    def forward(self, x):

        x = x + self.props_pe
        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        prompts = self.decoder(self.prompt_embeddings.expand(x.shape[0],-1,-1),x)
        prompts = self.proj(prompts)
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts
    


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    


class ImagePromptEncoderDecoder(nn.Module):
    def __init__(self,embed_num, embed_dim, prompt_num_s, prompt_num_m, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(ImagePromptEncoderDecoder, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.prompt_num_s = prompt_num_s
        self.prompt_num_m = prompt_num_m
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))
   
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)


        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

        self.patch_embed = PatchEmbed(img_size=224, patch_size=16,in_chans=6,embed_dim=self.proj_hidd_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.proj_hidd_dim))
        self.pos_drop = nn.Dropout(p=0.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

        
    def forward(self, x):
        tokenS = self.prepare_tokens(x)
        x = self.encoder(tokenS) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        prompts = self.decoder(self.prompt_embeddings.expand(x.shape[0],-1,-1),x)
        prompts = self.proj(prompts)
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts



class GeneralPromptEncoderDecoder(nn.Module):
    def __init__(self, embed_dim, proj_hidd_dim, promptL, promptLayers=12,nhead=8,layers=1,activation='relu'):
        super(GeneralPromptEncoderDecoder, self).__init__()

        # projector to generate cooeficients
        self.layers=layers
        self.embed_dim = embed_dim
        self.proj_hidd_dim = proj_hidd_dim
        self.proj_output_dim = embed_dim
        self.prompt_len_per_layer = promptL
        self.prompt_layers = promptLayers

        self.prompt_embeddings = nn.Parameter(torch.empty(1,
            self.prompt_len_per_layer*self.prompt_layers,
            self.proj_hidd_dim
        ))
        
        torch.nn.init.normal_(self.prompt_embeddings, std=0.02)


        # Attention: batch_first=True
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.proj_hidd_dim, nhead=nhead,batch_first=True,activation=activation),num_layers=layers)
        self.proj = nn.Linear(self.proj_hidd_dim, self.embed_dim)

        
    def forward(self, x):

        x = self.encoder(x) #-> B x prompt_s_num+prompt_m_num+1 x proj_hidd_dim
        # fetch cls token for prediction
        prompts = self.decoder(self.prompt_embeddings.expand(x.shape[0],-1,-1),x)
        prompts = self.proj(prompts)
        if self.prompt_layers>1:
            return prompts.view(-1,self.prompt_layers,self.prompt_len_per_layer,self.embed_dim)
        return prompts
    


class ImageEmbedding(nn.Module):
    def __init__(self,proj_hidd_dim,img_size=224, patch_size=16,in_chans=6,):
        super(ImageEmbedding, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, proj_hidd_dim))

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,in_chans=in_chans,embed_dim=proj_hidd_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, proj_hidd_dim))
        self.pos_drop = nn.Dropout(p=0.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)
    
    def forward(self,x):

        tokens = self.prepare_tokens(x)

        return tokens

    

        