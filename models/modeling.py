# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  #hidden size is the embedding size
        self.all_head_size = self.num_attention_heads * self.attention_head_size       #head size is the embedding size divided by the num of head, it is also the output embedding size by one attention head before concatenation, remember eight 64-element embeddings concatenate to one 512-element embedding
                                                                                       #normally all_head_size will equal embedding size, you can think of them as the same things(when it can be divided with 0 remainder)
        self.query = Linear(config.hidden_size, self.all_head_size)                    #think of these three as the query matrix, key matrix and value matrix
        self.key = Linear(config.hidden_size, self.all_head_size)                      #if you multiply you embedding with the three matrices, you get the query, key and value
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)                      #this is the W0 matrix that you multiply with the concatenated matrix to get the final output
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):                               #x is of the dim [B, N+1, embed size], B for batch size, N for patch size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)   #x.size()[:-1] gives you [B,N+1]. NOTICE that plus here is concatenation!!! new_x_shape is of dim [B, N+1, num_head, head_size] and num_head*head_size=embed_size
        x = x.view(*new_x_shape)                                     # star here unpack [B, N+1, num_head, head_size] to four elements, this is originally a torch.Size object
        return x.permute(0, 2, 1, 3)                                 #x.permute is of dim [B, num_head, N+1, head_size]

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)        #hidden state is of [B, N+1, embed size], the output after the linear layer remains the same (B is num per batch, N is patch size)
        mixed_key_layer = self.key(hidden_states)            #output is of dim [B, N+1, embed size]
        mixed_value_layer = self.value(hidden_states)        #output is of dim [B, N+1, embed size]

        query_layer = self.transpose_for_scores(mixed_query_layer)   #query_layer is of dim [B, num_head, N+1, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)       #key_layer is of dim [B, num_head, N+1, head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)   #value_layer is of dim [B, num_head, N+1, head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  #key_layer.transpose(-1, -2) is of dim [B, num_head, head_size, N+1], attention_scores is of dim [B, num_head, N+1, N+1]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  #divided by the sqrt of head_size to make gradient more stable
        attention_probs = self.softmax(attention_scores)                           #softmax gives weight of addition
        weights = attention_probs if self.vis else None                            #this is mainly for visualization, not for computation use.
        attention_probs = self.attn_dropout(attention_probs)                       #dropout prevents overfit, final dimension of attention_probs is [B, num_head, N+1, N+1]

        context_layer = torch.matmul(attention_probs, value_layer)                  #add value key according to weight, context_layer is of dim [B, num_head, N+1, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()              #permutation gives dimension of [B, N+1, num_head, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #new_context_layer_shape is [B, N+1, embed_size], type is torch.Size, not list
        context_layer = context_layer.view(*new_context_layer_shape)                #context_layer is of dimension [B, N+1, embed_size], you can actually do context_layer.flatten(2) to merge the last two dimensionw without making it complicated
        attention_output = self.out(context_layer)                                  #attention_out is still of dim [B, N+1, embed_size]
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):                                                         #this is the feed forward network, it is straight forward, needless of much explanation
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]                                         #the paper use gelu activation function
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()                                                 #initialize the weight

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  #n_patches equals (H/P)*(W/P), H and W is height and width of an image, P is the length of o patch
            self.hybrid = False

        if self.hybrid:                                                            #if hybrid, you collect patches at the output of CNN backbone
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,                    #actually you can also use matrix multiplication instead of conv2d to extract embedding
                                       out_channels=config.hidden_size,            #the final result will be the same if you use matrix multiplication
                                       kernel_size=patch_size,                     #here hidden_size is just embedding size
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))    #randomly initialize positional embeddings, they can be learned through back prop
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))                        #randomly initialize cls_token, they will be learned as well.
                                                                                                    #cls_token is to be feeded to the final MLP for classification, only cls_token will be used for classification

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):                               #x is of dim [B,3,H,W]
        B = x.shape[0]                                  #this is the batch num
        cls_tokens = self.cls_token.expand(B, -1, -1)   #cls_token after expand is of dim [B,1,embed_size]

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)                    #conv2d gives dim [B, embed_size, H/P, W/P], P denotes patch size
        x = x.flatten(2)                                #flatten gives dim [B, emb_size, n_patches], n_patches denotes HW/P^2
        x = x.transpose(-1, -2)                         #transpose gives dim [B, n_patches, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)           #cat gives dim [B, n_patches+1, embed_size]

        embeddings = x + self.position_embeddings       #dim [B, n_patches+1, embed_size]
        embeddings = self.dropout(embeddings)           #dim [B, n_patches+1, embed_size]
        return embeddings


class Block(nn.Module):                                                #one block consists of self-attention and feed forward
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)  #layer norm is to normalize across different channels but within the same batch, unlike batch norm
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)        #batch norm is to normalize across different batch but within the same channel
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x) #notice that here it first does layer norm then does self attention, but in the paper, layer norm is applied after the skip connection
        x, weights = self.attn(x)
        x = x + h                  #this is the skip connection for self attention, which is the idea from ResNet

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h                 #this is the skip connection for the MLP layer, which is the idea from ResNet
        return x, weights

    def load_from(self, weights, n_block):            #load pretrained parameters
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):                                         #encoder consists of several blocks, in the paper, it consists of 8 blocks(each block with one self-attention and one MLP)
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)                            #actually you cannot create the layer outside the for loop, that way all blocks will share the same parameters. We want all blocks to be independent
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)  #define forward pass
            if self.vis:
                attn_weights.append(weights)                     #get the weight for each self-attention layer for visualization
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):                                  #transformer is simply the encoder network plus the embedding network
    def __init__(self, config, img_size, vis):                 #by now you should understand the entrie embeddin network is learned, including the class token and the positional embedding
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):                       #VisionTransformer is just the transformer plus the classification head. It only uses the class token for classification, not all N+1 token
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
