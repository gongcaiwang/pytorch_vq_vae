from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/rwang/bravo/avatar/code/pytorch_vq_vae/')
import lreq as ln
from vqvae_layer import *

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs, offstage=None):
        # convert inputs from BCW -> BWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # if offstage is not None:
        #     offstage = offstage[...,:inputs.shape[-1]].view(-1,1)
        #     distances_0 = (torch.sum(flat_input**2, dim=1, keepdim=True) 
        #             + torch.sum(self._embedding.weight[0:1]**2, dim=1)
        #             - 2 * torch.matmul(flat_input, self._embedding.weight[0:1].t()))
        #     distances_0 = distances_0[offstage.squeeze()==1]

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if offstage is not None:
            offstage = offstage[...,:inputs.shape[1]].view(-1)
            encoding_indices[offstage.squeeze()==1] = torch.zeros_like(encoding_indices[offstage.squeeze()==1]) # force 0 token for silient
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # if offstage is not None:
        #     # convert quantized from BWC -> BCW
        #     return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings, distances_0
        # else:
        #     return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings, None
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings, distances
    

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs, offstage=None):
        # convert inputs from BCW -> BWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # if offstage is not None:
        #     offstage = offstage[...,:inputs.shape[-1]].view(-1,1)
        #     distances_0 = (torch.sum(flat_input**2, dim=1, keepdim=True) 
        #             + torch.sum(self._embedding.weight[0:1]**2, dim=1)
        #             - 2 * torch.matmul(flat_input, self._embedding.weight[0:1].t()))
        #     distances_0 = distances_0[offstage.squeeze()==1]

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        if offstage is not None:
            offstage = offstage[...,:inputs.shape[1]].view(-1)
            encoding_indices[offstage==1] = torch.zeros_like(encoding_indices[offstage==1]) # force 0 token for silient
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # if offstage is not None:
        #     # convert quantized from BWC -> BCW
        #     return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings, distances_0
        # else:
        #     return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings, None
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings, distances




class ResBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size,dilation=1,residual=True,resample=[],pool=None,shape='3D',causal=False,anticausal=False,norm = nn.GroupNorm,transpose=False):
        super(ResBlock, self).__init__()
        self.residual = residual
        self.pool = pool
        self.inputs_resample = resample
        self.dim_missmatch = (inputs!=outputs)
        self.resample = resample
        if not self.resample:
            self.resample=1
        self.padding = list(np.array(dilation)*(np.array(kernel_size)-np.array(resample))//2)
        if shape=='1D':
            conv=ln.ConvTranspose1d if transpose else ln.Conv1d
        if shape=='2D':
            conv=ln.ConvTranspose2d if transpose else ln.Conv2d
        if shape=='3D':
            conv=ln.ConvTranspose3d if transpose else ln.Conv3d
        # if np.any(np.array(causal)):
        #    norm = GroupNormXDim
        # else:
        #    norm = nn.GroupNorm
        # self.padding = [dilation[i]*(kernel_size[i]-1)//2 for i in range(len(dilation))]
        if residual:
            self.norm1 = norm(min(inputs,32),inputs)
        else:
            self.norm1 = norm(min(outputs,32),outputs)
        self.conv1 = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False,causal=causal,anticausal = anticausal)
        
        if self.inputs_resample or self.dim_missmatch:
            self.convskip = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False,causal=causal,anticausal = anticausal)
                
        self.conv2 = conv(outputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False,causal=causal,anticausal = anticausal)
        self.norm2 = norm(min(outputs,32),outputs)

    def forward(self,x):
        if self.residual:
            x = F.relu(self.norm1(x))
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
            else:
                x_skip = x
            x = F.relu(self.norm2(self.conv1(x)))
            x = self.conv2(x)
            x = x_skip + x

        else:
            x = F.relu(self.norm1(self.conv1(x)))
            x = F.relu(self.norm2(self.conv2(x)))
        return x

# class Encoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(Encoder, self).__init__()

# #         self._conv_1 = nn.Conv1d(in_channels=in_channels,
# #                                  out_channels=num_hiddens//2,
# #                                  kernel_size=4,
# #                                  stride=2, padding=1)
# #         self._conv_2 = nn.Conv1d(in_channels=num_hiddens//2,
# #                                  out_channels=num_hiddens,
# #                                  kernel_size=4,
# #                                  stride=2, padding=1)
# #         self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
# #                                  out_channels=num_hiddens,
# #                                  kernel_size=3,
# #                                  stride=1, padding=1)

#         self._conv_1 = nn.Conv1d(in_channels=in_channels,
#                                  out_channels=num_hiddens//8,
#                                  kernel_size=4,
#                                  stride=4, padding=0)
#         self._conv_2 = nn.Conv1d(in_channels=num_hiddens//8,
#                                  out_channels=num_hiddens//2,
#                                  kernel_size=4,
#                                  stride=4, padding=0)
#         self._conv_3 = nn.Conv1d(in_channels=num_hiddens//2,
#                                  out_channels=num_hiddens,
#                                  kernel_size=4,
#                                  stride=2, padding=1)
#         self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
#                                  out_channels=num_hiddens,
#                                  kernel_size=3,
#                                  stride=1, padding=1)
#         self._residual_stack = ResidualStack(in_channels=num_hiddens,
#                                              num_hiddens=num_hiddens,
#                                              num_residual_layers=num_residual_layers,
#                                              num_residual_hiddens=num_residual_hiddens)

#     def forward(self, inputs):
#         x = self._conv_1(inputs)
#         x = F.relu(x)
        
#         x = self._conv_2(x)
#         x = F.relu(x)
        
#         x = self._conv_3(x)
#         x = F.relu(x)
        
#         x = self._conv_4(x)
#         return self._residual_stack(x)





# # #######################convbased#############################
# class Residual(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
#         super(Residual, self).__init__()
#         self._block = nn.Sequential(
#             nn.ReLU(True),
#             nn.Conv1d(in_channels=in_channels,
#                       out_channels=num_residual_hiddens,
#                       kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(True),
#             nn.Conv1d(in_channels=num_residual_hiddens,
#                       out_channels=num_hiddens,
#                       kernel_size=1, stride=1, bias=False)
#         )
    
#     def forward(self, x):
#         return x + self._block(x)


# class ResidualStack(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(ResidualStack, self).__init__()
#         self._num_residual_layers = num_residual_layers
#         self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
#                              for _ in range(self._num_residual_layers)])

#     def forward(self, x):
#         for i in range(self._num_residual_layers):
#             x = self._layers[i](x)
#         return F.relu(x)

# class Encoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, downfolds):
#         super(Encoder, self).__init__()

# #         self._conv_1 = nn.Conv1d(in_channels=in_channels,
# #                                  out_channels=num_hiddens//2,
# #                                  kernel_size=4,
# #                                  stride=2, padding=1)
# #         self._conv_2 = nn.Conv1d(in_channels=num_hiddens//2,
# #                                  out_channels=num_hiddens,
# #                                  kernel_size=4,
# #                                  stride=2, padding=1)
# #         self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
# #                                  out_channels=num_hiddens,
# #                                  kernel_size=3,
# #                                  stride=1, padding=1)
#         inchans = [in_channels] + [num_hiddens//2**(downfolds-i-1) for i in range(downfolds-1)]
#         outchans = [num_hiddens//2**(downfolds-i-1) for i in range(downfolds-1)] + [num_hiddens]
#         self.convs = nn.ModuleList([nn.Conv1d(inchans[i],outchans[i],kernel_size=4,stride=2,padding=1) for i in range(len(inchans))])
#         # self._conv_1 = nn.Conv1d(in_channels=in_channels,
#         #                         out_channels=num_hiddens//16,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_2 = nn.Conv1d(in_channels=num_hiddens//16,
#         #                         out_channels=num_hiddens//8,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_3 = nn.Conv1d(in_channels=num_hiddens//8,
#         #                         out_channels=num_hiddens//4,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_4 = nn.Conv1d(in_channels=num_hiddens//4,
#         #                         out_channels=num_hiddens//2,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_5 = nn.Conv1d(in_channels=num_hiddens//2,
#         #                         out_channels=num_hiddens,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         self._conv_6 = nn.Conv1d(in_channels=num_hiddens,
#                                 out_channels=num_hiddens,
#                                 kernel_size=3,
#                                 stride=1, padding=1)
#         self._residual_stack = ResidualStack(in_channels=num_hiddens,
#                                             num_hiddens=num_hiddens,
#                                             num_residual_layers=num_residual_layers,
#                                             num_residual_hiddens=num_residual_hiddens)

#     def forward(self, inputs):
#         x = inputs
#         for l in self.convs:
#             x = l(x)
#             x = F.relu(x)
#         # x = self._conv_1(inputs)
#         # x = F.relu(x)
        
#         # x = self._conv_2(x)
#         # x = F.relu(x)
        
#         # x = self._conv_3(x)
#         # x = F.relu(x)
        
#         # x = self._conv_4(x)
#         # x = F.relu(x)

#         # x = self._conv_5(x)
#         # x = F.relu(x)
        
#         x = self._conv_6(x)
#         return self._residual_stack(x)

# class Decoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_dim, upfolds):
#         super(Decoder, self).__init__()
        
#         self._conv_1 = nn.Conv1d(in_channels=in_channels,
#                                  out_channels=num_hiddens,
#                                  kernel_size=3, 
#                                  stride=1, padding=1)
        
#         self._residual_stack = ResidualStack(in_channels=num_hiddens,
#                                              num_hiddens=num_hiddens,
#                                              num_residual_layers=num_residual_layers,
#                                              num_residual_hiddens=num_residual_hiddens)
        
#         inchans=[num_hiddens//2**(i) for i in range(upfolds)]
#         outchans=[num_hiddens//2**(i+1) for i in range(upfolds-1)] + [output_dim]
#         self.convs = nn.ModuleList([nn.ConvTranspose1d(inchans[i],outchans[i],4,2,1) for i in range(len(inchans))])
# #         self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens, 
# #                                                 out_channels=num_hiddens//2,
# #                                                 kernel_size=4, 
# #                                                 stride=2, padding=1)
        
# #         self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens//2, 
# #                                                 out_channels=num_hiddens//8,
# #                                                 kernel_size=4, 
# #                                                 stride=4)
        
# #         self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens//8, 
# #                                                 out_channels=12,
# #                                                 kernel_size=4, 
# #                                                 stride=4)

#     def forward(self, inputs):
#         x = self._conv_1(inputs)
        
#         x = self._residual_stack(x)
        
#         for l in self.convs[:-1]:
#             x = l(x)
#             x = F.relu(x)
#         x = self.convs[-1](x)
#         return x
#         # x = self._conv_trans_1(x)
#         # x = F.relu(x)
        
#         # x = self._conv_trans_2(x)
#         # x = F.relu(x)
        
#         # return self._conv_trans_3(x)

# class Model(nn.Module):
#     def __init__(self, num_hiddens, input_dim, num_residual_layers, num_residual_hiddens,
#                 num_embeddings, embedding_dim, commitment_cost, decay=0,folds=5):
#         super(Model, self).__init__()
#         self._encoder = Encoder(input_dim, num_hiddens, num_residual_layers, num_residual_hiddens,
#                                 downfolds=folds)
#         self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, 
#                                         out_channels=embedding_dim,
#                                         kernel_size=1, 
#                                         stride=1)
#         if decay > 0.0:
#             self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
#                                                 commitment_cost, decay)
#         else:
#             self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
#                                             commitment_cost)
#         self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens,
#                                 output_dim=input_dim,
#                                 upfolds=folds)

#     def forward(self, x):
#         z = self._encoder(x)
#         z = self._pre_vq_conv(z)
#         loss, quantized, perplexity, _ = self._vq_vae(z)
#         x_recon = self._decoder(quantized)

#         return loss, x_recon, perplexity


########################Dropout Covnet ##########################
# class Residual(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_hiddens,dropout=0.5):
#         super(Residual, self).__init__()
#         self._block = nn.Sequential(
#             nn.ReLU(True),
#             Conv1d(in_channels=in_channels,
#                       out_channels=num_residual_hiddens,
#                       kernel_size=3, stride=1, padding=1, bias=False,dropout=dropout),
#             nn.ReLU(True),
#             Conv1d(in_channels=num_residual_hiddens,
#                       out_channels=num_hiddens,
#                       kernel_size=1, stride=1, bias=False,dropout=dropout)
#         )
    
#     def forward(self, x):
#         return x + self._block(x)


# class ResidualStack(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(ResidualStack, self).__init__()
#         self._num_residual_layers = num_residual_layers
#         self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
#                              for i in range(self._num_residual_layers)])

#     def forward(self, x):
#         for i in range(self._num_residual_layers):
#             x = self._layers[i](x)
#         return F.relu(x)

# class Encoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, downfolds):
#         super(Encoder, self).__init__()

# #         self._conv_1 = nn.Conv1d(in_channels=in_channels,
# #                                  out_channels=num_hiddens//2,
# #                                  kernel_size=4,
# #                                  stride=2, padding=1)
# #         self._conv_2 = nn.Conv1d(in_channels=num_hiddens//2,
# #                                  out_channels=num_hiddens,
# #                                  kernel_size=4,
# #                                  stride=2, padding=1)
# #         self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
# #                                  out_channels=num_hiddens,
# #                                  kernel_size=3,
# #                                  stride=1, padding=1)
#         inchans = [in_channels] + [num_hiddens//2**(downfolds-i-1) for i in range(downfolds-1)]
#         outchans = [num_hiddens//2**(downfolds-i-1) for i in range(downfolds-1)] + [num_hiddens]
#         self.convs = nn.ModuleList([Conv1d(inchans[i],outchans[i],kernel_size=4,stride=2,padding=1) for i in range(len(inchans))])
#         # self._conv_1 = nn.Conv1d(in_channels=in_channels,
#         #                         out_channels=num_hiddens//16,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_2 = nn.Conv1d(in_channels=num_hiddens//16,
#         #                         out_channels=num_hiddens//8,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_3 = nn.Conv1d(in_channels=num_hiddens//8,
#         #                         out_channels=num_hiddens//4,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_4 = nn.Conv1d(in_channels=num_hiddens//4,
#         #                         out_channels=num_hiddens//2,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         # self._conv_5 = nn.Conv1d(in_channels=num_hiddens//2,
#         #                         out_channels=num_hiddens,
#         #                         kernel_size=4,
#         #                         stride=2, padding=1)
#         self._conv_6 = Conv1d(in_channels=num_hiddens,
#                                 out_channels=num_hiddens,
#                                 kernel_size=3,
#                                 stride=1, padding=1)
#         self._residual_stack = ResidualStack(in_channels=num_hiddens,
#                                             num_hiddens=num_hiddens,
#                                             num_residual_layers=num_residual_layers,
#                                             num_residual_hiddens=num_residual_hiddens)

#     def forward(self, inputs):
#         x = inputs
#         for l in self.convs:
#             x = l(x)
#             x = F.relu(x)
#         # x = self._conv_1(inputs)
#         # x = F.relu(x)
        
#         # x = self._conv_2(x)
#         # x = F.relu(x)
        
#         # x = self._conv_3(x)
#         # x = F.relu(x)
        
#         # x = self._conv_4(x)
#         # x = F.relu(x)

#         # x = self._conv_5(x)
#         # x = F.relu(x)
        
#         x = self._conv_6(x)
#         return self._residual_stack(x)

# class Decoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, output_dim, upfolds):
#         super(Decoder, self).__init__()
        
#         self._conv_1 = nn.Conv1d(in_channels=in_channels,
#                                  out_channels=num_hiddens,
#                                  kernel_size=3, 
#                                  stride=1, padding=1)
        
#         self._residual_stack = ResidualStack(in_channels=num_hiddens,
#                                              num_hiddens=num_hiddens,
#                                              num_residual_layers=num_residual_layers,
#                                              num_residual_hiddens=num_residual_hiddens)
        
#         inchans=[num_hiddens//2**(i) for i in range(upfolds)]
#         outchans=[num_hiddens//2**(i+1) for i in range(upfolds-1)] + [output_dim]
#         self.convs = nn.ModuleList([ConvTranspose1d(inchans[i],outchans[i],4,2,1,dropout=0. if i==(len(inchans)-1) else 0.5) for i in range(len(inchans))])
# #         self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens, 
# #                                                 out_channels=num_hiddens//2,
# #                                                 kernel_size=4, 
# #                                                 stride=2, padding=1)
        
# #         self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens//2, 
# #                                                 out_channels=num_hiddens//8,
# #                                                 kernel_size=4, 
# #                                                 stride=4)
        
# #         self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens//8, 
# #                                                 out_channels=12,
# #                                                 kernel_size=4, 
# #                                                 stride=4)

#     def forward(self, inputs):
#         x = self._conv_1(inputs)
        
#         x = self._residual_stack(x)
        
#         for l in self.convs[:-1]:
#             x = l(x)
#             x = F.relu(x)
#         x = self.convs[-1](x)
#         return x
#         # x = self._conv_trans_1(x)
#         # x = F.relu(x)
        
#         # x = self._conv_trans_2(x)
#         # x = F.relu(x)
        
#         # return self._conv_trans_3(x)

# class Model(nn.Module):
#     def __init__(self, num_hiddens, input_dim, num_residual_layers, num_residual_hiddens,
#                 num_embeddings, embedding_dim, commitment_cost, decay=0,folds=5):
#         super(Model, self).__init__()
#         self._encoder = Encoder(input_dim, num_hiddens, num_residual_layers, num_residual_hiddens,
#                                 downfolds=folds)
#         self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, 
#                                         out_channels=embedding_dim,
#                                         kernel_size=1, 
#                                         stride=1)
#         if decay > 0.0:
#             self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
#                                                 commitment_cost, decay)
#         else:
#             self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
#                                             commitment_cost)
#         self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens,
#                                 output_dim=input_dim,
#                                 upfolds=folds)

#     def forward(self, x):
#         z = self._encoder(x)
#         z = self._pre_vq_conv(z)
#         loss, quantized, perplexity, _ = self._vq_vae(z)
#         x_recon = self._decoder(quantized)

#         return loss, x_recon, perplexity




####################### resnet #############################

class ResBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size,dilation=1,residual=True,resample=[],pool=None,shape='3D',causal=False,anticausal=False,norm = nn.GroupNorm,transpose=False,isolated_token=False):
        super(ResBlock, self).__init__()
        self.residual = residual
        self.pool = pool
        self.inputs_resample = resample
        self.dim_missmatch = (inputs!=outputs)
        self.resample = resample
        if not self.resample:
            self.resample=1
        self.padding = list(np.array(dilation)*(np.array(kernel_size)-np.array(resample))//2)
        kernel_size2 = [k-1 for k in kernel_size] if isinstance(kernel_size, list) else kernel_size-1
        self.padding2 = list(np.array(dilation)*(np.array(kernel_size)-1-np.array([1 for i in range(len(resample))]))//2)
        if shape=='1D':
            conv=ln.ConvTranspose1d if transpose else ln.Conv1d
        if shape=='2D':
            conv=ln.ConvTranspose2d if transpose else ln.Conv2d
        if shape=='3D':
            conv=ln.ConvTranspose3d if transpose else ln.Conv3d
        # if np.any(np.array(causal)):
        #    norm = GroupNormXDim
        # else:
        #    norm = nn.GroupNorm
        # self.padding = [dilation[i]*(kernel_size[i]-1)//2 for i in range(len(dilation))]
        if residual:
            self.norm1 = norm(min(inputs,32),inputs)
        else:
            self.norm1 = norm(min(outputs,32),outputs)
        self.conv1 = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False,causal=causal,anticausal = anticausal)
        
        if self.inputs_resample or self.dim_missmatch:
            self.convskip = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False,causal=causal,anticausal = anticausal)
        self.conv2 = conv(outputs, outputs, 1 if isolated_token else 3, 1, 0 if isolated_token else 1, dilation=dilation, bias=False,causal=causal,anticausal = anticausal)
        self.norm2 = norm(min(outputs,32),outputs)

    def forward(self,x,p=0.):
        if self.residual:
            x = F.dropout(F.leaky_relu(self.norm1(x),0.2),p=p)
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
            else:
                x_skip = x
            x = F.dropout(F.leaky_relu(self.norm2(self.conv1(x)),0.2),p=p)
            x = self.conv2(x)
            x = x_skip + x

        else:
            x = F.relu(self.norm1(self.conv1(x)))
            x = F.relu(self.norm2(self.conv2(x)))
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, downfolds):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=num_hiddens//2**downfolds,
                                kernel_size=3,
                                stride=1, padding=1)
        self.resblocks = nn.ModuleList([ResBlock(num_hiddens//2**(downfolds-i),num_hiddens//2**(downfolds-i-1),kernel_size=[4],resample=[2],shape='1D') for i in range(downfolds)])

        self._conv_2 = nn.Conv1d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        for i, l in enumerate(self.resblocks):
            x = l(x)
        x = F.leaky_relu(x,0.2)
        x = self._conv_2(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, output_dim, upfolds, isolated_token):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=1 if isolated_token else 3, 
                                stride=1, padding=0 if isolated_token else 1)
        
        # self.resblocks = nn.ModuleList([ResBlock(num_hiddens//2**(i),num_hiddens//2**(i+1),kernel_size=[2 if isolated_token and i==0 else 4],resample=[2],shape='1D',transpose=True,isolated_token=isolated_token) for i in range(upfolds)]) # isolated token for the first layer
        # self.resblocks = nn.ModuleList([ResBlock(num_hiddens//2**(i),num_hiddens//2**(i+1),kernel_size=[2 if isolated_token and i<=(upfolds-3) else 4],resample=[2],shape='1D',transpose=True,isolated_token=isolated_token) for i in range(upfolds)]) # last 3 fold smooth
        # self.resblocks = nn.ModuleList([ResBlock(num_hiddens//2**(i),num_hiddens//2**(i+1),kernel_size=[2 if isolated_token and i!=0 else 4],resample=[2],shape='1D',transpose=True,isolated_token=isolated_token) for i in range(upfolds)]) # except 1st layer smooth
        # self.resblocks = nn.ModuleList([ResBlock(num_hiddens//2**(i),num_hiddens//2**(i+1),kernel_size=[2 if isolated_token else 4],resample=[2],shape='1D',transpose=True,isolated_token=isolated_token) for i in range(upfolds)]) # fully isolated
        self.resblocks = nn.ModuleList([ResBlock(num_hiddens//2**(i),num_hiddens//2**(i+1),kernel_size=[2 if isolated_token else 4],resample=[2],shape='1D',transpose=True,isolated_token=False) for i in range(upfolds)]) # fully smooth

        self._conv_2 = nn.Conv1d(in_channels=num_hiddens//2**upfolds,
                                out_channels=output_dim,
                                kernel_size=1 if isolated_token else 3,
                                stride=1, padding=0 if isolated_token else 1)


    def forward(self, inputs):
        x = self._conv_1(inputs)
        for i, l in enumerate(self.resblocks):
            x = l(x)
        x = F.leaky_relu(x,0.2)
        x = self._conv_2(x)
        return x




class Model(nn.Module):
    def __init__(self, num_hiddens, input_dim, num_residual_layers, num_residual_hiddens,
                num_embeddings, embedding_dim, commitment_cost, decay=0,folds=[5], input_dim_dual=None, isolated_token=False):
        super(Model, self).__init__()
        self.num_embeddings = num_embeddings
        self.folds = folds
        self._encoder = nn.ModuleList([])
        input_dim_org = input_dim
        input_dim = input_dim
        for i,f in enumerate(folds): # down -> top
            self._encoder.append(Encoder(input_dim, num_hiddens,
                                    downfolds=f))
            input_dim = num_hiddens

        self._pre_vq_conv = nn.ModuleList([])
        in_channels = num_hiddens
        for i,f in enumerate(folds[::-1]): # top -> down
            self._pre_vq_conv.append(nn.Conv1d(in_channels=in_channels, 
                                            out_channels=embedding_dim,
                                            kernel_size=1, 
                                            stride=1))
            in_channels = num_hiddens + embedding_dim
        
        self._vq_vae = nn.ModuleList([])
        for i,f in enumerate(folds[::-1]): # top -> down
            if decay > 0.0:
                self._vq_vae.append(VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                                    commitment_cost, decay))
            else:
                self._vq_vae.append(VectorQuantizer(num_embeddings, embedding_dim,
                                            commitment_cost))

        self._upsampler = nn.ModuleList([])
        in_channels = embedding_dim
        for i,f in enumerate(folds[::-1]): # top -> down
            self._upsampler.append(nn.ConvTranspose1d(in_channels=in_channels, 
                                            out_channels=embedding_dim,
                                            kernel_size=2**f, 
                                            stride=2**f))
            in_channels = embedding_dim*2

        self._decoder = nn.ModuleList([])
        for i,f in enumerate(folds[::-1]): # top -> down
            output_dim = embedding_dim if i != (len(folds)-1) else input_dim_org
            input_dim = (embedding_dim if i != (len(folds)-1) else embedding_dim*2) if len(folds)>1 else embedding_dim
            self._decoder.append(Decoder(input_dim,
                                    num_hiddens, output_dim=output_dim,
                                    upfolds=f,isolated_token=isolated_token))

    def encoding(self,x,offstage=None):
        encoded = []
        for i in range(len(self._encoder)): # down -> top
            x = self._encoder[i](x)
            encoded += [x]
        quantized = []
        encodings_lable = []
        encodings_onehot = []
        loss = 0
        perplexity=0
        enc = encoded[-1]
        for i in range(len(self._vq_vae)): # top -> down
            z = self._pre_vq_conv[i](enc)
            if offstage is not None:
                loss_, quantized_, perplexity_, encodings, distance_0 = self._vq_vae[i](z,offstage[len(self._vq_vae)-i-1])
            else:
                loss_, quantized_, perplexity_, encodings, distance_0 = self._vq_vae[i](z,None)
            encodings = encodings.reshape(x.shape[0],-1,encodings.shape[-1])
            encodings_onehot += [encodings]
            encodings_lable += [torch.argmax(encodings,-1).unsqueeze(1)]
            quantized += [quantized_]
            loss += loss_
            perplexity += perplexity_
            if i != len(self._vq_vae)-1:
                decoded = self._decoder[i](quantized_)
                enc = torch.cat([decoded, encoded[len(self._vq_vae)-i-2]], 1)
                
        return quantized, loss, perplexity, encodings_onehot, encodings_lable

    def decoding(self,quantized): #quantized: top->down
        quant = quantized[0] #top quantization
        if len(self._upsampler)>1:
            for i in range(len(self._upsampler)-1): # top -> down
                upsampled = self._upsampler[i](quant)
                quant = torch.cat([upsampled, quantized[i+1]], 1)
        dec = self._decoder[-1](quant)
        return dec

    def decode_from_token(self,code,onehot=True): #code: onehot embedding of token, BWC, top->down
        if not onehot:
            code = [F.one_hot(c.squeeze(1), self.num_embeddings).to(torch.float) for c in code]
        quantized = []
        for i,x in enumerate(code): #top->down
            input_shape = x.shape
            x = x.view(-1,input_shape[-1])
            x = torch.matmul(x, self._vq_vae[i]._embedding.weight).view(input_shape[0],input_shape[1],-1)
            x = x.permute(0, 2, 1).contiguous() #BWC -> BCW
            quantized+=[x]
        dec = self.decoding(quantized)
        return dec

    def forward(self, x, datatypes=None,offstage=None):
        x_return={}
        x_input = x
        x = x_input[datatypes[0]]
        quantized, loss, perplexity, encodings_onehot, encodings_lable = self.encoding(x,offstage)
        x_recon = self.decoding(quantized)
        x_return[datatypes[0]] = x_recon
        return loss, x_return, perplexity

        # if x_dual is not None:
        #     return loss, x_recon, x_dual_recon, perplexity, distance_0
        # else:
        #     return loss, x_recon, perplexity, distance_0

