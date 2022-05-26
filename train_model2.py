""" Script which trains the models """
import argparse
from distutils.command.config import config
import json
import os
import time
from turtle import distance
import torch
from torchvision.utils import make_grid
import wandb
import functools
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/rwang/bravo/avatar/code/')
from articulatory.parallel_wavegan.utils import load_model
from articulatory.parallel_wavegan.bin.decode import ar_loop
import torchaudio
import soundfile as sf

import torch.nn.functional as F
from collections import defaultdict

import neural_synthesis.models
import yaml
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import umap

import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloader_bravo import get_dataloaders
import tqdm

from vqvae2_model import *


os.environ["WANDB_SILENT"] = "true"

class Checkpointer(object):
    """Returns checkpoint path for saving a model."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, ckpt_number):
        return os.path.join(self.checkpoint_dir,
                            'model_' + str(ckpt_number) + '.pt')

def train_model():
    # run_name = 'resnet_spec_down3_weightdecay'
    # run_name = 'resnet_spg_down3_unit8_106wset_0token'
    # run_name = 'resnet_spec_down6_unit100_106wset_0token_isolated_token_except1layer'
    run_name = 'resnet_spec_down4_unit100_106wset_vqvae2'
    # run_name = 'resnet_ema_down4_apr11'
    wandb.init(project='vqvae', name=run_name,save_code=True)
    wandb.run.log_code(".")
    run_name = run_name+'_'+wandb.run.id
    checkpoint_dir = '/home/rwang/bravo/avatar/code/pytorch_vq_vae/saved_model/'+run_name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f, Dumper=yaml.Dumper)
    checkpointer = Checkpointer(checkpoint_dir)

    folds=[4] # 2,4,6
    num_embeddings = 100 #100
    isolated_token = False

    num_training_updates = 150000
    num_hiddens = 512
    embedding_dim = 1024
    num_residual_hiddens = 64
    num_residual_layers = 2
    commitment_cost = 0.25
    input_dim_dual = None #256
    decay = 0.99

    learning_rate = 1e-3
    
    with open('/home/rwang/bravo/avatar/code/pytorch_vq_vae/configs/dataloader.yaml') as f:
        dataloaderconfig = yaml.load(f, Loader=yaml.Loader)
        dataset, dataset_test, data_variance = get_dataloaders(dataloaderconfig['train_filename'], dataloaderconfig['test_filename'],
                                                        **dataloaderconfig["dataloader_params"], root_dir=dataloaderconfig['root_dir'],compute_var=True)
        datatype=dataloaderconfig['dataloader_params']['output_types'][0]
        # data_variance = data_variance[datatype]
    if datatype[:3]=='ema':
        input_dim=37#31#12
    elif datatype[:3]=='spc':
        input_dim=512
    elif datatype[:3]=='hgr':
        input_dim=256
    elif datatype[:3]=='spg':
        input_dim=16
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_hiddens, input_dim, num_residual_layers, num_residual_hiddens,
                num_embeddings, embedding_dim, 
                commitment_cost, decay, folds, input_dim_dual, isolated_token=isolated_token).to(device)
    # model = Model(num_hiddens, input_dim=input_dim,
    #                 num_embeddings=num_embeddings, embedding_dim=embedding_dim, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens,
    #                 commitment_cost=commitment_cost, decay=decay,folds=folds).to(device)
    
    # with open(config["generator_config_path"]) as f:
    #     generator_config = yaml.load(f, Loader=yaml.Loader)
    # if torch.cuda.is_available():
    #     generate_device = torch.device("cuda")
    # else:
    #     generate_device = torch.device("cpu")
    # if config['generator_checkpoint_path']:
    #     generator_model = load_model(config["generator_checkpoint_path"], generator_config, generator2=False)
    #     generator_model.remove_weight_norm()
    #     generator_model = generator_model.eval().to(generate_device)
    # else:
    #     generator_model = None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False, weight_decay=0.)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False, weight_decay=0.01)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    total_train_loss = defaultdict(float)
    datatypes = [dataloaderconfig['dataloader_params']['output_types'][0]]

    for i in xrange(num_training_updates):
        data = next(iter(dataset))
        onstage = data['spcaux']
        offstages=[]
        onstage = onstage.to(device)
        for f in folds:
            onstage = F.interpolate(onstage,scale_factor=1/2**f,mode='linear')
            offstages += [(onstage==0).float()]
        for datatype in dataloaderconfig['dataloader_params']['output_types']:
            data[datatype] = data[datatype].permute(0,2,1)[:,-input_dim:].to(device) # BTC->BCT
        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data,datatypes=datatypes,offstage=offstages)
        recon_error = 0
        for datatype in datatypes:
            recon_error_type = F.mse_loss(data_recon[datatype], data[datatype][:,:,:data_recon[datatype].shape[-1]]) / data_variance[datatype]
            total_train_loss["train/" + 'loss_'+datatype[:3]] += recon_error_type.detach().item()
            recon_error += recon_error_type
        # if offstage is not None:
            # logits = -distance[torch.nonzero(offstage[:,:,:data_recon[datatype].shape[-1]].view(-1)).squeeze()]
            # zero_token_loss = F.cross_entropy(logits,torch.zeros(logits.shape[0],dtype=torch.long).to(device))
        # else:
        #     zero_token_loss = torch.tensor(0.)
        zero_token_loss = torch.tensor(0.)
        loss = recon_error + vq_loss + zero_token_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        
        
        total_train_loss["train/" + 'zero_token_loss'] += zero_token_loss.detach().item()
        total_train_loss["train/" + 'perplex'] += perplexity.detach().item()
        if (i+1) % 100 == 0:
            # print('%d iterations' % (i+1))
            # print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            # print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            # print()
            for loss_name, loss_val in total_train_loss.items():
                wandb.log({loss_name: loss_val}, step=i, commit=False)
            total_train_loss = defaultdict(float)
        if (i+1) % 1000 == 0:
            torch.save(model.state_dict(), checkpointer(i))



if __name__ == "__main__":
    print('training')
    train_model()
