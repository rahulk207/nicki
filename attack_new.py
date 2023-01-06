import numpy as np
from sig_torch import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim, hid_dim, budget, feat_budget, drop_prob=0.0, acti_fn=nn.ReLU(), bias=True, old_nodes=0, x=0,y=0,z=0,r1=0,r2=0):
        super().__init__()
        self.old = old_nodes
        self.fc11 = Dense(latent_dim, y, drop_prob, acti_fn, bias)
        self.fc12 = Dense(y, latent_dim, drop_prob, acti_fn, bias)
        self.fc13 = Dense(latent_dim, x, drop_prob, acti_fn, bias)
        self.fc14 = Dense(x, x, drop_prob, acti_fn, bias)
        self.fc15 = Dense(x, z, drop_prob, acti_fn, bias)
        self.fc16 = Dense(z, 1, drop_prob, None, bias)
        self.fc21 = Dense(latent_dim, r1, drop_prob, acti_fn, bias)
        self.fc22 = Dense(r1, r2, drop_prob, acti_fn, bias)
        self.fc23 = Dense(r2, latent_dim, drop_prob, None, bias)
        self.encoder = Encoder(input_dim, hid_dim, latent_dim, drop_prob=0.0, acti_fn=nn.ReLU())
        self.budget = budget; self.feat_budget = feat_budget
        self.topk_activ_adj = TopK_soft(self.budget, 2)
        self.topk_activ_feat = TopK_soft(self.feat_budget, 2)

    def forward(self, x, adj, mask, adj_o,features_o, mask_internal_edges):
        """
        call encoder here
        then generate_dense
        then add edges (modification of delete_k_edge_min)
        """
        z = self.encoder(x, adj)
        A_tilde, X_tilde = self.generate_dense(z, mask, adj_o,features_o, mask_internal_edges)
        return A_tilde, X_tilde

    def generate_dense(self, z, mask, adj_o,features_o, mask_internal_edges):
        

        reconstructions_X1 = self.fc21(z[self.old:])
        reconstructions_X2 = self.fc22(reconstructions_X1)
        reconstructions_X = self.fc23(reconstructions_X2)


        reconstructions_X = torch.reshape(reconstructions_X,(1,reconstructions_X.shape[0]*reconstructions_X.shape[1]))
        reconstructions_X = F.softmax(reconstructions_X, dim=-1)

        reconstructions_X = self.topk_activ_feat(reconstructions_X)

        temp_mask = reconstructions_X >= 0.9
        reconstructions_X = torch.reshape(reconstructions_X, (mask.shape[0], features_o.shape[1]))
        reconstructions_X = torch.cat((features_o, reconstructions_X), dim=0)

        update_temp=[]
        for i in range(self.old, reconstructions_X.shape[0]):
            update_temp.append(reconstructions_X[i,:] * reconstructions_X)
        final_update_A = torch.stack(update_temp, axis=0)
        final_update_A1 = self.fc11(final_update_A)
        final_update_A2 = self.fc12(final_update_A1)
        final_update_A3 = self.fc13(final_update_A2)
        final_update_A4 = self.fc14(final_update_A3)
        final_update_A5 = self.fc15(final_update_A4)
        reconstructions_A = self.fc16(final_update_A5)
        reconstructions_A = torch.squeeze(reconstructions_A)

        reconstructions_A = torch.where((mask) & (~ mask_internal_edges), reconstructions_A, torch.full(reconstructions_A.shape, -1e+20)) 
        reconstructions_A = torch.reshape(reconstructions_A,(1,reconstructions_A.shape[0]*reconstructions_A.shape[1]))
        reconstructions_A = F.softmax(reconstructions_A, dim=-1)
        
        
        reconstructions_A = self.topk_activ_adj(reconstructions_A) 

        temp_mask = reconstructions_A == 1

        reconstructions_A = torch.reshape(reconstructions_A, (mask.shape[0], mask.shape[1]))
        reconstructions_A = torch.where(mask_internal_edges, torch.ones(reconstructions_A.size()), reconstructions_A)
        temp_adj = torch.cat((adj_o,torch.zeros((mask.shape[1]-mask.shape[0], mask.shape[0]))),dim=1)
        reconstructions_A = torch.cat((temp_adj, reconstructions_A),dim=0)
        lower_reconstructions_A = torch.tril(reconstructions_A,diagonal=-1)
        diag_reconstructions_A = torch.diag(torch.ones(reconstructions_A.shape[0]))
        reconstructions_A = lower_reconstructions_A + torch.transpose(lower_reconstructions_A, 0, 1) + diag_reconstructions_A
        

        return reconstructions_A, reconstructions_X


    
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, latent_dim, drop_prob=0.0, acti_fn=nn.ReLU()):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hid_dim, drop_prob, nn.ReLU())
        self.gc2 = GraphConvolution(hid_dim, latent_dim, drop_prob, nn.ReLU())

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        z = self.gc2(x, adj)
        return z


