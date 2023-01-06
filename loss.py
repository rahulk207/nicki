import torch
import torch.nn as nn

class MisclassificationLoss(nn.Module):
    def __init__(self, victim_nodes, num_fake_nodes):
        super().__init__()
        self.victim_nodes = victim_nodes
        self.num_fake_nodes = num_fake_nodes

    def forward(self, x, feature_orig,feature_gen,HIDDEN):
        logits = torch.exp(x)
        score = -torch.sum(torch.mean(logits[self.victim_nodes],dim=0)*torch.log(logits[-self.num_fake_nodes:]),dim=1)

        if HIDDEN:
            sum_feat = torch.sum(feature_gen, dim=1,keepdim=True)
            sum_feat[sum_feat==0] = 1
            feature_gen = torch.div(feature_gen,sum_feat)
            score_feature = -torch.sum(feature_gen[-self.num_fake_nodes:] * torch.log(feature_orig[-self.num_fake_nodes:]+1e-14),dim=1)
            return torch.mean(score)+0.1*torch.mean(score_feature)
        

        return torch.mean(score)

