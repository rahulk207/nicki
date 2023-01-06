import torch.nn.functional as F
import torch


class TopK_soft(torch.nn.Module):
    def __init__(self,k,t):
        super(TopK_soft, self).__init__()
        self.k = k
        self.t = t

    def forward(self, x):
        m = torch.distributions.gumbel.Gumbel(0,1)
        z = m.sample(torch.tensor(x.size()))
        w = torch.log(x+1e-30)
        keys = w + z
        onehot_approx = torch.zeros_like(x)
        khot_list = None
        for i in range(self.k):
            khot_mask = torch.maximum(1 - onehot_approx, torch.full(x.size(),1e-20))
            keys += torch.log(khot_mask)
            onehot_approx = F.softmax(keys / self.t, dim=-1)
            if khot_list is None:
                khot_list = onehot_approx.clone()
            else:
                khot_list += onehot_approx
        return khot_list


