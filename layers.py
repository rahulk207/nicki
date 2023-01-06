import torch
import torch.nn as nn

class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        self.kprob=1-dprob

    def forward(self, x):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)

        return torch.sparse.FloatTensor(rc, val)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, drop_prob=0.0, acti_fn=nn.ReLU()):
        super().__init__()
        self.dr = SparseDropout(drop_prob)
        self.ac = acti_fn
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        self.adj = adj
        x = self.fc(x)
        x = torch.matmul(self.adj, x)
        output = self.ac(x)

        return output

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, drop_prob=0.0, acti_fn=nn.ReLU(), bias=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias)
        self.dr = nn.Dropout(drop_prob)
        self.ac = acti_fn

    def forward(self, x):
        x = self.fc(x)
        if self.ac == None:
            return x
        output = self.ac(x).clone()

        return output

