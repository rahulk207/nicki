import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import *
import pickle
import argparse



parser = argparse.ArgumentParser(description='Budget input')
parser.add_argument('-budget', type=float,
                    help='Budget for node and edge')

args = parser.parse_args()
#print(args.budget)

# cuda setup
device = torch.device("cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} 

# hyper params
batch_size = 64
latent_size = 20
epochs = 500

base_class = 5
budget_coeff = args.budget
num_target_nodes = 359
num_attacker_nodes = int(budget_coeff*num_target_nodes)

f = open("cora_graph", "rb")
graph = pickle.load(f)

features = torch.from_numpy(graph['features']) #normalized features
# print(list(features[0]))
features = torch.ceil(features)
labels_all = torch.from_numpy(graph['labels'])
num_classes = len(np.unique(labels_all))
feat_dim = features.shape[1]

avg_feat, c = 0, 0
for i in range(len(features)):
    if labels_all[i] == base_class:
        c += 1
        avg_feat += sum(features[i])
avg_feat /= c
print("True average num ones", avg_feat)

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# create a CVAE model
model = CVAE(feat_dim, latent_size, num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    num_batches = (len(features)//batch_size) + 1
    if len(features)%batch_size == 0:
        num_batches -= 1
    for b in range(num_batches):
        data = features[b*batch_size: min((b+1)*batch_size, len(features)), :]
        labels = labels_all[b*batch_size: min((b+1)*batch_size, len(features))]
        data, labels = data.to(device), labels.to(device)
        labels = one_hot(labels, num_classes)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if b % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, b * len(data), len(features),
                100. * b / len(features),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(features)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, labels) in enumerate(test_loader):
#             data, labels = data.to(device), labels.to(device)
#             labels = one_hot(labels, 10)
#             recon_batch, mu, logvar = model(data, labels)
#             test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
#             if i == 0:
#                 n = min(data.size(0), 5)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(-1, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
    train(epoch)
#     # test(epoch)

f = open("cvae_feature_generator.pickle", "wb")
pickle.dump(model, f)

f = open("cvae_feature_generator.pickle", "rb")
model = pickle.load(f)

with torch.no_grad():
    c = torch.zeros(num_attacker_nodes, num_classes)
    c[:, base_class] = 1
    sample = torch.randn(num_attacker_nodes, latent_size).to(device)
    sample = model.decode(sample, c).cpu()
    
    for i in range(num_attacker_nodes):
        s = sample[i]
        mask1 = (s >= 1e-5) * (s < 1e-4)
        mask2 = (s >= 1e-4) * (s < 1e-3)
        mask3 = (s >= 1e-3) * (s < 1e-2)
        mask4 = (s >= 1e-2) * (s < 1e-1)
        final_mask = (s >= 0.09)
        print("Num between 1e-5 and 1e-4", sum(mask1.type(torch.float)))
        print("Num between 1e-4 and 1e-3", sum(mask2.type(torch.float)))
        print("Num between 1e-3 and 1e-2", sum(mask3.type(torch.float)))
        print("Num between 1e-2 and 1e-1", sum(mask4.type(torch.float)))
        print("Num final", sum(final_mask.type(torch.float)))
        sample[i] = torch.where(final_mask, torch.ones(s.shape), torch.zeros(s.shape))
        print(torch.sum(sample[i]))
        
    f = open(f"attacker_features_{args.budget}", "wb")
    pickle.dump(sample, f)


