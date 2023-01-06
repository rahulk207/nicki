import torch
import torch.optim as optim
from attack_new import *
from loss import *
from utils import *
from gcn import *
import pickle
import numpy as np
import scipy.sparse as sp
import copy
import math
import powerlaw
import argparse
import os
import time

parser = argparse.ArgumentParser(description='Budget input')
parser.add_argument('-budget', type=float,
                    help='Budget for node and edge')
parser.add_argument('-target',type=int,help='target class')

args = parser.parse_args()
start_time = time.time()
print('Time: ',(time.time()-start_time))
if not os.path.exists(str(args.budget)):
    os.makedirs(str(args.budget))
def preprocess(base_class,labels,adj,avg_degree,attacker_node_count,num_old_nodes,internal_degree):
    mask_internal_edges = np.zeros((attacker_node_count, num_old_nodes+attacker_node_count))
    base_idx = (labels==base_class).nonzero(as_tuple=True)[0]
    #print(label_ind)

    deg = torch.sum(adj,dim=1)
    deg_base = torch.clone(deg[labels==base_class])

    deg_base_uniq,deg_base_count = torch.unique(deg_base,return_counts=True)
    prob = deg_base_count/torch.sum(deg_base_count)
    CCDF = torch.sum(prob)-torch.cumsum(prob,dim=0)
    co_ord = np.polyfit(np.log(deg_base_uniq.numpy()),np.log(CCDF.numpy()+1e-4),deg=1)
    '''
    plt.scatter(degree,CCDF)
    plt.scatter(degree,prob)
    plt.plot(degree,degree**(co_ord[0])*np.exp(co_ord[1]))
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('dd.png')
    plt.close()
    '''
    fit_=powerlaw.Power_Law(xmin=1, parameters=[-(co_ord[0]-1)]).generate_random(10000)
    deg_fit_,count_fit_ = np.unique(np.round(fit_),return_counts=True)
    
    for i in range(int(avg_degree)+1):
        ind = np.where(deg_fit_==i)
        deg_fit_ = np.delete(deg_fit_,ind)
        count_fit_ = np.delete(count_fit_,ind)
    prob2 = count_fit_/sum(count_fit_)
    attack_node_degree = np.random.choice(deg_fit_,size=attacker_node_count,p=prob2)
    '''
    plt.scatter(degree,CCDF)
    plt.scatter(degree,prob)
    plt.plot(degree,degree**(co_ord[0])*np.exp(co_ord[1]))
    plt.scatter(val,prob2)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('d2.png')
    '''
    edge_attack = []
    base_idx = torch.cat((base_idx,torch.tensor(range(num_old_nodes,num_old_nodes+attacker_node_count))))
    deg_base = torch.cat((deg_base,torch.tensor(attack_node_degree)))
    for i in range(len(attack_node_degree)):
        edge_attack.append(np.random.choice(base_idx,size = int(internal_degree*attack_node_degree[i]),replace=False,p=torch.div(deg_base, torch.sum(deg_base)))) 

    for i in range(len(edge_attack)):
        for j in range(len(edge_attack[i])):
            mask_internal_edges[i][edge_attack[i][j]] = 1

    return torch.tensor(mask_internal_edges).type(torch.bool),int(np.sum((1-internal_degree)*attack_node_degree))       


def load_params(feat_dim,budget,num_fake_nodes,average_feat_one):
    params = {
    'EPOCHS': 60,
    'input_dim': feat_dim,
    'hid_dim_gen': feat_dim,
    'latent_dim': feat_dim,
    'budget':budget,
    'feat_budget': int(average_feat_one*num_fake_nodes),
    'acti_fn': torch.nn.ReLU()
    }

    return params




def get_train_val_test_gcn(labels, seed=None):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: int((0.10*len(labels))/nclass)])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[int((0.10*len(labels))/nclass): ])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: int(0.10*len(labels))]
    idx_test = idx_unlabeled[int(0.10*len(labels)):]
    return idx_train, idx_val, idx_test



def get_victim_nodes(labels,idx_test,target_class):
    return idx_test[labels==target_class]
    
 

    
    
HIDDEN = False
base_class = 5
target_class = args.target

torch.autograd.set_detect_anomaly(True)


#For pre-training, will have to generate mask for attacker/real nodes
f = open("datasets/cora_graph", "rb")
graph = pickle.load(f)
adj = torch.from_numpy(graph['adj_matrix']); features = torch.from_numpy(graph['features']); labels = torch.from_numpy(graph['labels'])
num_classes = len(np.unique(labels))
feat_dim = features.shape[1]

adj = torch.ceil(adj); features = torch.ceil(features)
adj_o = adj.clone().float()
features_o = features.clone().float()



temp_mask = adj == 1
print("Num ones adj initial", torch.sum(temp_mask.type(torch.float)))

temp_mask = features == 1
print("Num ones features initial", torch.sum(temp_mask.type(torch.float)))




#Addition of fake nodes
budget_coeff = args.budget
average_feat_one = 18.174
avg_degree = 4.89 #for cora
num_old_nodes = len(labels); # print(num_old_nodes)
num_fake_nodes = int(budget_coeff*359) #class 0 test nodes
num_nodes = num_old_nodes + num_fake_nodes
budget = int(num_fake_nodes*avg_degree)
internal_degree = 0.7
print("Num attacker", num_fake_nodes); print("Budget coefficient", budget_coeff)

print("Total nodes", num_nodes)
if HIDDEN:
    f = open(f"attacker_features_{args.budget}", "rb")
    attacker_features = pickle.load(f)

mask = torch.ones((num_fake_nodes,num_old_nodes))
mask2 = torch.zeros(adj.shape)
adj = torch.cat((adj, torch.ones((num_old_nodes, num_fake_nodes))), dim = 1)
adj = torch.cat((adj, torch.ones((num_fake_nodes, num_nodes))), dim = 0).type(torch.float)

if HIDDEN:
    mask_internal_edges,budget = preprocess(base_class,labels,adj_o,avg_degree,num_fake_nodes,num_old_nodes, internal_degree)
    labels = torch.cat((labels, torch.full((num_fake_nodes,),base_class)))
    features = torch.cat((features, attacker_features), dim = 0)

else:
    labels = torch.cat((labels, torch.full((num_fake_nodes,),base_class)))
    features = torch.cat((features, torch.ones((num_fake_nodes, feat_dim))), dim = 0)
    mask_internal_edges = torch.zeros((num_fake_nodes, num_old_nodes+num_fake_nodes)).type(torch.bool)
#print('seed',torch.seed())
mask2 = torch.cat((mask2, torch.ones((num_old_nodes, num_fake_nodes))), dim = 1)
mask2 = torch.cat((mask2, torch.ones((num_fake_nodes, num_nodes))), dim = 0).type(torch.bool)
mask2 = torch.triu(mask2, diagonal=1)
#print('seed',torch.seed())
mask_sq_fake = torch.ones((num_fake_nodes,num_fake_nodes))
mask_sq_fake = torch.tril(mask_sq_fake, diagonal = -1)
mask = torch.cat((mask,mask_sq_fake), dim=1).type(torch.bool)


params = load_params(feat_dim,budget,num_fake_nodes,average_feat_one)


idx_train, idx_val, idx_test = get_train_val_test_gcn(labels[:-num_fake_nodes], seed = 1)
idx_train = np.append(idx_train,np.arange(len(labels)-num_fake_nodes,len(labels)))
victim_nodes = get_victim_nodes(labels[idx_test],idx_test,target_class)
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0.5, weight_decay=5e-4, device=torch.device('cpu'))
surrogate.fit(features, adj, labels, idx_train, idx_val,retrain=False)

adj_unnorm = torch.clone(adj)
features_unnorm = torch.clone(features)
adj = sp.coo_matrix(adj.numpy())
adj = normalize(adj)
adj = torch.from_numpy(adj.todense())

features = sp.coo_matrix(features.numpy())
features = normalize(features)
features = torch.from_numpy(features.todense())

def train():
    generator = Generator(params['input_dim'], params['latent_dim'], params['hid_dim_gen'], params['budget'], params['feat_budget'], 0.3, params['acti_fn'], True, num_old_nodes, params['latent_dim']//2,(5*params['latent_dim'])//4,params['latent_dim']//16,(5*params['latent_dim'])//4,(5*params['latent_dim'])//4)
    generator_optimiser = optim.Adam(generator.parameters(), lr=10**(-3))
    final_run(generator,generator_optimiser,surrogate,labels)



def final_run(generator, generator_optimiser, surrogate,labels):
    feat_mask = torch.zeros(num_nodes, feat_dim)
    feat_mask[num_old_nodes: , :] = 1
    feat_mask = feat_mask.type(torch.bool)


    misclassification_loss = MisclassificationLoss(victim_nodes,num_fake_nodes)


    scheduler = optim.lr_scheduler.StepLR(generator_optimiser, step_size=100)
    

    best_acc = 100
    for epoch in range(params['EPOCHS']):
        
        print('Time: ',(time.time()-start_time))
    
        print("Epoch", epoch)
        generator_optimiser.zero_grad()

    
    
        new_adj, new_feat = generator(features, adj, mask,adj_o,features_o,mask_internal_edges)
        torch.manual_seed(1)
        surrogate = GCN(nfeat=new_feat.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0.5, weight_decay=5e-4, device=torch.device('cpu'))
        loss1 = surrogate.fit(new_feat, new_adj, labels, idx_train, idx_val)
        surr_test = surrogate.test(victim_nodes)
        torch.seed()
        embeddings = surrogate.predict(new_feat, new_adj)
        attack_loss = misclassification_loss(embeddings, features, new_feat,HIDDEN)
        
        print("Attack Loss", attack_loss)
        total_loss = 10*attack_loss
        total_loss.backward()
        generator_optimiser.step()
        scheduler.step()
        new_adj_s = new_adj.detach(); new_feat_s = new_feat.detach()
        new_adj = new_adj.detach(); new_feat = new_feat.detach()

        temp_adj = torch.where(mask2, new_adj, torch.full(new_adj.shape, -math.inf))
        temp_feat = torch.where(feat_mask, new_feat, torch.full(new_feat.shape, -math.inf))
        top_adj_inds = torch.topk(temp_adj.flatten(), params['budget']).indices
        unravel_adj_inds = np.array(np.unravel_index(top_adj_inds.numpy(), new_adj.shape)).T

        unravel_adj_inds_low = []
        for ind in unravel_adj_inds:
            unravel_adj_inds_low.append([ind[1], ind[0]])

        unravel_adj_inds = list(unravel_adj_inds)
        unravel_adj_inds.extend(unravel_adj_inds_low)
        unravel_adj_inds = np.array(unravel_adj_inds)
        top_feat_inds = torch.topk(temp_feat.flatten(), params['feat_budget']).indices
        unravel_feat_inds = np.array(np.unravel_index(top_feat_inds.numpy(), new_feat.shape)).T

        mask_complete = torch.zeros((num_old_nodes, num_old_nodes))
        mask_complete = torch.cat((mask_complete, torch.ones((num_old_nodes, num_fake_nodes))), dim = 1)
        mask_complete = torch.cat((mask_complete, torch.ones((num_fake_nodes, num_nodes))), dim = 0).type(torch.bool)
        new_adj = torch.where(mask_complete, torch.zeros(new_adj.shape), adj_unnorm)
        for i in range(len(unravel_adj_inds)):
            new_adj[unravel_adj_inds[i][0],unravel_adj_inds[i][1]] = 1
        for i in range(num_fake_nodes):
            new_adj[num_old_nodes+i][num_old_nodes+i] = 1

        new_feat = torch.where(feat_mask, torch.zeros(new_feat.shape), features_unnorm)
        for i in range(len(unravel_feat_inds)):
            new_feat[unravel_feat_inds[i][0],unravel_feat_inds[i][1]] = 1

        temp_mask = new_adj == 1
        print("Num ones adj final", torch.sum(temp_mask.type(torch.float)))
        temp_mask = new_feat == 1
        print("Num ones features final", torch.sum(temp_mask.type(torch.float)))
        
        torch.manual_seed(1)
        surrogate = GCN(nfeat=new_feat.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0.5, weight_decay=5e-4, device=torch.device('cpu'))
        
        loss1 = surrogate.fit(new_feat, new_adj, labels, idx_train, idx_val,retrain=False)

        surr_test2 = surrogate.test(victim_nodes)
        torch.seed()
        
        

        if surr_test2 < best_acc:
             best_acc=surr_test2
             print('Best Accuracy: ',best_acc,' Epoch: ',epoch)
             f = open(f"{str(args.budget)}/model_best.pickle", "wb")
             pickle.dump(generator, f)
             f.close()
             torch.save(new_adj,f'{str(args.budget)}/adj.pt')
             torch.save(new_feat,f'{str(args.budget)}/feat.pt')
             torch.save(new_adj_s,f'{str(args.budget)}/adj_s.pt')
             torch.save(new_feat_s,f'{str(args.budget)}/feat_s.pt')
        print({"misclassification loss": attack_loss, "surr_test":surr_test, "surr_test2":surr_test2})
    f = open(f"{str(args.budget)}/model_last.pickle", "wb")
    pickle.dump(generator, f)
    f.close()

train()
print('Time: ',(time.time()-start_time))
