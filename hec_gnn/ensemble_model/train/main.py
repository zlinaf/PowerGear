import imp
import random
import argparse
from typing import Union, Tuple
import numpy as np
import pandas as pd
import csv
import sys
import os
import __init__
from utils import dataloader
from hec_gnn.conv.Conv import HECConv
from utils.base_func import mape_loss,list_of_groups,split_dataset,generate_dataset,label_norm,lase_direction_enhance,masked_edge_index,masked_edge_attr,get_mean_and_std_overall,norm_overall
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, BatchNorm1d,LSTM 
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size, OptPairTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import ASAPooling, LEConv, global_mean_pool, global_add_pool, global_max_pool, JumpingKnowledge,SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_data_set = ["atax.pt","bicg.pt","mvt.pt","gemm.pt","gemver.pt","syrk.pt","syr2k.pt","k2mm.pt","k3mm.pt","ABC_1.pt","ABx_1.pt","AB_1.pt","Ax_1.pt"]
new_base_data_set = ["aA.pt","aAB.pt","aABplusbBC.pt","aAplusbB.pt","aAx.pt","AB_1.pt","AB_2.pt","ABC_1.pt","ABplusAC.pt","ABplusC.pt","ABplusCD.pt","ABx_1.pt","ABx_2.pt","AplusB.pt","Ax_1.pt","xy_1.pt"]


class HECConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dim, 
        overall_dim = 0, use_overall: bool = False, batch_norm: bool = False,drop_out = 0.5,pool_aggr = "add",overall_dim_large = 128,relations = 4,aggregate = "add",simple_JK = "last"):
        super(HECConvNet, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(HECConv(in_channels, hidden_channels,dim,num_relation=relations,aggr=aggregate))
        if use_overall:
            self.fc1 = Linear(hidden_channels+overall_dim_large , hidden_channels).cuda()
            self.fc2 = Linear(hidden_channels, 1).cuda()
            self.large_overall = Linear(overall_dim,overall_dim_large)
        else:
            self.fc1 = Linear(hidden_channels, hidden_channels//2).cuda()
            self.fc2 = Linear(hidden_channels//2, 1).cuda()
        if simple_JK == "cat":
            self.cat_fc = Linear(num_layers*hidden_channels,hidden_channels) 
        if simple_JK == 'lstm':
            assert in_channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                hidden_channels, (num_layers * hidden_channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * hidden_channels) // 2), 1)

        self.use_overall = use_overall
        self.drop_out = drop_out
        self.pool_aggr = pool_aggr
        self.overall_dim_large = overall_dim_large
        self.JK = simple_JK
        print("relations:",relations)
    def forward(self, data):
        x, edge_index, edge_attr, batch, overall_attr,edge_type = data.x, data.edge_index, data.edge_attr, data.node_batch, data.overall , data.edge_type
        h_list = [x]
        for i, conv in enumerate(self.convs):
            h = conv(h_list[i], edge_index,edge_weight = edge_attr,edge_type = edge_type )
            if i != self.num_layers - 1:
                h = h.relu()
                h = F.dropout(h, p=self.drop_out, training=self.training)
            h_list.append(h)
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim = 0), dim = 0)
        elif self.JK == 'cat':
            node_representation = torch.cat(h_list[1:], dim = 1)
            node_representation = self.cat_fc(node_representation)
        elif self.JK == 'max':
            node_representation = torch.stack(h_list[1:], dim=-1).max(dim=-1)[0]
        elif self.JK == 'lstm':
            node_representation = torch.stack(h_list[1:], dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(node_representation)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            node_representation = (node_representation * alpha.unsqueeze(-1)).sum(dim=1)
        if self.pool_aggr == "add":
            node_representation = global_add_pool(node_representation, batch)
        elif self.pool_aggr == "mean":
            node_representation = global_mean_pool(node_representation, batch)
        if self.use_overall:
            overall_attr = self.large_overall(overall_attr.view(node_representation.size(0), -1))
            overall_attr = overall_attr.relu()
            node_representation = torch.cat([node_representation, overall_attr], dim = -1)
        node_representation = self.fc1(node_representation)
        # x = self.bn1(x)
        node_representation = F.relu(node_representation)
        node_representation = F.dropout(node_representation, p = self.drop_out, training = self.training)
        node_representation = self.fc2(node_representation)
        x_return = torch.squeeze(node_representation)
        return x_return




#############################################
def model_train(model, train_loader, loss_func = 'mse'):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        batch_out = model(data)
        batch_out = batch_out.view(-1)
        batch_out = batch_out.reshape(len(batch_out))
        if loss_func == 'mse':
            loss = F.mse_loss(batch_out, data.y).float()
        elif loss_func == 'mape':
            loss = mape_loss(batch_out, data.y).float()
        elif loss_func == 'HuberLoss':
            loss  = F.smooth_l1_loss(batch_out, data.y,reduction="mean",beta = 4.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def model_test(model, loader):
    model.eval()
    with torch.no_grad():
        mae = 0
        mape = 0
        mse = 0
        y = []
        y_hat =[]
        residual = []
        column=['y','y_hat','residual']
        for data in loader:
            data = data.to(device)
            batch_out = model(data)
            batch_out = batch_out.view(-1)
            batch_out = batch_out.reshape(len(batch_out))
            mse += F.mse_loss(batch_out, data.y).float().item() * data.num_graphs # MSE
            mae += (batch_out * std - data.y * std).abs().sum().item()  # MAE
            mape += ((batch_out - data.y) / (data.y + mean / std)).abs().sum().item()  # MAPE
            y.extend(data.y.cpu().numpy().tolist())
            y_hat.extend(batch_out.cpu().detach().numpy().tolist())
            residual.extend((data.y-batch_out).cpu().detach().numpy().tolist())
        # print('pred.y:', batch_out)
        # print('data.y:', data.y)
        tem_dir = {"y": y, "y_hat": y_hat, "residual": residual}
        test_df = pd.DataFrame(tem_dir)
    return mse / len(loader.dataset), mae / len(loader.dataset), mape / len(loader.dataset), test_df


######################################################################
if __name__ == "__main__":
    ############################
    parser = argparse.ArgumentParser(description = 'training LASENet GNN for HLS power estimation', epilog = '')
    parser.add_argument('--layer_num', help = "number of conv layers", action = 'store', default = 3)
    parser.add_argument('--hidden_dim', help = "hidden layer dimension", action = 'store', default = 128)
    parser.add_argument('--batch_size', help = "batch size", action = 'store', default = 128)
    parser.add_argument('--learning_rate', help = "learning rate", action = 'store', default = 0.0005)
    parser.add_argument('--use_overall', help = "whether to use overall attr: 1 or 0", action = 'store', default = 1)
    parser.add_argument('--drop_out', help = "drop_out", action = 'store', default = 0.2)
    parser.add_argument('--weight_decay', help = "weight_decay", action = 'store', default = None)
    parser.add_argument('--relations', help = "relations", action = 'store', default = 4)
    parser.add_argument('--edge_dim', help = "edge_dim", action = 'store', default = 4)
    parser.add_argument('--overall_dim_large', help = "overall_dim_large", action = 'store', default = 128)
    parser.add_argument('--edge_feature', help = "edge_feature", action = 'store', default = 1)
    parser.add_argument('--node_feature', help = "node_feature", action = 'store', default = 1)
    parser.add_argument('--test_dataset', help = "hold out dataset for testing purpose", action = 'store', default = 'atax')
    parser.add_argument('--train_dataset', help = "train", action = 'store', default = 'all')
    parser.add_argument('--seed', help = "set seed", action = 'store', default = 10)
    parser.add_argument('--onevone', help = "onevone", action = 'store', default = 0)
    parser.add_argument('--aggr_type', help = "aggr_type", action = 'store', default = "add")
    parser.add_argument('--pool_type', help = "pool_type", action = 'store', default = "add")
    parser.add_argument('--k', help = "k_fold", action = 'store', default = 10)
    parser.add_argument('--fold_index', help = "k fold", action = 'store', default = 9)
    parser.add_argument('--loss_function', help = "loss:mape or mse", action = 'store', default = "mape")
    parser.add_argument('--JK', help = "Jumping Knowledge :last ,sum,max,lstm", action = 'store', default = "sum")
    args = parser.parse_args()

    ############################
    # model parameters
    num_conv_layers = int(args.layer_num)
    hidden_channels = int(args.hidden_dim)
    batch_size = int(args.batch_size)
    lr = float(args.learning_rate)
    drop_out = float(args.drop_out)
    use_overall = True if int(args.use_overall) > 0 else False
    edge_feature = True if int(args.edge_feature) > 0 else False
    node_feature = True if int(args.node_feature) > 0 else False
    onevone = True if int(args.onevone) > 0 else False
    test_dataset = args.test_dataset
    train_dataset = args.train_dataset
    seed_number = int(args.seed) if args.seed is not None else None
    if train_dataset == "multi":
        train_dataset = ["AB_1.pt","Ax_1.pt","xy_1.pt"]
    elif train_dataset == "all":
        train_dataset = "all"
    elif train_dataset == "base":
        train_dataset = "base"
    relations = int(args.relations)
    edge_dim = args.edge_dim
    overall_dim_large = args.overall_dim_large
    print('parameters:', num_conv_layers, hidden_channels, batch_size, lr, use_overall)
    print('dataset:', test_dataset)
    fold_index = int(args.fold_index)
    k = int(args.k)

    conv_type = HECConv # LASEConv / HECConv / LEConv / etc
    num_fc_layers = 2
    aggr_type = args.aggr_type
    pool_type = args.pool_type
    loss_func = args.loss_function
    simple_JK = args.JK
    mean = 0
    std = 1

    ############################
    PROJECT_ROOT  = __init__.ROOT_DIR 
    model_dir = os.path.abspath('')
    dataset_dir = '{a}/dataset/graph_sample'.format(a = PROJECT_ROOT)
    run_dir = '{a}/seed<{z}>/fold_index<{s}>/HECCovNet_multi_relation<{h}>_{b}_{c}_{d}_{e}_{f}_overall<{i}>_edge_feature<{j}>_nodes_feature<{k}>_onevone<{g}>_train_set<{l}>_aggregate<{m}>_pooling<{n}>_loss<{o}>_JK<{p}>_seed<{q}>_k<{r}>_fold_index<{s}>'.format(z = seed_number,a = model_dir, b = num_conv_layers, c = hidden_channels, d = batch_size, e = lr,f = drop_out,h = relations,i = overall_dim_large,j = edge_feature,k = node_feature,g =onevone, l = train_dataset,m = aggr_type , n = pool_type,o = loss_func,p = simple_JK,q = seed_number,r = k,s = fold_index)
    if use_overall:
        run_dir = run_dir + '_overall'
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    run_dir = '{a}/{b}_{c}_seed<{e}>'.format(a=run_dir, b=args.train_dataset,c=test_dataset,e = seed_number)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    
    flog = open('{}/run.log'.format(run_dir), 'a+', newline = '')
    
    ############################
    ############################
    dataset_name_list = os.listdir(dataset_dir) 
    test_dataset_name = test_dataset + '.pt'
    if args.train_dataset == "multi":
        train_dataset_name = train_dataset
        dataset_list = generate_dataset(dataset_dir,dataset_dir,train_dataset_name,print_info = False)
    elif args.train_dataset == "all":
        if test_dataset_name not in dataset_name_list:
            print('test_dataset = {} is not found in dataset directory'.format(test_dataset))
            assert(0)
        else:
            dataset_name_list.remove(test_dataset_name) 
        dataset_list = generate_dataset(dataset_dir,dataset_name_list,print_info = False)  
    elif args.train_dataset == "base":
        dataset_name_list = base_data_set
        if test_dataset_name not in dataset_name_list:
            print('test_dataset = {} is not found in dataset directory'.format(test_dataset))
            assert(0)
        else:
            dataset_name_list.remove(test_dataset_name) 
        dataset_list = generate_dataset(dataset_dir,dataset_name_list,print_info = False)  

    elif args.train_dataset == "new_base":
        dataset_name_list = new_base_data_set
        dataset_list = generate_dataset(dataset_dir,dataset_name_list,print_info = False) 

    else:
        train_dataset_name = train_dataset + '.pt'
        dataset_list = generate_dataset(dataset_dir,[train_dataset_name],print_info = False)
    
    train_set, val_set = split_dataset(dataset_list, shuffle = True,seed=seed_number,k = k,fold = fold_index)
    test_set = generate_dataset(dataset_dir,[test_dataset_name], print_info = False)
    print("type number:",set(train_set[0].edge_type.tolist()))
    train_set = lase_direction_enhance(train_set, enhance = False,edge_attr_set= edge_feature,nodes_attr_set=node_feature,onevone=onevone,relation=relations)           
    val_set = lase_direction_enhance(val_set, enhance = False,edge_attr_set= edge_feature,nodes_attr_set=node_feature,onevone=onevone,relation=relations)
    test_set = lase_direction_enhance(test_set, enhance = False,edge_attr_set= edge_feature,nodes_attr_set=node_feature,onevone=onevone,relation=relations)
    print("type number after dealing:",set(train_set[0].edge_type.tolist()))
    for data in test_set:
        if type(data.overall) == list:
            data.overall = torch.FloatTensor(data.overall)
        elif type(data.overall) != torch.Tensor:
            print('type(data.overall) != torch.Tensor or list: kernel_name = {}, prj_name = {}'.format(data.kernel_name, data.prj_name))
            assert(0)
    for data in val_set:
        if type(data.overall) == list:
            data.overall = torch.FloatTensor(data.overall)
        elif type(data.overall) != torch.Tensor:
            print('type(data.overall) != torch.Tensor or list: kernel_name = {}, prj_name = {}'.format(data.kernel_name, data.prj_name))
            assert(0)
    for data in train_set:
        if type(data.overall) == list:
            data.overall = torch.FloatTensor(data.overall)
        elif type(data.overall) != torch.Tensor:
            print('type(data.overall) != torch.Tensor or list: kernel_name = {}, prj_name = {}'.format(data.kernel_name, data.prj_name))
            assert(0)
    if use_overall:
        overall_mean, overall_std = get_mean_and_std_overall(train_set + val_set )
        norm_overall(train_set, overall_mean, overall_std)
        norm_overall(val_set, overall_mean, overall_std)
        norm_overall(test_set, overall_mean, overall_std)
        overall_dim = train_set[0].overall.size(0)
        print('overall_mean:', overall_mean)
        print('overall_std:', overall_std)
        flog.write('overall_mean: {}\n'.format(overall_mean))
        flog.write('overall_std: {}\n'.format(overall_std))
    else:
        overall_dim = 0

    print('train_set size = {}, val_set size = {}, test_set size = {}'.format(len(train_set), len(val_set), len(test_set)))
    flog.write('train_set size = {}, val_set size = {}, test_set size = {}\n'.format(len(train_set), len(val_set), len(test_set)))

    train_loader = dataloader.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 6)
    val_loader = dataloader.DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 6)
    test_loader = dataloader.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 6)

    ############################
    best_mape = 1
    if os.path.exists('{}/best_model.pt'.format(run_dir)):
        model = torch.load('{}/best_model.pt'.format(run_dir)).to(device)
        with open('{}/best_model.csv'.format(run_dir), 'r') as rfile:
            reader = csv.reader(rfile)
            last_line = list(reader)[-1]
            best_mape = float(last_line[2])
        print('loading best model with MAPE = {}'.format(best_mape))
        flog.write('loading best model with MAPE = {}\n'.format(best_mape))
    else:
        model = HECConvNet(in_channels = train_loader.dataset[0].num_node_features, hidden_channels = hidden_channels, 
            num_layers = num_conv_layers, dim = edge_dim, use_overall = use_overall, overall_dim = overall_dim,drop_out = drop_out,pool_aggr = pool_type,overall_dim_large=overall_dim_large,relations= relations,aggregate=aggr_type,simple_JK=simple_JK).to(device)
        print('constructing new model')
        flog.write('constructing new model\n')
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    print('model info: conv_type = {a}, num_conv_layers = {b}, in_channels = {c}; hidden_channels = {d}, batch_size = {e}, lr = {f}, aggr_type = {g}, pool_type = {h}, loss = {i}, use_overall = {j},drop_out = {k}'.format(
        a = conv_type, b = num_conv_layers, c = train_loader.dataset[0].num_node_features, d = hidden_channels, 
        e = batch_size, f = lr, g = aggr_type, h = pool_type, i = loss_func, j = use_overall,k = drop_out))
    flog.write('model info: conv_type = {a}, num_conv_layers = {b}, in_channels = {c}; hidden_channels = {d}, batch_size = {e}, lr = {f}, aggr_type = {g}, pool_type = {h}, loss = {i}, use_overall = {j},drop_out = {k}\n'.format(
        a = conv_type, b = num_conv_layers, c = train_loader.dataset[0].num_node_features, d = hidden_channels, 
        e = batch_size, f = lr, g = aggr_type, h = pool_type, i = loss_func, j = use_overall,k=drop_out))
    print('number of used overall attributes: {}'.format(overall_dim))
    flog.write('number of used overall attributes: {}\n'.format(overall_dim))

    ############################
    fmodel = open('{}/best_model.csv'.format(run_dir), 'a+', newline = '')
    fm_wr = csv.writer(fmodel)
    fm_wr.writerow(['epoch', 'training_mape', 'val_mape', 'test_mape'])
    best_epoch = 0

    for epoch in range(1, 2400+1):
        loss = model_train(model, train_loader, loss_func)
        val_mse, val_mae, val_mape, _ = model_test(model, val_loader)
        test_mse, test_mae, test_mape, _ = model_test(model, test_loader)
        print('Epoch: {:04d}, Train {}_loss: {:.4f}, Val MSE-MAE-MAPE: {:.1f} - {:.2f} - {:.4f}, Test MSE-MAE-MAPE: {:.1f} - {:.2f} - {:.4f}'.format(
            epoch, loss_func, loss, val_mse, val_mae, val_mape, test_mse, test_mae, test_mape))
        flog.write('Epoch: {:04d}, Train {}_loss: {:.4f}, Val MSE-MAE-MAPE: {:.1f} - {:.2f} - {:.4f}, Test MSE-MAE-MAPE: {:.1f} - {:.2f} - {:.4f}\n'.format(
            epoch, loss_func, loss, val_mse, val_mae, val_mape, test_mse, test_mae, test_mape))

        if best_mape > val_mape:
            best_mape = val_mape
            torch.save(model, '{}/best_model.pt'.format(run_dir))
            torch.save(model, '{}/best_model_{}.pt'.format(run_dir, epoch))
            print("INFO: save current best model: epoch = {}, val_mape = {}".format(epoch, best_mape))
            flog.write("INFO: save current best model: epoch = {}, val_mape = {}\n".format(epoch, best_mape))
            flog.flush()
            fm_wr.writerow([epoch, loss, val_mape, test_mape])
            fmodel.flush()
            
        if epoch % 400 == 0:
            torch.save(model.state_dict(), '{a}/epoch_{b}.pt'.format(a = run_dir, b = epoch))

    flog.close()
    fmodel.close()
