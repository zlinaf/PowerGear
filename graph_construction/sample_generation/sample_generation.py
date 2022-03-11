import os
import time
import csv
import argparse
import collections
import pathlib
import numpy as np 
import networkx as nx
import sklearn
from sklearn.preprocessing import OneHotEncoder
import torch
import torch_geometric.data


######################################################################
pwr_status = {'total': 0, 'dynamic': 1}

numeric_item = ['hd_fan_in', 'hd_fan_out', 'hd_sum', 'act_ratio', 'op_cnt', 'latency', 'delay', 'fan_in', 'fan_out', 
    'lut', 'dsp', 'bram', 'ff', 'mem_words', 'mem_bits', 'mem_banks', 'mem_wxbitsxbanks', 'opnd_num', 'bw_out', 'bw_0', 'bw_1', 'bw_2']
categ_item = ['optype', 'opcode']

optype_categ = [['arith'], ['logic'], ['arbit'], ['io'], ['memory'], ['others']]
opcode_categ = [['fadd'], ['fsub'], ['fmul'], ['fdiv'], ['fsqrt'], ['fcmp'], ['fadd_fsub'], 
    ['store'], ['load'], ['load_store'], 
    ['add'], ['sub'], ['mul'], ['div'], ['sqrt'], 
    ['mux'], ['select'], 
    ['others']]


######################################################################
def gen_edge_type_dict():
    edge_type_dict = dict()
    edge_type_dict['arith'] = dict()
    edge_type_dict['others'] = dict()
    edge_type_dict['arith']['arith'] = 0
    edge_type_dict['arith']['others'] = 1
    edge_type_dict['others']['arith'] = 2
    edge_type_dict['others']['others'] = 3
    return edge_type_dict


######################################################################
def onehot_enc_gen():
    optype_enc = OneHotEncoder(handle_unknown = 'ignore')
    optype_enc.fit(optype_categ)
    opcode_enc = OneHotEncoder(handle_unknown = 'ignore')
    opcode_enc.fit(opcode_categ)
    return optype_enc, opcode_enc


def fill_blank_info(DG, numeric_item, categ_item):
    for nodeid in DG.nodes():
        for n_item in numeric_item:
            if n_item not in DG.nodes[nodeid]:
                DG.nodes[nodeid][n_item] = 0

        for o_item in categ_item:
            if o_item not in DG.nodes[nodeid]:
                DG.nodes[nodeid][o_item] = 'others'


#############################################################
def read_metadata(file_dir, prj_name):
    with open('{}/{}/preprocess/global_dict.csv'.format(file_dir, prj_name), 'r') as rfile:
        reader = csv.reader(rfile)
        for row in reader:
            if row[0] == 'clk_estimated':
                hls_clock = float(row[1])
            elif row[0] == 'latency_max':
                max_latency = float(row[1])
            elif row[0] == 'latency_min':
                min_latency = float(row[1])
            elif row[0] == 'lut':
                hls_lut = float(row[1])
            elif row[0] == 'ff':
                hls_ff = float(row[1])
            elif row[0] == 'dsp':
                hls_dsp = float(row[1])
            elif row[0] == 'bram':
                hls_bram = float(row[1])

    return hls_clock, max_latency, min_latency, hls_lut, hls_ff, hls_dsp, hls_bram


class metadata:
    def __init__(self, sample_name, clock, max_latency, min_latency, avg_latency, 
        lut, ff, dsp, bram, sf_lut, sf_ff, sf_dsp, sf_bram, sf_latency):
        self.sample_name = sample_name
        self.clock = clock
        self.max_latency = max_latency
        self.min_latency = min_latency
        self.avg_latency = avg_latency
        self.lut = lut
        self.ff = ff
        self.dsp = dsp
        self.bram = bram
        self.sf_latency = sf_latency
        self.sf_lut = sf_lut
        self.sf_ff = sf_ff
        self.sf_dsp = sf_dsp
        self.sf_bram = sf_bram


def get_metadata_feature(file_dir, prj_list):
    metadata_dict = dict()
    init = False
    baseline_latency, baseline_lut, baseline_ff, baseline_dsp, baseline_bram = -1, -1, -1, -1, -1
    
    for prj_name in prj_list:
        hls_clock, max_latency, min_latency, hls_lut, hls_ff, hls_dsp, hls_bram = read_metadata(file_dir, prj_name)
        latency = (max_latency + min_latency) / 2

        if init == False:
            print('baseline prj_name = {}'.format(prj_name))
            baseline_latency, baseline_lut, baseline_ff, baseline_dsp, baseline_bram = latency, hls_lut, hls_ff, hls_dsp, hls_bram
            init = True
        
        sf_latency = latency / baseline_latency
        sf_lut = hls_lut / baseline_lut
        sf_ff = hls_ff / baseline_ff
        sf_dsp = hls_dsp / baseline_dsp
        sf_bram = hls_bram / baseline_bram

        metadata_dict[prj_name] = metadata(sample_name = prj_name, clock = hls_clock, max_latency = max_latency, 
            min_latency = min_latency, avg_latency = latency, lut = hls_lut, ff = hls_ff, dsp = hls_dsp, bram = hls_bram, 
            sf_lut = sf_lut, sf_ff = sf_ff, sf_dsp = sf_dsp, sf_bram = sf_bram, sf_latency = sf_latency)

    return metadata_dict


######################################################################
def get_pwr(pwr_dir):
    pwr_dict = dict()
    with open('{}/power_measurement.csv'.format(pwr_dir), 'r') as rfile:
        reader = csv.reader(rfile)
        next(reader)
        for row in reader:
            if not row[0] in pwr_dict: # the units of total power (uW) and dynamic power (mW) is not the same, should make an alignment
                pwr_dict[row[0]] = list([float(row[1]) / 1000, float(row[3])]) 
            else:
                print('CHECK: power for design {} appears multiple times'.format(row[0]))
    return collections.OrderedDict(sorted(pwr_dict.items()))


######################################################################
def generate_dot(DG, optype_enc, opcode_enc, dot_store_path):
    pyg_DG = DG.__class__()
    pyg_DG.add_nodes_from(DG)
    pyg_DG.add_edges_from(DG.edges)
        
    for nodeid in DG.nodes():
        node_feat = list()
        for feat_item in numeric_item:
            if feat_item not in DG.nodes[nodeid]:
                print("ERROR: feat_item = {}, not in nodeid = {}".format(feat_item, nodeid))
                raise AssertionError()
            else:
                if type(DG.nodes[nodeid][feat_item]) == int:
                    node_feat.append(int(DG.nodes[nodeid][feat_item]))
                elif type(DG.nodes[nodeid][feat_item]) == str:
                    float_val = float(DG.nodes[nodeid][feat_item].strip('\"'))
                    node_feat.append(float_val)
                else:
                    print("ERROR: feat_item = {}, not with type str or int, with type {}".format(feat_item, type(DG.nodes[nodeid][feat_item])))
                    raise AssertionError()

        c_optype = DG.nodes[nodeid]['optype']
        c_opcode = DG.nodes[nodeid]['opcode']
        if c_opcode == 'fsub_fadd': c_opcode = 'fadd_fsub'
        elif c_opcode == 'store_load': c_opcode = 'load_store'
            
        onehot_optype = optype_enc.transform([[c_optype]]).toarray()
        onehot_optype = onehot_optype.reshape(np.shape(onehot_optype)[1])
        onehot_opcode = opcode_enc.transform([[c_opcode]]).toarray()
        onehot_opcode = onehot_opcode.reshape(np.shape(onehot_opcode)[1])
        
        node_feat = np.concatenate((node_feat, onehot_optype), axis = 0)
        node_feat = np.concatenate((node_feat, onehot_opcode), axis = 0)
        pyg_DG.nodes[nodeid]['x'] = list(node_feat)

    edge_type_dict = gen_edge_type_dict()
    for srcid, dstid in DG.edges():
        pyg_DG[srcid][dstid]['edge_attr'] = [float(DG[srcid][dstid]['src_activity'].strip('\"')), float(DG[srcid][dstid]['src_act_ratio'].strip('\"')), 
            float(DG[srcid][dstid]['dst_activity'].strip('\"')), float(DG[srcid][dstid]['dst_act_ratio'].strip('\"'))]
        if DG.nodes[srcid]['opcode'] in ['fadd', 'fsub', 'fsub_fadd', 'fadd_fsub', 'fmul', 'fdiv', 'fsqrt']:
            src_type = 'arith'
        else:
            src_type = 'others'
        if DG.nodes[dstid]['opcode'] in ['fadd', 'fsub', 'fsub_fadd', 'fadd_fsub', 'fmul', 'fdiv', 'fsqrt']:
            dst_type = 'arith'
        else:
            dst_type = 'others'
        
        pyg_DG[srcid][dstid]['edge_type'] = edge_type_dict[src_type][dst_type]

    nx.nx_pydot.write_dot(pyg_DG, dot_store_path)
    return pyg_DG


def generate_dataframe(DG, pwr_list, pwr_status, overall_attr, kernel_name, prj_name, df_store_path):
    data = {}

    for i, (_, feat_dict) in enumerate(DG.nodes(data = True)):
        for key, value in feat_dict.items():
            val_list = [float(val) for val in value]
            data[str(key)] = [val_list] if i == 0 else data[str(key)] + [val_list]
                
    for i, (_, _, feat_dict) in enumerate(DG.edges(data = True)):
        for key, value in feat_dict.items():
            if type(value) == str:
                data[key] = [[float(value.strip('\"'))]] if i == 0 else data[key] + [[float(value.strip('\"'))]]
            elif type(value) == int:
                data[key] = [value] if i == 0 else data[key] + [value]
            elif type(value) == list:
                data[key] = [value] if i == 0 else data[key] + [value]

    data['overall'] = [overall_attr.clock, overall_attr.max_latency, overall_attr.min_latency, overall_attr.avg_latency, 
        overall_attr.lut, overall_attr.ff, overall_attr.dsp, overall_attr.bram, overall_attr.sf_lut, overall_attr.sf_ff, 
        overall_attr.sf_dsp, overall_attr.sf_bram, overall_attr.sf_latency]
    data['y'] = pwr_list[pwr_status]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass
 
    edge_index = torch.LongTensor(list(DG.edges)).t().contiguous()
    data['edge_index'] = edge_index.view(2, -1)
    data['kernel_name'] = kernel_name
    data['prj_name'] = prj_name
    dataframe = torch_geometric.data.Data.from_dict(data)
    dataframe.num_nodes = DG.number_of_nodes()
    torch.save(dataframe, df_store_path)

    return dataframe


######################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Sample generation flow for one kernel', epilog = '')
    parser.add_argument('kernel_name', help = "input: kernel_name for feature extraction")
    parser.add_argument('--objective', help = "power prediction objective: total/dynamic", action = 'store', default = 'dynamic')
    parser.add_argument('--prj_dir', help = "directory of the output from the feature extraction stage", action = 'store', default = './')
    parser.add_argument('--pwr_dir', help = "directory of the file storing the power measurement", action = 'store', default = './')
    parser.add_argument('--out_dir', help = "directory to store the graph samples for training/testing", action = 'store', default = './')
    args = parser.parse_args()

    kernel_name = args.kernel_name
    objective = args.objective
    prj_dir = args.prj_dir
    pwr_dir = args.pwr_dir
    out_dir = args.out_dir
    prj_list = next(os.walk(prj_dir))[1]
    prj_list = sorted(prj_list, reverse = False)
    prj_cnt = len(prj_list)
    cur_dir = pathlib.Path(__file__).parent.absolute()
    if not os.path.isdir('{}/dot'.format(cur_dir)):
        os.mkdir('{}/dot'.format(cur_dir))
    if not os.path.isdir('{}/pyg_pt'.format(cur_dir)):
        os.mkdir('{}/pyg_pt'.format(cur_dir))
    if not os.path.isdir('{}/dot/{}'.format(cur_dir, kernel_name)):
        os.mkdir('{}/dot/{}'.format(cur_dir, kernel_name))
    if not os.path.isdir('{}/pyg_pt/{}'.format(cur_dir, kernel_name)):
        os.mkdir('{}/pyg_pt/{}'.format(cur_dir, kernel_name))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    print('##### start processing #####')
    print('##### kernel_name = {}, prediction objective = {} #####'.format(kernel_name, objective))

    optype_enc, opcode_enc = onehot_enc_gen()
    pwr_dict = get_pwr(pwr_dir)
    metadata_dict = get_metadata_feature(prj_dir, prj_list)
    dataframe_list = list()
    processed_cnt = 0

    for i, prj_name in enumerate(prj_list):
        if prj_name not in pwr_dict:
            print('INFO: #{} prj_name = {}, design is not complete, pwr_val not found, skip it'.format(i+1, prj_name))
            continue
        elif not os.path.exists('{}/{}/graph/DG_0.dot'.format(prj_dir, prj_name)):
            print('CHECK: #{} prj_name = {}, DG_0.dot file not found'.format(i+1, prj_name))
            continue
        else:
            processed_cnt += 1
            c_DG = nx.DiGraph(nx.drawing.nx_pydot.read_dot('{}/{}/graph/DG_0.dot'.format(prj_dir, prj_name)))
            c_DG = nx.convert_node_labels_to_integers(c_DG)
            fill_blank_info(c_DG, numeric_item, categ_item)
            pyg_DG = generate_dot(c_DG, optype_enc, opcode_enc, '{a}/dot/{b}/{b}_{c}.dot'.format(a = cur_dir, b = kernel_name, c = prj_name))
            generate_dataframe(pyg_DG, pwr_dict[prj_name], pwr_status[objective], metadata_dict[prj_name], kernel_name, prj_name,
                '{a}/pyg_pt/{b}/{b}_{c}.pt'.format(a = cur_dir, b = kernel_name, c = prj_name)) 
            dataframe = torch.load('{a}/pyg_pt/{b}/{b}_{c}.pt'.format(a = cur_dir, b = kernel_name, c = prj_name)) # try loading to see whether successfully written or not
            dataframe_list.append(dataframe)
            print('processing #{} - {}/{} @ {}: {}'.format(processed_cnt, i+1, prj_cnt, time.asctime(time.localtime(time.time())), dataframe))
    
    torch.save(dataframe_list, '{a}/{b}.pt'.format(a = out_dir, b = kernel_name))
    # df_list = torch.load('{a}/pyg_pt/{b}/{b}.pt'.format(a = cur_dir, b = kernel_name)) # try loading to see whether successfully written or not
    print('{}: finished processing {}/{} files at: {}'.format(kernel_name, processed_cnt, prj_cnt, time.asctime(time.localtime(time.time()))))
    print('{}: total number of samples extracted in assembled .pt file: {}'.format(kernel_name, len(dataframe_list)))
    
