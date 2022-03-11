import re
import csv
import argparse
import xml.etree.ElementTree as ET
import networkx as nx
import graphviz as gvz
from ast import literal_eval as make_tuple

from op_extract import INVOCATION_NUM, PLOT_EDGE_LIMIT
from op_extract import df_graph_visualize, df_cc_trim, cdfg_fsm_node

op_act_out = [ "mux", "load", "read"] # trace seq: opnd_out
op_act_in1 = ["store"] # trace seq: opnd_1
op_act_in2 = ["write"] # trace seq: opnd_2
op_act_2 = ["sqrt", "fsqrt"] # tracer seq: opnd_1, opnd_out
op_act_3 = ["add", "sub", "mul", "div", "fadd", "fsub", "fmul", "fdiv", "icmp", "fcmp", "and", "or", "xor"] # tracer seq: opnd_1, opnd_2, opnd_out
op_act_select = ["select"] # tracer seq: opnd_2, opnd_3, opnd_out (opnd_1 is not needed)


###################################################################
def get_cdfg_fsm_dict(in_dir):
    cdfg_fsm_dict = dict()
    with open('{}/cdfg_fsm_node_dict.csv'.format(in_dir), 'r') as fop:
        rdop = csv.reader(fop)
        next(rdop, None)
        for row in rdop:
            nodeid, name, rtl_name, opcode, line_num, opid, c_step, latency, opnd_num, bw_out, bw_0, bw_1, bw_2, m_delay, from_node_set, to_node_set, instruction = \
                int(row[0]), row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16]
            cdfg_fsm_dict[nodeid] = cdfg_fsm_node(nodeid = nodeid, name = name, rtl_name = rtl_name, opcode = opcode, line_num = line_num, opid = opid, 
                c_step = c_step, latency = latency, opnd_num = opnd_num, bw_out = bw_out, bw_0 = bw_0, bw_1 = bw_1, bw_2 = bw_2, 
                m_delay = m_delay, from_node_set = from_node_set, to_node_set = to_node_set, instruction = instruction)
    return cdfg_fsm_dict


###################################################################
class node_info:
    def __init__(self, nodeid, op_cnt, hd_list = [0.0, 0.0, 0.0]):
        self.nodeid = nodeid
        self.op_cnt = op_cnt
        self.hd_list = hd_list


class edge_info:
    def __init__(self, edge_id, edge_src_id, edge_dst_id, src_id, dst_id, latency, op_cnt = 0, hd_list = list()):
        self.edge_id = edge_id
        self.edge_src_id = edge_src_id
        self.edge_dst_id = edge_dst_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.latency = latency
        self.op_cnt = op_cnt
        self.hd_list = hd_list


def act_trace(in_dir, out_dir, inv_num):
    ########################
    with open('{}/global_dict.csv'.format(in_dir), 'r') as rfile:
        reader = csv.reader(rfile)
        next(reader)
        for row in reader:
            if row[0] == 'latency_max':
                max_latency = float(row[1])
            elif row[0] == 'latency_min':
                min_latency = float(row[1])
        overall_latency = (max_latency + min_latency) / 2

    ########################
    node_act_dict = dict()
    with open('{}/../act_trace/hd_node.csv'.format(out_dir), 'r') as fh:
        frd = csv.reader(fh)
        next(frd, None)
        for row in frd:
            nodeid = row[0]
            op_cnt = int(row[1]) if row[1] != '' else 0
            if len(row) == 3:
                hd_list = [float(row[2]), 0.0, 0.0]
            elif len(row) == 4:
                hd_list = [float(row[2]), float(row[3]), 0.0]
            elif len(row) == 5:
                hd_list = [float(row[2]), float(row[3]), float(row[4])]
            else:
                print('CHECK: act_trace: node hamming distance size not correct: {}'.format(row))

            for i in range(len(hd_list)):
                hd_list[i] = hd_list[i] * op_cnt / overall_latency / inv_num

            node_act_dict[nodeid] = node_info(nodeid = nodeid, op_cnt = op_cnt, hd_list = hd_list)

    with open('{}/node_act_dict.csv'.format(out_dir), 'w+', newline = '') as wfile:
        fwr = csv.writer(wfile)
        title = ['nodeid', 'op_cnt', 'hd_0', 'hd_1', 'hd_2']
        fwr.writerow(title)
        for nodeid in node_act_dict:
            node = node_act_dict[nodeid]
            wr_line = [node.nodeid, node.op_cnt, node.hd_list[0], node.hd_list[1], node.hd_list[2]]
            fwr.writerow(wr_line)

    ########################
    edge_act_dict = dict()
    with open('{}/../preprocess/edge_obj_dict.csv'.format(out_dir), 'r') as fedge:
        frd = csv.reader(fedge)
        next(frd, None)
        for row in frd:
            edge_id = int(row[0])
            edge_src_id = int(row[1])
            edge_dst_id = int(row[2])
            src_id = row[3]
            dst_id = row[4]
            latency = float(row[5])
            edge_act_dict[edge_id] = edge_info(edge_id = edge_id, edge_src_id = edge_src_id, edge_dst_id = edge_dst_id,
                src_id = src_id, dst_id = dst_id, latency = latency, hd_list = [0.0, 0.0, 0.0])
    
    with open('{}/../act_trace/hd_edge.csv'.format(out_dir), 'r') as fhd:
        rdhd = csv.reader(fhd)
        next(rdhd, None)
        for row in rdhd:
            edge_id = int(row[0])
            op_cnt = int(row[1]) if row[1] != '' else 0
            if len(row) == 3:
                hd_list = [float(row[2]), 0.0, 0.0]
            elif len(row) == 4:
                hd_list = [float(row[2]), float(row[3]), 0.0]
            elif len(row) == 5:
                hd_list = [float(row[2]), float(row[3]), float(row[4])]
            else:
                print('CHECK: act_trace: edge hamming distance size not correct: {}'.format(row))

            for i in range(len(hd_list)):
                hd_list[i] = hd_list[i] * op_cnt / overall_latency / inv_num
            edge_act_dict[edge_id].op_cnt = op_cnt
            edge_act_dict[edge_id].hd_list = hd_list

    with open('{}/edge_act_dict.csv'.format(out_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        writer.writerow(['edge_id', 'edge_src_id', 'edge_dst_id', 'src_id', 'dst_id', 'latency', 'op_cnt', 'hd_0', 'hd_1', 'hd_2'])
        for edge_id in edge_act_dict:
            edge = edge_act_dict[edge_id]
            writer.writerow([edge.edge_id, edge.edge_src_id, edge.edge_dst_id, edge.src_id, edge.dst_id, edge.latency, 
                edge.op_cnt, edge.hd_list[0], edge.hd_list[1], edge.hd_list[2]])

    return node_act_dict, edge_act_dict, overall_latency


###################################################################
def is_str_num(str_in):
    try:
        float(str_in)
        return 1
    except ValueError:
        if '0x' in str_in:
            try:
                int(str_in, 16)
                return 1
            except ValueError:
                return 0
        else:
            return 0

def get_opnd_index(opcode, opnd, instruction):
    ins_split = re.split(' |,|\)', instruction)
    opnd_list = list()
    for item in ins_split:
        if '%' in item:
            opnd_list.append(item.strip('%').strip('\)'))
        elif is_str_num(item):
            opnd_list.append(item)
        elif item in ['true', 'false']:
            opnd_list.append(item)
    if opcode in op_act_out or opcode in op_act_in1:
        opnd_list = [opnd_list[0]]
    elif opcode in op_act_in2:
        opnd_list = [opnd_list[1]]
    elif opcode in op_act_2:
        opnd_list = [opnd_list[1], opnd_list[0]]
    elif opcode in op_act_3:
        opnd_list = [opnd_list[1], opnd_list[2], opnd_list[0]]
    elif opcode in op_act_select: # one situation is that the op connects to the mux signal of 'select', we neglect this case
        opnd_list = [opnd_list[2], opnd_list[3], opnd_list[0]]

    op_index = opnd_list.index(opnd) if opnd in opnd_list else -1
    return op_index


###################################################################
def act_annotate(in_dir, out_dir, DG, inv_num):
    CG = DG.copy()
    cdfg_fsm_dict = get_cdfg_fsm_dict(in_dir)
    node_act_dict, edge_act_dict, overall_latency = act_trace(in_dir, out_dir, inv_num)

    ####################
    for nodeid in CG.nodes:
        opcode = CG.nodes[nodeid]['opcode']
        latency = float(CG.nodes[nodeid]['latency'].strip('\"'))
        if nodeid in node_act_dict:
            act_node = node_act_dict[nodeid]
            CG.nodes[nodeid]['act_ratio'] = act_node.op_cnt * latency / overall_latency / inv_num
            CG.nodes[nodeid]['op_cnt'] = act_node.op_cnt
            CG.nodes[nodeid]['hd_0'] = act_node.hd_list[0]
            CG.nodes[nodeid]['hd_1'] = act_node.hd_list[1]
            CG.nodes[nodeid]['hd_2'] = act_node.hd_list[2]
            CG.nodes[nodeid]['hd_sum'] = act_node.hd_list[0] + act_node.hd_list[1] + act_node.hd_list[2]
        elif opcode not in ['phi', 'fexp']:
            print('CHECK: act_annotate: graph nodeid = {} with opcode = {} not found in node_act_dict'.format(nodeid, opcode))
            
    ###############################
    for src_id, dst_id in CG.edges: 
        #####################
        edge_id = int(CG[src_id][dst_id]['edge_src_id']) # trace output of source node as activity
        opcode = CG.nodes[src_id]['opcode']
        CG[src_id][dst_id]['src_act_ratio'] = edge_act_dict[edge_id].op_cnt / overall_latency / inv_num
        if len(edge_act_dict[edge_id].hd_list) == 0:
            edge_act = 0.0
        elif opcode in op_act_2:
            edge_act = edge_act_dict[edge_id].hd_list[1]
        elif opcode in op_act_3 or opcode in op_act_select:
            edge_act = edge_act_dict[edge_id].hd_list[2]
        else:
            edge_act = edge_act_dict[edge_id].hd_list[0]
        CG[src_id][dst_id]['src_activity'] = edge_act
        
        #####################
        edge_id = int(CG[src_id][dst_id]['edge_dst_id']) # trace sink node activity according to reg_flow
        opcode = CG.nodes[dst_id]['opcode']
        CG[src_id][dst_id]['dst_act_ratio'] = edge_act_dict[edge_id].op_cnt / overall_latency / inv_num
        CG[src_id][dst_id]['dst_activity'] = 0.0
        reg_flow = make_tuple(DG[src_id][dst_id]['reg_flow'].strip('\"').strip(']['))
        reg_flow = list(reg_flow) if type(reg_flow[0]) == tuple else [reg_flow]

        for item in reg_flow:
            dst_node_id = item[3]
            opnd_index = -1
            if is_str_num(dst_node_id) and int(dst_node_id) in cdfg_fsm_dict:
                if CG.nodes[src_id]['opcode'] in ['io_port_in', 'io_mem', 'internal_mem']:
                    opnd = item[1] # find corresponding source operand in sink node
                else:
                    opnd = item[0] # find corresponding source operand in sink node
                instruction = cdfg_fsm_dict[int(dst_node_id)].instruction
                opnd_index = get_opnd_index(opcode, opnd, instruction)
                if opnd_index != -1: # for multiple reg_flow, find the corresponding interface and break the loop
                    edge_act = edge_act_dict[edge_id].hd_list[opnd_index] if opnd_index != -1 and opnd_index < len(edge_act_dict[edge_id].hd_list) else 0.0
                    CG[src_id][dst_id]['dst_activity'] = edge_act
                    break

    for src_id, dst_id in CG.edges:
        if CG[src_id][dst_id]['dst_activity'] == 0 and CG[src_id][dst_id]['src_activity'] != 0:
            CG[src_id][dst_id]['dst_activity'] =  CG[src_id][dst_id]['src_activity']
        elif CG[src_id][dst_id]['dst_activity'] != 0 and CG[src_id][dst_id]['src_activity'] == 0:
            CG[src_id][dst_id]['src_activity'] = CG[src_id][dst_id]['dst_activity']
        
        if CG[src_id][dst_id]['dst_act_ratio'] == 0 and CG[src_id][dst_id]['src_act_ratio'] != 0:
            CG[src_id][dst_id]['dst_act_ratio'] = CG[src_id][dst_id]['src_act_ratio']
        elif CG[src_id][dst_id]['dst_act_ratio'] != 0 and CG[src_id][dst_id]['src_act_ratio'] == 0:
            CG[src_id][dst_id]['src_act_ratio'] = CG[src_id][dst_id]['dst_act_ratio']

    return CG, node_act_dict, edge_act_dict


###################################################################
def fan_in_out_compute(DG):
    for nodeid in DG.nodes:
        DG.nodes[nodeid]['fan_in'] = len(list(DG.predecessors(nodeid)))
        DG.nodes[nodeid]['fan_out'] = len(list(DG.successors(nodeid)))


def sum_hd_compute(DG): # activity appear in the same op with different time steps, sum up
    for nodeid in DG.nodes:
        hd_fan_in = 0.0
        hd_fan_out = 0.0
        if len(list(DG.predecessors(nodeid))) > 0:
            for pre_id in list(DG.predecessors(nodeid)):
                hd_fan_in += float(DG[pre_id][nodeid]['dst_activity'])
        if len(list(DG.successors(nodeid))) > 0:
            for suc_id in list(DG.successors(nodeid)):
                hd_fan_out += float(DG[nodeid][suc_id]['src_activity'])

        DG.nodes[nodeid]['hd_fan_in'] = hd_fan_in
        DG.nodes[nodeid]['hd_fan_out'] = hd_fan_out


###################################################################
def reconnect_reg_flow(pre_reg_flow, suc_reg_flow):
    updated_reg_flow = list()
    for pre_item in pre_reg_flow:
        pre_src_name, pre_dst_name, pre_src_id, pre_dst_id = pre_item[0], pre_item[1], pre_item[2], pre_item[3]
        for suc_item in suc_reg_flow:
            suc_src_name, suc_dst_name, suc_src_id, suc_dst_id = suc_item[0], suc_item[1], suc_item[2], suc_item[3]
            if pre_dst_name == suc_src_name:
                updated_reg_flow.append((pre_src_name, suc_dst_name, pre_src_id, suc_dst_id))
    
    if len(updated_reg_flow) < max(len(pre_reg_flow), len(suc_reg_flow)):
        print('CHECK: reconnect_reg_flow: reg_flow reconnect size not matched')
        print('pre_reg_flow: {}'.format(pre_reg_flow))
        print('suc_reg_flow: {}'.format(suc_reg_flow))
        print('updated_reg_flow: {}'.format(updated_reg_flow))
    return updated_reg_flow


def load_bypass(DG):
    rm_node_list = list()
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] in ['load', 'read', 'phi']:
            for pre_id in list(DG.predecessors(nodeid)):
                for suc_id in list(DG.successors(nodeid)):
                    if type(DG[pre_id][nodeid]['reg_flow']) == str:
                        pre_reg_flow = make_tuple(DG[pre_id][nodeid]['reg_flow'].strip('\"').strip(']['))
                        pre_reg_flow = list(pre_reg_flow) if type(pre_reg_flow[0]) == tuple else [pre_reg_flow]
                    else:
                        pre_reg_flow = DG[pre_id][nodeid]['reg_flow']
                    if type(DG[nodeid][suc_id]['reg_flow']) == str:
                        suc_reg_flow = make_tuple(DG[nodeid][suc_id]['reg_flow'].strip('\"').strip(']['))
                        suc_reg_flow = list(suc_reg_flow) if type(suc_reg_flow[0]) == tuple else [suc_reg_flow]
                    else:
                        suc_reg_flow = DG[nodeid][suc_id]['reg_flow']
                    reg_flow = reconnect_reg_flow(pre_reg_flow, suc_reg_flow)
                    DG.add_edge(pre_id, suc_id, src_activity = DG[pre_id][nodeid]['src_activity'], dst_activity = DG[nodeid][suc_id]['dst_activity'], 
                        src_act_ratio = DG[pre_id][nodeid]['src_act_ratio'], dst_act_ratio = DG[nodeid][suc_id]['dst_act_ratio'], 
                        edge_id = DG[pre_id][nodeid]['edge_id'], edge_src_id = DG[pre_id][nodeid]['edge_src_id'], 
                        edge_dst_id = DG[nodeid][suc_id]['edge_dst_id'], reg_flow = reg_flow)
            rm_node_list.append(nodeid)
    DG.remove_nodes_from(rm_node_list)


###################################################################
# if the control signal of 'select' (activity = 0) is found, delete this edge
# later trim the unimportant connected_components in the whole graph again
def select_edge_trim(DG):
    rm_edge_list = list()
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] == 'select':
            for pre_id in list(DG.predecessors(nodeid)):
                    if float(DG[pre_id][nodeid]['dst_activity']) < 0.000001:
                        rm_edge_list.append((pre_id, nodeid))

    DG.remove_edges_from(rm_edge_list)
    df_cc_trim(DG)


def edge_act_consistency_check(DG):
    for src_id, dst_id in DG.edges:
        if DG[src_id][dst_id]['dst_activity'] == 0 and DG[src_id][dst_id]['src_activity'] != 0:
            DG[src_id][dst_id]['dst_activity'] =  DG[src_id][dst_id]['src_activity']
        elif DG[src_id][dst_id]['dst_activity'] != 0 and DG[src_id][dst_id]['src_activity'] == 0:
            DG[src_id][dst_id]['src_activity'] = DG[src_id][dst_id]['dst_activity']
        
        if DG[src_id][dst_id]['dst_act_ratio'] == 0 and DG[src_id][dst_id]['src_act_ratio'] != 0:
            DG[src_id][dst_id]['dst_act_ratio'] = DG[src_id][dst_id]['src_act_ratio']
        elif DG[src_id][dst_id]['dst_act_ratio'] != 0 and DG[src_id][dst_id]['src_act_ratio'] == 0:
            DG[src_id][dst_id]['src_act_ratio'] = DG[src_id][dst_id]['dst_act_ratio']


###################################################################
def graph_node_feat_to_file(out_dir, DG):
    with open('{}/graph_node_feature.csv'.format(out_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title_1 = ['nodeid', 'node_id', 'optype', 'opcode', 'latency', 'delay', 'act_ratio', 'hd_0', 'hd_1', 'hd_2', 'hd_sum', # 'hd_out', 'hd_in',
            'hd_fan_in', 'hd_fan_out', 'merge_op_set', 'fan_in', 'fan_out']
        title_2 = ['rtl_id', 'rtl_name', 'lut', 'dsp', 'bram', 'ff', 'mem_words', 'mem_bits', 'mem_banks', 'mem_wxbitsxbanks', 'opnd_num', 
            'bw_out', 'bw_0', 'bw_1', 'bw_2', 'cop_set', 'line_num_set', 'latency_set']
        title = title_1 + title_2
        writer.writerow(title)
        for nodeid in DG.nodes:
            cnode = DG.nodes[nodeid]
            wr_line = [nodeid]
            for val_name in title[1:]:
                if val_name in cnode:
                    if type(cnode[val_name]) == str:
                        c_val = cnode[val_name].strip('\"')
                    else:
                        c_val = cnode[val_name]
                else:
                    c_val = 0
                wr_line += [c_val]
            writer.writerow(wr_line)


def graph_edge_feat_to_file(out_dir, DG):
    with open('{}/graph_edge_feature.csv'.format(out_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['edge_id', 'src_id', 'dst_id', 'src_activity', 'dst_activity', 'src_act_ratio', 'dst_act_ratio', 'reg_flow']
        writer.writerow(title)
        for src_id, dst_id in DG.edges:
            wr_line = [DG[src_id][dst_id]['edge_id'], src_id, dst_id]
            for val_name in title[3:]:
                if val_name in DG[src_id][dst_id]:
                    if type(DG[src_id][dst_id][val_name]) == str:
                        c_val = DG[src_id][dst_id][val_name].strip('\"')
                    else:
                        c_val = DG[src_id][dst_id][val_name]
                else:
                    c_val = 0
                wr_line += [c_val]
            writer.writerow(wr_line)


###################################################################
def feature_embed(in_dir, out_dir, inv_num):
    DG = nx.DiGraph(nx.drawing.nx_pydot.read_dot('{}/DG.dot'.format(in_dir)))
    CG, node_act_dict, edge_act_dict = act_annotate(in_dir, out_dir, DG, inv_num)
    fan_in_out_compute(CG)
    sum_hd_compute(CG)
    load_bypass(CG)
    select_edge_trim(CG)
    edge_act_consistency_check(CG)
    graph_node_feat_to_file(out_dir, CG)
    graph_edge_feat_to_file(out_dir, CG)
    nx.nx_pydot.write_dot(CG, '{}/DG_0.dot'.format(out_dir))
    if CG.number_of_edges() < PLOT_EDGE_LIMIT:
        df_graph_visualize(out_dir, 'DG_0', act_plot = True)


###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Graph activity annotation and write out to file', epilog = '')
    parser.add_argument('kernel_name', help = "input: kernel_name of HLS function")
    parser.add_argument('--src_path', required  = True, help = "directory path of input files", action = 'store')
    parser.add_argument('--dest_path', required  = True, help = "directory path of output files", action = 'store')
    args = parser.parse_args()

    feature_embed(args.src_path, args.dest_path, INVOCATION_NUM)
