import csv
import re
import argparse
import xml.etree.ElementTree as ET
import networkx as nx
import graphviz as gvz
from operator import itemgetter
from collections import defaultdict


INVOCATION_NUM = 8 # number of kernel invocations in a c main function
PLOT_EDGE_LIMIT = 400

iarith_opcode = ['add', 'sub', 'mul', 'div', 'sqrt']
farith_opcode = ['fadd', 'fsub', 'fmul', 'fdiv', 'fsqrt']
arith_opcode = iarith_opcode + farith_opcode
logic_opcode = ['icmp', 'fcmp', 'and', 'or', 'xor']
arbit_opcode = ['mux', 'select']
mem_opcode = ['store', 'load']

# for the purpose of adding dummy nodes
# 'write', 'store' and 'load' only trace the output operand, no need to take care here
# 'mux' and 'select' have variable op num, no need to consider here
opnd_num_dict = {'add': 3, 'sub': 3, 'mul': 3, 'div': 3, 'sqrt': 2, 
    'fadd': 3, 'fsub': 3, 'fmul': 3, 'fdiv': 3, 'fsqrt': 2, 
    'icmp': 3, 'fcmp': 3, 'and': 3, 'or': 3, 'xor': 3}

bypass_opcode_set = ['const', 'load', 'store', 'reg', 'zext', 'alloca', 'getelementptr', 'bitconcatenate', 'partselect', 'trunc', 'sext',
    'and', 'or', 'xor', 'phi', 'icmp', 'fcmp']


###################################################################
class cdfg_node:
    def __init__(self, nodeid, name, rtl_name, opcode, bitwidth, m_delay, line_num):
        self.nodeid = nodeid
        self.name = name
        if rtl_name == None:
            self.rtl_name = 'not_exist'
        else:
            self.rtl_name = rtl_name
        self.opcode = opcode
        if bitwidth == None:
            self.bitwidth = 0
        else:
            self.bitwidth = bitwidth
        if m_delay == None:
            self.m_delay = 0
        else:
            self.m_delay = m_delay
        if line_num == None:
            self.line_num = -1
        else:
            self.line_num = line_num

def get_cdfg_node(IRinfo):
    cdfg_node_dict = dict()
    op_set = set()
    for IRinfo in IRinfo.iter('cdfg'):
        for item in IRinfo.iter('item'):
            if item.find('Value') != None and item.find('opcode') != None:
                node_obj = item.find('Value').find('Obj')
                opcode = item.find('opcode').text
                nodeid = int(node_obj.find('id').text)
                name = node_obj.find('name').text
                rtl_name = node_obj.find('rtlName').text

                if item.find('Value').find('bitwidth') != None:
                    bitwidth = item.find('Value').find('bitwidth').text
                else:
                    bitwidth = 0

                if item.find('m_delay') != None:
                    m_delay = item.find('m_delay').text
                else:
                    m_delay = 0

                if node_obj.find('lineNumber') != None:
                    line_num = node_obj.find('lineNumber').text
                else:
                    line_num = -1

                cdfg_node_dict[nodeid] = cdfg_node(nodeid, name, rtl_name, opcode, bitwidth, m_delay, line_num)
                op_set.add(opcode)

    return cdfg_node_dict, sorted(list(op_set))

def cdfg_node_explore(info_dir, IRinfo):
    cdfg_node_dict, op_set = get_cdfg_node(IRinfo)

    with open('{}/cdfg_node_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['nodeid', 'name', 'rtl_name', 'opcode', 'bitwidth', 'm_delay', 'line_num']
        writer.writerow(title)
        for nodeid in cdfg_node_dict:
            node_objs = cdfg_node_dict[nodeid]
            wr_line = [node_objs.nodeid, node_objs.name, node_objs.rtl_name, node_objs.opcode, node_objs.bitwidth, node_objs.m_delay, node_objs.line_num]
            writer.writerow(wr_line)

    with open('{}/op_set.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        for opcode in op_set:
            wr_line = [opcode]
            writer.writerow(wr_line)

    return cdfg_node_dict, op_set


###################################################################
class fsm_node:
    def __init__(self, opid, nodeid, c_step, stage, latency, instruction, opnd_num, bw_out, bw_0, bw_1, bw_2):
        self.opid = opid
        self.nodeid = nodeid
        self.c_step = c_step
        self.stage = stage
        self.latency = latency
        self.instruction = instruction
        if bw_out == None:
            self.bw_out = 0
        else:
            self.bw_out = bw_out
        if bw_0 == None:
            self.bw_0 = 0
        else:
            self.bw_0 = bw_0
        if bw_1 == None:
            self.bw_1 = 0
        else:
            self.bw_1 = bw_1
        if bw_2 == None:
            self.bw_2 = 0
        else:
            self.bw_2 = bw_2
        if opnd_num == None:
            self.opnd_num = 0
        else:
            self.opnd_num = opnd_num

        self.rtl_name = ''
        self.opcode = ''
        self.from_node_set = set()
        self.to_node_set = set()


class port_node:
    def __init__(self, portid, name, direction, mem_type):
        self.portid = portid
        self.name = name
        self.direction = direction
        self.mem_type = mem_type


class df_trace_node:
    def __init__(self, src_name, src_opid, dst_name, dst_opid):
        self.src_name = src_name
        self.src_opid = src_opid
        self.dst_name = dst_name
        self.dst_opid = dst_opid


def get_fsm_node(FSMDinfo):
    fsm_node_dict = dict()
    op_node_mapping_dict = dict()
    for op in FSMDinfo.iter('operation'):
        opid = int(op.get('id'))
        c_step = op.get('st_id')
        stage = op.get('stage')
        latency = op.get('lat')

        node_list = op.findall('Node')
        for node in node_list:
            nodeid = int(node.get('id'))
            instruction = node.text.strip().partition(' ')[2].strip()
            bw_out = node.get('bw')
            bw_0 = node.get('op_0_bw')
            bw_1 = node.get('op_1_bw')
            bw_2 = node.get('op_2_bw')
            opnd_num = len(node.attrib) - 1
            if nodeid not in fsm_node_dict:
                fsm_node_dict[nodeid] = fsm_node(opid, nodeid, c_step, stage, latency, instruction, opnd_num, bw_out, bw_0, bw_1, bw_2)
            if opid not in op_node_mapping_dict:
                op_node_mapping_dict[opid] = nodeid
            else:
                print("CHECK: the same opid ({}) exists multiple times in FSMDinfo".format(opid))
    return fsm_node_dict, op_node_mapping_dict


def get_fsm_port(FSMDinfo):
    port_dict = dict()
    for port in FSMDinfo.iter('port'):
        portid = int(port.get('id'))
        port_name = port.get('name')
        direction = port.get('dir')
        mem_type = port.find('core').text
        port_dict[portid] = port_node(int(portid), port_name, direction, mem_type)
    return port_dict


def get_df_trace(FSMDinfo):
    df_trace_dict = dict()
    for dataflow in FSMDinfo.iter('dataflow'):
        src_name = dataflow.get('from')
        src_opid = int(dataflow.get('fromId'))
        dst_name = dataflow.get('to')
        dst_opid = int(dataflow.get('toId'))
        if dst_opid not in df_trace_dict:
            df_trace_dict[dst_opid] = list()
        df_trace_dict[dst_opid].append(df_trace_node(src_name, src_opid, dst_name, dst_opid))
    return df_trace_dict


def fsm_node_trace(fsm_node_dict, op_node_mapping_dict, df_trace_dict):
    for dst_opid in df_trace_dict:
        for df_node in df_trace_dict[dst_opid]:
            src_opid = df_node.src_opid
            if src_opid in op_node_mapping_dict and dst_opid in op_node_mapping_dict:
                from_nodeid = op_node_mapping_dict[src_opid]
                to_nodeid = op_node_mapping_dict[dst_opid]
                fsm_node_dict[from_nodeid].to_node_set.add(to_nodeid)
                fsm_node_dict[to_nodeid].from_node_set.add(from_nodeid)


def fsm_node_explore(info_dir, FSMDinfo):
    fsm_node_dict, op_node_mapping_dict = get_fsm_node(FSMDinfo)
    port_dict = get_fsm_port(FSMDinfo)
    df_trace_dict = get_df_trace(FSMDinfo)
    fsm_node_trace(fsm_node_dict, op_node_mapping_dict, df_trace_dict)

    with open('{}/fsm_node_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['opid', 'nodeid', 'c_step', 'stage', 'latency', 'opnd_num', 'bw_out', 'bw_0', 'bw_1', 'bw_2', 'from_node_set', 'to_node_set', 'instruction']
        writer.writerow(title)
        for nodeid in fsm_node_dict:
            node = fsm_node_dict[nodeid]
            wr_line = [node.opid, node.nodeid, node.c_step, node.stage, node.latency, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, 
                str(sorted(node.from_node_set)).strip('[]'), str(sorted(node.to_node_set)).strip('[]'), node.instruction]
            writer.writerow(wr_line)

    with open('{}/port_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['portid', 'name', 'direction', 'mem_type']
        writer.writerow(title)
        for pid in port_dict:
            pnode = port_dict[pid]
            wr_line = [pnode.portid, pnode.name, pnode.direction, pnode.mem_type]
            writer.writerow(wr_line)

    return fsm_node_dict, op_node_mapping_dict


###################################################################
class cdfg_fsm_node:
    def __init__(self, nodeid, name, rtl_name, opcode, line_num, opid, c_step, latency, instruction, opnd_num, bw_out, bw_0, bw_1, bw_2, m_delay, from_node_set, to_node_set):
        self.nodeid = nodeid
        self.name = name
        self.rtl_name = rtl_name
        self.opcode = opcode
        self.line_num = line_num
        self.opid = opid
        self.c_step = c_step
        self.latency = latency
        self.instruction = instruction
        self.opnd_num = opnd_num
        self.bw_out = bw_out
        self.bw_0 = bw_0
        self.bw_1 = bw_1
        self.bw_2 = bw_2
        self.m_delay = m_delay
        self.from_node_set = from_node_set
        self.to_node_set = to_node_set


def get_cdfg_fsm_node(cdfg_node_dict, fsm_node_dict):
    cdfg_fsm_node_dict = dict()
    for nodeid in cdfg_node_dict:
        if nodeid in fsm_node_dict:
            node = cdfg_node_dict[nodeid]
            cop = fsm_node_dict[nodeid]
            cdfg_fsm_node_dict[nodeid] = cdfg_fsm_node(nodeid, node.name, node.rtl_name, node.opcode, node.line_num, cop.opid, cop.c_step, cop.latency, cop.instruction, \
                cop.opnd_num, cop.bw_out, cop.bw_0, cop.bw_1, cop.bw_2, node.m_delay, cop.from_node_set, cop.to_node_set)

    return cdfg_fsm_node_dict


def cdfg_fsm_node_explore(info_dir, cdfg_node_dict, fsm_node_dict):
    cdfg_fsm_node_dict = get_cdfg_fsm_node(cdfg_node_dict, fsm_node_dict)
    with open('{}/cdfg_fsm_node_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['nodeid', 'name', 'rtl_name', 'opcode', 'line_num', 'opid', 'c_step', 'latency', 'opnd_num', 'bw_out', 'bw_0', 'bw_1', 'bw_2', 'm_delay', 'from_node_set', 'to_node_set', 'instruction']
        writer.writerow(title)
        for nodeid in cdfg_fsm_node_dict:
            node = cdfg_fsm_node_dict[nodeid]
            wr_line = [node.nodeid, node.name, node.rtl_name, node.opcode, node.line_num, node.opid, node.c_step, node.latency, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, \
                node.bw_2, node.m_delay, str(sorted(node.from_node_set)).strip('[]'), str(sorted(node.to_node_set)).strip('[]'), node.instruction]
            writer.writerow(wr_line)

    return cdfg_fsm_node_dict


def get_df_reg(FSMDinfo, op_node_mapping_dict, cdfg_fsm_node_dict):
    df_reg_dict = defaultdict(lambda: defaultdict(tuple))
    for dataflow in FSMDinfo.iter('dataflow'):
        src_name = dataflow.get('from')
        src_opid = int(dataflow.get('fromId'))
        dst_name = dataflow.get('to')
        dst_opid = int(dataflow.get('toId'))

        if src_opid in op_node_mapping_dict and dst_opid in op_node_mapping_dict:
            src_nodeid = op_node_mapping_dict[src_opid]
            dst_nodeid = op_node_mapping_dict[dst_opid]
        else:
            continue

        if not df_reg_dict[src_nodeid][dst_nodeid]:
            if dst_nodeid in cdfg_fsm_node_dict:
                cdfg_fsm_node = cdfg_fsm_node_dict[dst_nodeid]
                if cdfg_fsm_node.opcode == 'store': # for 'store'/'write', need to change the dst_name to the first operand instead of output = StgVal
                    dst_name = re.split(' |,', cdfg_fsm_node.instruction.strip(''))[2].strip('%')
                elif cdfg_fsm_node.opcode == 'write':
                    dst_name = re.split(' |,', cdfg_fsm_node.instruction.strip(''))[6].strip('%').strip('\)')
            df_reg_dict[src_nodeid][dst_nodeid] = (src_name, dst_name, src_nodeid, dst_nodeid)
        elif df_reg_dict[src_nodeid][dst_nodeid][0] != src_name:
            print('CHECK: multiple src_nodeid = {} to dst_nodeid = {} with different src_name = {}'.format(src_nodeid, dst_nodeid, src_name))
            assert(0)
    return df_reg_dict


###################################################################
class component_rsc:
    def __init__(self, cname, coptype):
        self.cname = cname
        self.coptype = coptype
        self.module = ''
        self.operation = ''
        self.bram_18k = 0
        self.dsp = 0
        self.ff = 0
        self.lut = 0

        self.mem_words = 0
        self.mem_bits = 0
        self.mem_banks = 0
        self.mem_wxbitsxbanks = 0

        self.ff_depth = 0
        self.ff_bits = 0
        self.ff_size = 0

        self.bitwidth_p0 = 0
        self.bitwidth_p1 = 0

        self.mux_inputsize = 0
        self.mux_bits = 0
        self.mux_totalbits = 0

        self.reg_bits = 0
        self.reg_const_bits = 0


def get_component_rsc(RSCinfo):  #some components appear in different types, add up in this case
    rsc_dict = dict()
    for section in RSCinfo.findall('section'):
        if section.get('name') == 'Utilization Estimates':
            for item in section.findall('item'):
                if item.get('name') == 'Detail':
                    for subitem in item.find('section').findall('item'):
                        if subitem.get('name') == 'Instance':
                            for column in subitem.iter('column'):
                                name = column.get('name')
                                uti_list = column.text.split(', ')
                                if name in rsc_dict:
                                    rsc_dict[name].coptype = rsc_dict[name].coptype + '_instance'
                                    if rsc_dict[name].modul == '':
                                        rsc_dict[name].module = uti_list[0]
                                    else:
                                        rsc_dict[name].module = rsc_dict[name].module + '_' + uti_list[0]
                                else:
                                    rsc_dict[name] = component_rsc(name, 'instance')
                                    rsc_dict[name].module = uti_list[0]

                                rsc_dict[name].bram_18k += int(uti_list[1])
                                rsc_dict[name].dsp += int(uti_list[2])
                                rsc_dict[name].ff += int(uti_list[3])
                                rsc_dict[name].lut += int(uti_list[4])

                        elif subitem.get('name') == 'Memory':
                            for column in subitem.iter('column'):
                                name = column.get('name')
                                uti_list = column.text.split(', ')
                                if name in rsc_dict:
                                    rsc_dict[name].coptype = rsc_dict[name].coptype + '_memory'
                                    if rsc_dict[name].modul == '':
                                        rsc_dict[name].module = uti_list[0]
                                    else:
                                        rsc_dict[name].module = rsc_dict[name].module + '_' + uti_list[0]
                                else:
                                    rsc_dict[name] = component_rsc(name, 'memory')
                                    rsc_dict[name].module = uti_list[0]

                                rsc_dict[name].bram_18k += int(uti_list[1])
                                rsc_dict[name].ff += int(uti_list[2])
                                rsc_dict[name].lut += int(uti_list[3])
                                rsc_dict[name].mem_words += int(uti_list[4])
                                rsc_dict[name].mem_bits += int(uti_list[5])
                                rsc_dict[name].mem_banks += int(uti_list[6])
                                rsc_dict[name].mem_wxbitsxbanks += int(uti_list[7])

                        elif subitem.get('name') == 'FIFO':
                            for column in subitem.iter('column'):
                                name = column.get('name')
                                uti_list = column.text.split(', ')
                                if name in rsc_dict:
                                    rsc_dict[name].coptype = rsc_dict[name].coptype + '_fifo'
                                else:
                                    rsc_dict[name] = component_rsc(name, 'memory')

                                rsc_dict[name].bram_18k += int(uti_list[0])
                                rsc_dict[name].ff += int(uti_list[1])
                                rsc_dict[name].lut += int(uti_list[2])
                                rsc_dict[name].ff_depth += int(uti_list[3])
                                rsc_dict[name].ff_bits += int(uti_list[4])
                                rsc_dict[name].ff_size += int(uti_list[5])

                        elif subitem.get('name') == 'Expression':
                            for column in subitem.iter('column'):
                                name = column.get('name')
                                uti_list = column.text.split(', ')
                                if name in rsc_dict:
                                    rsc_dict[name].coptype = rsc_dict[name].coptype + '_expression'
                                    if rsc_dict[name].operation == '':
                                        rsc_dict[name].operation = uti_list[0]
                                    else:
                                        rsc_dict[name].operation = rsc_dict[name].operation + '_' + uti_list[0]
                                else:
                                     rsc_dict[name] = component_rsc(name, 'expression')
                                     rsc_dict[name].operation = uti_list[0]

                                rsc_dict[name].dsp += int(uti_list[1])
                                rsc_dict[name].ff += int(uti_list[2])
                                rsc_dict[name].lut += int(uti_list[3])
                                if int(rsc_dict[name].bitwidth_p0) == 0:
                                    rsc_dict[name].bitwidth_p0 = uti_list[4]
                                else:
                                    rsc_dict[name].bitwidth_p0 = rsc_dict[name].bitwidth_p0 + '_' + uti_list[4]
                                    print('ERROR: check expression bitwidth_p0 {}'.format(name))
                                if int(rsc_dict[name].bitwidth_p1) == 0:
                                    rsc_dict[name].bitwidth_p1 = uti_list[5]
                                else:
                                    rsc_dict[name].bitwidth_p1 = rsc_dict[name].bitwidth_p1 + '_' + uti_list[5]
                                    print('ERROR: check expression bitwidth_p1 {}'.format(name))
                        
                        elif subitem.get('name') == 'Multiplexer':
                            for column in subitem.iter('column'):
                                name = column.get('name')
                                uti_list = column.text.split(', ')
                                if name in rsc_dict:
                                    rsc_dict[name].coptype = rsc_dict[name].coptype + '_multiplexer'
                                else:
                                    rsc_dict[name] = component_rsc(name, 'multiplexer')

                                rsc_dict[name].lut += int(uti_list[0])
                                rsc_dict[name].mux_inputsize += int(uti_list[1])
                                rsc_dict[name].mux_bits += int(uti_list[2])
                                rsc_dict[name].mux_totalbits += int(uti_list[3])

                        elif subitem.get('name') == 'Register':
                            for column in subitem.iter('column'):
                                name = column.get('name')
                                uti_list = column.text.strip().split(', ')
                                if name in rsc_dict:
                                    rsc_dict[name].coptype = rsc_dict[name].coptype + '_register'
                                else:
                                    rsc_dict[name] = component_rsc(name, 'register')

                                rsc_dict[name].ff += int(uti_list[0])
                                rsc_dict[name].lut += int(uti_list[1])
                                rsc_dict[name].reg_bits += int(uti_list[2])
                                rsc_dict[name].reg_const_bits += int(uti_list[3])
    return rsc_dict
                        

def component_rsc_explore(info_dir, RSCinfo):
    rsc_dict = get_component_rsc(RSCinfo)

    with open('{}/rsc_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['name', 'coptype', 'module', 'operation', 'bram_18k', 'dsp', 'ff', 'lut', 'mem_words', 'mem_bits', 'mem_banks', 'mem_wxbitsxbanks', \
            'ff_depth', 'ff_bits', 'ff_size', 'bitwidth_p0', 'bitwidth_p1', 'mux_inputsize', 'mux_bits', 'mux_totalbits', 'reg_bits', 'reg_const_bits']
        writer.writerow(title)
        for name in rsc_dict:
            component = rsc_dict[name]
            wr_line = [component.cname, component.coptype, component.module, component.operation, component.bram_18k, \
                component.dsp, component.ff, component.lut, component.mem_words, component.mem_bits, component.mem_banks, component.mem_wxbitsxbanks, \
                    component.ff_depth, component.ff_bits, component.ff_size, component.bitwidth_p0, component.bitwidth_p1, component.mux_inputsize, \
                        component.mux_bits, component.mux_totalbits, component.reg_bits, component.reg_const_bits]
            writer.writerow(wr_line)
            
    return rsc_dict


###################################################################
class c_operator:
    def __init__(self, nodeid, name, rtl_id, rtl_name, optype, opcode, line_num, c_step, latency, instruction, opnd_num, bw_out, bw_0, bw_1, bw_2, m_delay):
        self.nodeid = nodeid
        self.name = name
        self.rtl_id = rtl_id
        self.rtl_name = rtl_name
        self.optype = optype
        self.opcode = opcode
        self.line_num = line_num
        self.c_step = c_step
        self.latency = latency
        self.instruction = instruction
        self.opnd_num = opnd_num
        self.bw_out = bw_out
        self.bw_0 = bw_0
        self.bw_1 = bw_1
        self.bw_2 = bw_2
        self.m_delay = m_delay


class rtl_operator:
    def __init__(self, rtl_name, rtl_id, opnd_num, bw_out, bw_0, bw_1, bw_2, optype, opcode, delay):
        self.rtl_name = rtl_name
        self.rtl_id = rtl_id
        self.reg_name = []
        self.opnd_num = opnd_num
        self.bw_out = bw_out
        self.bw_0 = bw_0
        self.bw_1 = bw_1
        self.bw_2 = bw_2
        self.lut = 0
        self.dsp = 0
        self.bram = 0
        self.ff = 0
        self.mem_words = 0
        self.mem_bits = 0
        self.mem_banks = 0
        self.mem_wxbitsxbanks = 0
        self.optype = optype # arith/logic/memory/aribit
        self.opcode = opcode
        self.cop_set = set()
        self.line_num_set = set()
        self.comp_name = 'not_exist'
        self.latency_set = set()
        self.latency = 0
        self.delay = delay

    def add_reg_name(self, reg_name):
        self.reg_name.append(reg_name) 

    def add_cop(self, cop):
        self.cop_set.add(cop)

    def add_line_num(self, line_num):
        self.line_num_set.add(line_num)

    def add_latency(self, lat):
        self.latency_set.add(lat)
        self.latency = sum(list(self.latency_set)) / len(list(self.latency_set))


def get_arith_logic_arbit_op(cdfg_fsm_node_dict, rsc_dict, cop_dict, rop_dict, rtl_id):
    for nodeid in cdfg_fsm_node_dict:
        node = cdfg_fsm_node_dict[nodeid]
        if (node.opcode in arith_opcode) or (node.opcode in arbit_opcode) or (node.opcode in logic_opcode): # arithmetic/arbit/logic operators
            if node.opcode in arith_opcode:
                optype = 'arith'
            elif node.opcode in logic_opcode:
                optype = 'logic'
            elif node.opcode in arbit_opcode:
                optype = 'arbit'

            if node.rtl_name in rop_dict:
                rop_dict[node.rtl_name].add_cop(int(nodeid))
                rop_dict[node.rtl_name].add_line_num(int(node.line_num))
                rop_dict[node.rtl_name].add_latency(int(node.latency))
                if not node.opcode in rop_dict[node.rtl_name].opcode:
                    rop_dict[node.rtl_name].opcode = rop_dict[node.rtl_name].opcode + '_' + node.opcode
                if not rop_dict[node.rtl_name].optype == optype:
                    print('CHECK: one rtl_name ({}) has multiple optypes: {} and {}.'.format(node.rtl_name, rop_dict[node.rtl_name].optype, optype))
                if rop_dict[node.rtl_name].delay != node.m_delay:
                    print('CHECK: rtl_name ({}) has multiple delay values: {} and {}.'.format(node.rtl_name, rop_dict[node.rtl_name].delay, node.m_delay))

                cop_dict[nodeid] = c_operator(node.nodeid, node.name, rop_dict[node.rtl_name].rtl_id, node.rtl_name, optype, node.opcode, node.line_num, \
                    node.c_step, node.latency, node.instruction, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, node.m_delay)
            else:
                rop_dict[node.rtl_name] = rtl_operator(node.rtl_name, rtl_id, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, \
                    optype, node.opcode, node.m_delay)
                cop_dict[nodeid] = c_operator(node.nodeid, node.name, rtl_id, node.rtl_name, optype, node.opcode, node.line_num, node.c_step, \
                    node.latency, node.instruction, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, node.m_delay)
                rtl_id = rtl_id + 1
                if node.rtl_name not in rsc_dict:
                    print('CHECK: node.rtl_name ({}) not in rsc_dict, may be simple operation'.format(node.rtl_name))
                    continue
                rop_dict[node.rtl_name].lut = rsc_dict[node.rtl_name].lut
                rop_dict[node.rtl_name].dsp = rsc_dict[node.rtl_name].dsp
                rop_dict[node.rtl_name].bram = rsc_dict[node.rtl_name].bram_18k
                rop_dict[node.rtl_name].ff = rsc_dict[node.rtl_name].ff
                rop_dict[node.rtl_name].add_cop(int(nodeid))
                rop_dict[node.rtl_name].add_line_num(int(node.line_num))
                rop_dict[node.rtl_name].add_latency(int(node.latency))
                if (node.opcode in iarith_opcode) or (node.opcode in arbit_opcode) or (node.opcode in logic_opcode):
                    if 'expression' in rsc_dict[node.rtl_name].coptype:
                        rop_dict[node.rtl_name].bw_0 = rsc_dict[node.rtl_name].bitwidth_p0
                        rop_dict[node.rtl_name].bw_1 = rsc_dict[node.rtl_name].bitwidth_p1

            if node.rtl_name == 'not_exist':
                print('CHECK: node.rtl_name ({}) does not exist.'.format(nodeid))

    return rtl_id


def get_mem_op(IRinfo, cdfg_fsm_node_dict, rsc_dict, cop_dict, rop_dict, rtl_id):
    optype = 'memory'
    for dp_mem_port_nodes in IRinfo.iter('dp_mem_port_nodes'):
        for memitem in dp_mem_port_nodes.findall('item'):
            rtl_name = memitem.find('first').find('first').text
            if rtl_name not in rsc_dict:
                if '{}_U'.format(rtl_name) in rsc_dict:
                    rtl_name = '{}_U'.format(rtl_name)
                else:
                    # print('CHECK: rtl_name ({}) and rtl_name_U not in rsc_dict. This is IO port.'.format(rtl_name))
                    continue 
            else:
                print('CHECK: memory rtl_name ({}) directly appear in rsc_dict'.format(rtl_name))

            for nodeitem in memitem.find('second').findall('item'):
                nodeid = int(nodeitem.text)
                if nodeid in cdfg_fsm_node_dict:
                    node = cdfg_fsm_node_dict[nodeid]
                    if rtl_name in rop_dict:
                        rop_dict[rtl_name].add_cop(int(nodeid))
                        rop_dict[rtl_name].add_line_num(int(node.line_num))
                        rop_dict[rtl_name].add_latency(int(node.latency))
                        if not node.opcode in rop_dict[rtl_name].opcode:
                            rop_dict[rtl_name].opcode = rop_dict[rtl_name].opcode + '_' + node.opcode
                        if not rop_dict[rtl_name].optype == optype:
                            print('CHECK: one rtl_name ({}) has multiple optypes: {} and {}.'.format(rtl_name, rop_dict[rtl_name].optype, optype))
                        if rop_dict[rtl_name].delay != node.m_delay:
                            print('CHECK: rtl_name ({}) has multiple delay values: {} and {}.'.format(node.rtl_name, rop_dict[node.rtl_name].delay, node.m_delay))

                        cop_dict[nodeid] = c_operator(node.nodeid, node.name, rop_dict[rtl_name].rtl_id, rtl_name, optype, node.opcode, node.line_num, node.c_step, \
                            node.latency, node.instruction, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, node.m_delay)
                    else:
                        rop_dict[rtl_name] = rtl_operator(rtl_name, rtl_id, node.bw_out, node.opnd_num, node.bw_0, node.bw_1, node.bw_2, \
                            optype, node.opcode, node.m_delay)
                        cop_dict[nodeid] = c_operator(node.nodeid, node.name, rtl_id, rtl_name, optype, node.opcode, node.line_num, node.c_step, \
                            node.latency, node.instruction, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, node.m_delay)
                        rtl_id = rtl_id + 1
                        rop_dict[rtl_name].lut = rsc_dict[rtl_name].lut
                        rop_dict[rtl_name].dsp = rsc_dict[rtl_name].dsp
                        rop_dict[rtl_name].bram = rsc_dict[rtl_name].bram_18k
                        rop_dict[rtl_name].ff = rsc_dict[rtl_name].ff
                        rop_dict[rtl_name].mem_words = rsc_dict[rtl_name].mem_words
                        rop_dict[rtl_name].mem_bits = rsc_dict[rtl_name].mem_bits
                        rop_dict[rtl_name].mem_banks = rsc_dict[rtl_name].mem_banks
                        rop_dict[rtl_name].mem_wxbitsxbanks = rsc_dict[rtl_name].mem_wxbitsxbanks
                        rop_dict[rtl_name].add_cop(int(nodeid))
                        rop_dict[rtl_name].add_line_num(int(node.line_num))
                        rop_dict[rtl_name].add_latency(int(node.latency))
                else:
                    print('CHECK: get_mem_op: nodeid ({}) not in cdfg_fsm_node_dict'.format(nodeid))
    return rtl_id


# NOTE: BRAM io ports opcode = 'load'/'store'
#       REG/FIFO io ports opcode = 'read'/'write' (HLS-added opcode and functions, need external define in llvm annotation)
def get_io_op(IRinfo, cdfg_fsm_node_dict, rsc_dict, cop_dict, rop_dict, rtl_id):
    optype = 'io'
    for dp_port_io_nodes in IRinfo.iter('dp_port_io_nodes'):
        for ioitem in dp_port_io_nodes.findall('item'):
            rtl_name = ioitem.find('first').text
            if rtl_name.count('(') > 1:
                print('CHECK: dp_port_io_nodes item has name with multiple brackets.')
            else:
                start = rtl_name.find('(')
                if start != -1:
                    rtl_name = rtl_name[0:start]

            for nodeitem in ioitem.find('second').find('item').find('second').findall('item'):
                nodeid = int(nodeitem.text)
                if nodeid in cdfg_fsm_node_dict:
                    node = cdfg_fsm_node_dict[nodeid]
                    if rtl_name in rop_dict:
                        rop_dict[rtl_name].add_cop(int(nodeid))
                        rop_dict[rtl_name].add_line_num(int(node.line_num))
                        rop_dict[rtl_name].add_latency(int(node.latency))
                        if not node.opcode in rop_dict[rtl_name].opcode:
                            rop_dict[rtl_name].opcode = rop_dict[rtl_name].opcode + '_' + node.opcode
                        if not rop_dict[rtl_name].optype == optype:
                            print('CHECK: one rtl_name ({}) has multiple optypes: {} and {}.'.format(rtl_name, rop_dict[rtl_name].optype, optype))
                        if rop_dict[rtl_name].delay != node.m_delay:
                            print('CHECK: rtl_name ({}) has multiple delay values: {} and {}.'.format(node.rtl_name, rop_dict[node.rtl_name].delay, node.m_delay))

                        cop_dict[nodeid] = c_operator(node.nodeid, node.name, rop_dict[rtl_name].rtl_id, rtl_name, optype, node.opcode, node.line_num, node.c_step, \
                            node.latency, node.instruction, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, node.m_delay)
                    else:
                        rop_dict[rtl_name] = rtl_operator(rtl_name, rtl_id, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, \
                            optype, node.opcode, node.m_delay)
                        cop_dict[nodeid] = c_operator(node.nodeid, node.name, rtl_id, rtl_name, optype, node.opcode, node.line_num, node.c_step, \
                            node.latency, node.instruction, node.opnd_num, node.bw_out, node.bw_0, node.bw_1, node.bw_2, node.m_delay)
                        rtl_id = rtl_id + 1
                        rop_dict[rtl_name].add_cop(int(nodeid))
                        rop_dict[rtl_name].add_line_num(int(node.line_num))
                        rop_dict[rtl_name].add_latency(int(node.latency))
                else:
                    print('CHECK: get_io_op: nodeid ({}) not in cdfg_fsm_node_dict'.format(nodeid))
    return rtl_id


def get_rtlop_reg_name(IRinfo, cdfg_fsm_node_dict, rop_dict):
    for dp_regname_nodes in IRinfo.iter('dp_regname_nodes'):
        for regitem in dp_regname_nodes.findall('item'):
            real_name = regitem.find('first').text
            for regnode in regitem.find('second').findall('item'):
                nodeid = int(regnode.text)
                if nodeid in cdfg_fsm_node_dict:
                    node = cdfg_fsm_node_dict[nodeid]
                if (node.rtl_name in rop_dict) and (real_name not in rop_dict[node.rtl_name].reg_name):
                    rop_dict[node.rtl_name].add_reg_name(real_name)


def get_rtlop_comp_name(IRinfo, rop_dict):
    for dp_fu_nodes_expression in IRinfo.iter('dp_fu_nodes_expression'):
        for fu_node in dp_fu_nodes_expression.findall('item'):
            comp_name = fu_node.find('first').text
            comp_node_set = set()
            for node_item in fu_node.find('second').findall('item'):
                comp_node_set.add(int(node_item.text))
            for name in rop_dict:
                if len(rop_dict[name].cop_set - comp_node_set) == 0:
                    rop_dict[name].comp_name = comp_name

    for dp_fu_nodes_expression in IRinfo.iter('dp_fu_nodes_module'):
        for fu_node in dp_fu_nodes_expression.findall('item'):
            comp_name = fu_node.find('first').text
            comp_node_set = set()
            for node_item in fu_node.find('second').findall('item'):
                comp_node_set.add(int(node_item.text))
            for name in rop_dict:
                if len(rop_dict[name].cop_set - comp_node_set) == 0:
                    rop_dict[name].comp_name = comp_name


def c_rtl_op_explore(info_dir, IRinfo, cdfg_fsm_node_dict, rsc_dict):
    rtl_id = 0
    cop_dict = dict()
    rop_dict = dict()
    rtl_id = get_arith_logic_arbit_op(cdfg_fsm_node_dict, rsc_dict, cop_dict, rop_dict, rtl_id)
    rtl_id = get_mem_op(IRinfo, cdfg_fsm_node_dict, rsc_dict, cop_dict, rop_dict, rtl_id)
    rtl_id = get_io_op(IRinfo, cdfg_fsm_node_dict, rsc_dict, cop_dict, rop_dict, rtl_id)
    get_rtlop_reg_name(IRinfo, cdfg_fsm_node_dict, rop_dict)
    get_rtlop_comp_name(IRinfo, rop_dict)

    with open('{}/rop_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['rtl_name', 'comp_name', 'rtl_id', 'optype', 'opcode', 'reg_name', 'opnd_num', 'bw_out', 'bw_0', 'bw_1', 'bw_2', 'lut', 'dsp', 'bram', 'ff', \
            'mem_words',  'mem_bits', 'mem_banks', 'mem_wxbitsxbanks', 'cop_set', 'line_num_set', 'latency_set', 'latency', 'delay']
        writer.writerow(title)
        for name in rop_dict:
            rop = rop_dict[name]
            wr_line = [rop.rtl_name, rop.comp_name, rop.rtl_id, rop.optype, rop.opcode, '{}'.format(', '.join(map(str, rop.reg_name))), rop.opnd_num, rop.bw_out,
                rop.bw_0, rop.bw_1, rop.bw_2, rop.lut, rop.dsp, rop.bram, rop.ff, rop.mem_words, rop.mem_bits, rop.mem_banks, rop.mem_wxbitsxbanks, 
                str(sorted(rop.cop_set)).strip('[]'), str(sorted(rop.line_num_set)).strip('[]'), str(sorted(rop.latency_set)).strip('[]'),
                rop.latency, rop.delay]
            writer.writerow(wr_line)

    with open('{}/cop_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        title = ['nodeid', 'name', 'rtl_id', 'rtl_name', 'optype', 'opcode', 'line_num', 'c_step', 'latency', 'opnd_num', 'bw_out', 'bw_0', 'bw_1', 'bw_2',\
            'm_delay', 'instruction']
        writer.writerow(title)
        for nodeid in cop_dict:
            cop = cop_dict[nodeid]
            if cop.rtl_name == 'not_exist':
                print('CHECK: cop ({}) in cop_dict does not has rtl_name'.format(nodeid))
            wr_line = [cop.nodeid, cop.name, cop.rtl_id, cop.rtl_name, cop.optype, cop.opcode, cop.line_num, cop.c_step, cop.latency, cop.opnd_num, cop.bw_out, cop.bw_0, \
                cop.bw_1, cop.bw_2, cop.m_delay, cop.instruction]
            writer.writerow(wr_line)

    # with open('{}/cop_dict.txt'.format(info_dir), 'w+', newline = '') as wfile:
    #     for nodeid in cop_dict:
    #         cop = cop_dict[nodeid]
    #         wr_line = "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(cop.nodeid, cop.name, cop.rtl_id, cop.rtl_id, cop.rtl_name, cop.optype, 
    #             cop.opcode, cop.c_step, cop.latency, cop.opnd_num, cop.bw_out, cop.bw_0, cop.bw_1, cop.bw_2, cop.m_delay, cop.instruction)
    #         wfile.write(wr_line + "\n")

    return cop_dict, rop_dict, rtl_id


###################################################################
def get_global_info(RSCinfo):
    global_dict = dict()
    for section in RSCinfo.findall('section'):
        if section.get('name') == 'Performance Estimates':
            for item in section.findall('item'):
                if item.get('name') == 'Timing (ns)':
                    for column in item.iter('column'):
                        if column.get('name') == 'ap_clk':
                            timing_list = column.text.split(', ')
                            global_dict['clk_target'] = float(timing_list[0])
                            global_dict['clk_estimated'] = float(timing_list[1])
                            global_dict['clk_uncertainty'] = float(timing_list[2])

                elif item.get('name') == 'Latency (clock cycles)':
                    for column in item.iter('column'):
                        if column.get('name') == '':
                            latency_list = column.text.split(', ')
                            global_dict['latency_min'] = int(latency_list[0])
                            global_dict['latency_max'] = int(latency_list[1])
                            global_dict['interval_min'] = int(latency_list[2])
                            global_dict['interval_max'] = int(latency_list[3])

        elif section.get('name') == 'Utilization Estimates':
            global_dict['bram'] = 0
            global_dict['dsp'] = 0
            global_dict['ff'] = 0
            global_dict['lut'] = 0
            global_dict['uram'] = 0
            for item in section.findall('item'):
                if item.get('name') == 'Summary':
                    for column in item.iter('column'):
                        res_list = column.text.split(', ')
                        if not res_list[0] == '-':
                            global_dict['bram'] += int(res_list[0])
                        if not res_list[1] == '-':
                            global_dict['dsp'] += int(res_list[1])
                        if not res_list[2] == '-':
                            global_dict['ff'] += int(res_list[2])
                        if not res_list[3] == '-':
                            global_dict['lut'] += int(res_list[3])
                        if not res_list[4] == '-':
                            global_dict['uram'] += int(res_list[4])
                    break
    return global_dict


def global_explore(info_dir, RSCinfo):
    global_dict = get_global_info(RSCinfo)
    with open('{}/global_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        for name in global_dict:
            val = global_dict[name]
            wr_line = [name, val]
            writer.writerow(wr_line)
    return global_dict


###################################################################
def set_graph_node_info(DG, nodeid, **kwargs): 
    graph_node = DG.nodes[nodeid]
    for attr_name in kwargs:
        graph_node[attr_name] = kwargs[attr_name]


###########################
# compute the rop fanin/fanout before bypassing op and adding mux, not the very precise one in the RTL, but can use as a reference
def fan_in_out_compute(DG):
    for nodeid in DG.nodes:
        DG.nodes[nodeid]['fan_in'] = len(list(DG.predecessors(nodeid)))
        DG.nodes[nodeid]['fan_out'] = len(list(DG.successors(nodeid)))


###########################
# construct the graph: connects nodes and edges of fsm_node_dict; the edges use reg_flow = (from_reg, to_reg) to represent the dataflow with registers
def df_node_edge_construct(DG, fsm_node_dict, cdfg_fsm_node_dict, df_reg_dict, cop_dict):
    for nodeid in fsm_node_dict:
        fsm_node = fsm_node_dict[nodeid]
        DG.add_node(nodeid)
        if nodeid in cdfg_fsm_node_dict:
            opcode = cdfg_fsm_node_dict[nodeid].opcode
        else:
            opcode = 'not_exist'

        if nodeid in cop_dict:
            optype = cop_dict[nodeid].optype
        else:
            optype = 'not_exist'
        set_graph_node_info(DG, nodeid, node_id = nodeid, opid = fsm_node.opid, c_step = fsm_node.c_step, stage = fsm_node.stage, opcode = opcode, optype = optype,
            latency = fsm_node.latency, opnd_num = fsm_node.opnd_num, bw_out = fsm_node.bw_out, bw_0 = fsm_node.bw_0, bw_1 = fsm_node.bw_1,
            bw_2 = fsm_node.bw_2, from_node_set = fsm_node.from_node_set, to_node_set = fsm_node.to_node_set, instruction = fsm_node.instruction)

    for nodeid in fsm_node_dict:
        fsm_node = fsm_node_dict[nodeid]
        for from_nodeid in fsm_node.from_node_set:
            if from_nodeid in DG.nodes:
                DG.add_edge(from_nodeid, nodeid, reg_flow = [df_reg_dict[from_nodeid][nodeid]])
            else:
                print('CHECK: df_node_edge_construct: from_nodeid = {} not in DG'.format(from_nodeid))
                assert(0)
        for to_nodeid in fsm_node.to_node_set:
            if to_nodeid in DG.nodes:
                DG.add_edge(nodeid, to_nodeid, reg_flow = [df_reg_dict[nodeid][to_nodeid]])
            else:
                print('CHECK: df_node_edge_construct: to_nodeid = {} not in DG'.format(to_nodeid))
                assert(0)
    
    DG.remove_nodes_from(list(nx.isolates(DG)))


###########################
# add bram nodes: for 'alloca'->'getelementptr', insert new internam_mem node parallel to alloca, and then delete alloca
#                 for 'getelementptr' without 'alloca', this is io_mem, confirm with rop_dcit and connect it to 'getelementptr'
def df_mem_insert(DG, rop_dict, cdfg_fsm_node_dict):
    internal_node_dict = dict()
    new_edge_list = list()
    internal_index = 0
    io_index = 0
    io_mem_name_dict = dict()
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] == 'getelementptr': # found 'alloca' -> internal_mem
            found_alloca = False
            for pre_id in list(DG.predecessors(nodeid)):
                if DG.nodes[pre_id]['opcode'] == 'alloca':
                    found_alloca = True
                    if pre_id not in internal_node_dict:
                        internal_id = 'internal_mem_' + str(internal_index)
                        internal_node_dict[pre_id] = internal_id
                        internal_index += 1

                    reg_flow = [(DG[pre_id][nodeid]['reg_flow'][0][0], DG[pre_id][nodeid]['reg_flow'][0][1], internal_node_dict[pre_id], nodeid)]
                    new_edge_list.append((internal_node_dict[pre_id], nodeid, reg_flow))

            if found_alloca == False: # cannot find 'alloca' -> io_mem
                instruction = DG.nodes[nodeid]['instruction']
                src_name = re.split(' |,', instruction)[6].strip('%')
                dst_name = re.split(' |,', instruction)[0].strip('%')
                if src_name in rop_dict:
                    if src_name not in io_mem_name_dict:
                        io_id = 'io_mem_' + str(io_index)
                        io_mem_name_dict[src_name] = io_id
                        io_index += 1
                    else:
                        io_id = io_mem_name_dict[src_name]

                    reg_flow = [(src_name, dst_name, io_id, nodeid)]
                    new_edge_list.append((io_id, nodeid, reg_flow))
                else:
                    print('CHECK: df_mem_insert: do not find io_mem = {} in rop_dict'.format(src_name))

    for nodeid in internal_node_dict:
        internal_id = internal_node_dict[nodeid]
        DG.add_node(internal_id)
        set_graph_node_info(DG, internal_id, node_id = internal_id, opid = internal_id, opcode = 'internal_mem')
        cdfg_fsm_node = cdfg_fsm_node_dict[nodeid]
        if cdfg_fsm_node.rtl_name in rop_dict:
            rop_node = rop_dict[cdfg_fsm_node.rtl_name]
            set_graph_node_info(DG, internal_id, node_id = internal_id, rtl_name = rop_node.rtl_name, rtl_id = rop_node.rtl_id, 
                reg_name = rop_node.reg_name, opnd_num = rop_node.opnd_num, bw_out = rop_node.bw_out, bw_0 = rop_node.bw_0, 
                bw_1 = rop_node.bw_1, bw_2 = rop_node.bw_2, lut = rop_node.lut, dsp = rop_node.dsp, bram = rop_node.bram, ff = rop_node.ff, 
                mem_words = rop_node.mem_words,mem_bits = rop_node.mem_bits, mem_banks = rop_node.mem_banks, mem_wxbitsxbanks = rop_node.mem_wxbitsxbanks, 
                optype = rop_node.optype, cop_set = rop_node.cop_set, line_num_set = rop_node.line_num_set, 
                latency_set = rop_node.latency_set, latency = rop_node.latency, delay = rop_node.delay)
        else:
            print('CHECK: df_mem_insert: do not find internel_mem = {} in rop_dict'.format(cdfg_fsm_node.rtl_name))

    for io_mem_name in io_mem_name_dict:
        io_id = io_mem_name_dict[io_mem_name]
        DG.add_node(io_id)
        rop_node = rop_dict[io_mem_name]
        set_graph_node_info(DG, io_id, node_id = io_id, opid = io_id, opcode = 'io_mem', rtl_name = rop_node.rtl_name, rtl_id = rop_node.rtl_id, 
            reg_name = rop_node.reg_name, opnd_num = rop_node.opnd_num, bw_out = rop_node.bw_out, bw_0 = rop_node.bw_0, bw_1 = rop_node.bw_1, 
            bw_2 = rop_node.bw_2, lut = rop_node.lut, dsp = rop_node.dsp, bram = rop_node.bram, ff = rop_node.ff, mem_words = rop_node.mem_words, 
            mem_bits = rop_node.mem_bits, mem_banks = rop_node.mem_banks, mem_wxbitsxbanks = rop_node.mem_wxbitsxbanks, 
            optype = rop_node.optype, cop_set = rop_node.cop_set, line_num_set = rop_node.line_num_set, 
            latency_set = rop_node.latency_set, latency = rop_node.latency, delay = rop_node.delay)

    for srcid, dstid, reg_flow in new_edge_list:
        DG.add_edge(srcid, dstid, reg_flow = reg_flow)

    DG.remove_nodes_from(internal_node_dict.keys()) # delete 'alloca' nodes, because they are meaningless in the graphs


###########################
# the prior function deals with 'alloca'->'getelementptr', but if 'alloca' allocates registers, there are no 'getelementptr'
# to deal with this case: insert 'reg' type nodes parallel to 'alloca'; the 'store' edge should be revise; delete 'alloca' at last
def df_reg_insert(DG):
    reg_node_list = list()
    new_edge_list = list()
    rm_node_list = list()
    reg_index = 0
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] == 'alloca': # after the processing of prior function, 'alloca'->'getelementptr' does not exist
            reg_id = 'reg_' + str(reg_index)
            reg_node_list.append(reg_id)
            reg_index += 1

            for pre_id in list(DG.predecessors(nodeid)):
                reg_flow = [(DG[pre_id][nodeid]['reg_flow'][0][0], DG[pre_id][nodeid]['reg_flow'][0][1], pre_id, nodeid)]
                new_edge_list.append((pre_id, reg_id, reg_flow))

            for suc_id in list(DG.successors(nodeid)):
                if DG.nodes[suc_id]['opcode'] == 'store': # reverse the connection for 'store'
                    reg_flow = [(DG[nodeid][suc_id]['reg_flow'][0][1], DG[nodeid][suc_id]['reg_flow'][0][0], suc_id, nodeid)]
                    new_edge_list.append((suc_id, reg_id, reg_flow))
                else:
                    reg_flow = [(DG[nodeid][suc_id]['reg_flow'][0][0], DG[nodeid][suc_id]['reg_flow'][0][1], nodeid, suc_id)]
                    new_edge_list.append((reg_id, suc_id, reg_flow))

            rm_node_list.append(nodeid)

    for reg_id in reg_node_list:
        DG.add_node(reg_id)
        set_graph_node_info(DG, reg_id, node_id = reg_id, opid = reg_id, opcode = 'reg', optype = 'reg')
    
    for srcid, dstid, reg_flow in new_edge_list:
        DG.add_edge(srcid, dstid, reg_flow = reg_flow)

    DG.remove_nodes_from(rm_node_list)


###########################
# op = 'read' indicates input variable; op = 'write' indicates output FIFO
# add io entities according to 'read' and 'write'
def df_io_port_insert(DG, rop_dict):
    new_edge_list = list()

    # deal with op = 'read'
    rd_node_dict = dict()
    rd_index = 0
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] == 'read':
            instruction = DG.nodes[nodeid]['instruction']
            src_name = re.split(' |\)', instruction.strip(''))[5].strip('%')
            dst_name = re.split(' |\)', instruction.strip(''))[0].strip('%')
            if src_name in rop_dict:
                if src_name not in rd_node_dict:
                    io_val_id = 'io_val_' + str(rd_index)
                    rd_node_dict[src_name] = io_val_id
                    rd_index += 1
                else:
                    io_val_id = rd_node_dict[src_name]

                reg_flow = [(src_name, dst_name, io_val_id, nodeid)]
                new_edge_list.append((io_val_id, nodeid, reg_flow))
            else:
                print('CHECK: df_io_port_insert: not find src_name = {} in rop_dict'.format(src_name))

    for io_val_name in rd_node_dict:
        io_val_id = rd_node_dict[io_val_name]
        DG.add_node(io_val_id)
        rop_node = rop_dict[io_val_name]
        set_graph_node_info(DG, io_val_id, node_id = io_val_id, opid = io_val_id, opcode = 'io_port_in', rtl_name = rop_node.rtl_name, rtl_id = rop_node.rtl_id, 
            reg_name = rop_node.reg_name, opnd_num = rop_node.opnd_num, bw_out = rop_node.bw_out, bw_0 = rop_node.bw_0, bw_1 = rop_node.bw_1, 
            bw_2 = rop_node.bw_2, lut = rop_node.lut, dsp = rop_node.dsp, bram = rop_node.bram, ff = rop_node.ff, mem_words = rop_node.mem_words,
            mem_bits = rop_node.mem_bits, mem_banks = rop_node.mem_banks, mem_wxbitsxbanks = rop_node.mem_wxbitsxbanks, 
            optype = rop_node.optype, cop_set = rop_node.cop_set, line_num_set = rop_node.line_num_set, 
            latency_set = rop_node.latency_set, latency = rop_node.latency, delay = rop_node.delay)

    # deal with op = 'write'
    wr_node_dict = dict()
    wr_index = 0
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] == 'write': 
            instruction = DG.nodes[nodeid]['instruction']
            src_name = re.split(' |,|\)', instruction.strip(''))[6].strip('%')
            dst_name = re.split(' |,', instruction.strip(''))[3].strip('%') # for 'write', the result_op is the first input op of the instruction
            if dst_name in rop_dict:
                if dst_name not in wr_node_dict:
                    io_fifo_id = 'io_fifo_' + str(wr_index)
                    wr_node_dict[dst_name] = io_fifo_id
                    wr_index += 1
                else:
                    io_fifo_id = wr_node_dict[dst_name]

                reg_flow = [(src_name, dst_name, nodeid, io_fifo_id)]
                new_edge_list.append((nodeid, io_fifo_id, reg_flow))
            else:
                print('CHECK: df_io_port_insert: not find dst_name = {} in rop_dict'.format(dst_name))

    for io_fifo_name in wr_node_dict:
        io_fifo_id = wr_node_dict[io_fifo_name]
        DG.add_node(io_fifo_id)
        rop_node = rop_dict[io_fifo_name]
        set_graph_node_info(DG, io_fifo_id, node_id = io_fifo_id, opid = io_fifo_id, opcode = 'io_port_out', rtl_name = rop_node.rtl_name, rtl_id = rop_node.rtl_id, 
            reg_name = rop_node.reg_name, opnd_num = rop_node.opnd_num, bw_out = rop_node.bw_out, bw_0 = rop_node.bw_0, bw_1 = rop_node.bw_1, 
            bw_2 = rop_node.bw_2, lut = rop_node.lut, dsp = rop_node.dsp, bram = rop_node.bram, ff = rop_node.ff, mem_words = rop_node.mem_words,
            mem_bits = rop_node.mem_bits, mem_banks = rop_node.mem_banks, mem_wxbitsxbanks = rop_node.mem_wxbitsxbanks, 
            optype = rop_node.optype, cop_set = rop_node.cop_set, line_num_set = rop_node.line_num_set, 
            latency_set = rop_node.latency_set, latency = rop_node.latency, delay = rop_node.delay)

    for srcid, dstid, reg_flow in new_edge_list:
        DG.add_edge(srcid, dstid, reg_flow = reg_flow)


###########################
# for 'load' nodes, connect to upper internal_mem/io_mem nodes and disconnect with 'getelementptr'
# for 'store' nodes, trace back to 'getelementptr' and the upper internal_mem/io_mem, reversely connect to mem nodes; disconnect with 'getelementptr'
def df_mem_connect(DG):
    new_edge_list = list()
    rm_edge_list = list()
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] == 'store':
            for pre_id in list(DG.predecessors(nodeid)):
                if DG.nodes[pre_id]['opcode'] == 'getelementptr':
                    for pre_pre_id in list(DG.predecessors(pre_id)):
                        if DG.nodes[pre_pre_id]['opcode'] in ['internal_mem', 'io_mem']:
                            # check the correctness of reg_flow connection: from_reg1->to_reg1/from_reg2->to_reg2, to_reg1 = from_reg2
                            if DG[pre_pre_id][pre_id]['reg_flow'][0][1] != DG[pre_id][nodeid]['reg_flow'][0][0]:
                                print('CHECK: df_mem_connect: reg_flow correspondence not match: {} != {} for connecting mem id = {} to store id = {}'.format(
                                    DG[pre_pre_id][pre_id]['reg_flow'][0][1], DG[pre_id][nodeid]['reg_flow'][0][0], pre_pre_id, nodeid))
                                assert(0)
                            reg_flow = [(DG[pre_id][nodeid]['reg_flow'][0][1], DG[pre_pre_id][pre_id]['reg_flow'][0][0],
                                DG[pre_id][nodeid]['reg_flow'][0][3], DG[pre_pre_id][pre_id]['reg_flow'][0][2])] # reverse connection: from store to mem
                            new_edge_list.append((nodeid, pre_pre_id, reg_flow))
                            rm_edge_list.append((pre_id, nodeid))

        elif DG.nodes[nodeid]['opcode'] == 'load':
            for pre_id in list(DG.predecessors(nodeid)): 
                if DG.nodes[pre_id]['opcode'] == 'getelementptr':
                    for pre_pre_id in list(DG.predecessors(pre_id)): 
                        if DG.nodes[pre_pre_id]['opcode'] in ['internal_mem', 'io_mem']:
                            # check the correctness of reg_flow connection: from_reg1->to_reg1/from_reg2->to_reg2, to_reg1 = from_reg2
                            if DG[pre_pre_id][pre_id]['reg_flow'][0][1] != DG[pre_id][nodeid]['reg_flow'][0][0]:
                                print('CHECK: df_mem_connect: reg_flow correspondence not match: {} != {} for connecting mem id = {} to load id = {}'.format(
                                    DG[pre_pre_id][pre_id]['reg_flow'][0][1], DG[pre_id][nodeid]['reg_flow'][0][0], pre_pre_id, nodeid))
                                assert(0)
                            reg_flow = [(DG[pre_pre_id][pre_id]['reg_flow'][0][0], DG[pre_id][nodeid]['reg_flow'][0][1],
                                DG[pre_pre_id][pre_id]['reg_flow'][0][2], DG[pre_id][nodeid]['reg_flow'][0][3])] # forward connection: from mem to load
                            new_edge_list.append((pre_pre_id, nodeid, reg_flow))
                            rm_edge_list.append((pre_id, nodeid))

    DG.remove_edges_from(rm_edge_list)
    for srcid, dstid, reg_flow in new_edge_list:
        DG.add_edge(srcid, dstid, reg_flow = reg_flow)


###########################
# graph trimming: remove trivial nodes in order to highlight arithmetic-intensive nodes and paths
df_bypass_op = ['const', 'br', 'icmp', 'bitconcatenate', 'zext', 'not_exist', 'and', 'or', 'xor', 'getelementptr', 'partselect', 'reg',
    'bitcast', 'bitselect', 'call', 'trunc', 'switch', 'sext', 'urem']

def df_op_bypass(DG, df_bypass_op):
    for bypass_opcode in df_bypass_op:
        rm_node_list = list()
        for nodeid in DG.nodes:
            if DG.nodes[nodeid]['opcode'] == bypass_opcode:
                for pre_id in list(DG.predecessors(nodeid)):
                    for suc_id in list(DG.successors(nodeid)):
                        # check the correctness of reg_flow connection: from_reg1->to_reg1/from_reg2->to_reg2, to_reg1 = from_reg2
                        if DG[pre_id][nodeid]['reg_flow'][0][1] != DG[nodeid][suc_id]['reg_flow'][0][0]:
                            print('CHECK: df_op_bypass: reg_flow correspondence not match: {} != {} for connecting id: {} -> {} -> {}'.format(
                                DG[pre_id][nodeid]['reg_flow'][0][1], DG[nodeid][suc_id]['reg_flow'][0][0], pre_id, nodeid, suc_id))
                            assert(0)

                        reg_flow = [(DG[pre_id][nodeid]['reg_flow'][0][0], DG[nodeid][suc_id]['reg_flow'][0][1],
                            DG[pre_id][nodeid]['reg_flow'][0][2], DG[nodeid][suc_id]['reg_flow'][0][3])]
                        if DG.has_edge(pre_id, suc_id):
                            DG[pre_id][suc_id]['reg_flow'] = list(set(DG[pre_id][suc_id]['reg_flow'] + reg_flow))
                        else:
                            DG.add_edge(pre_id, suc_id, reg_flow = reg_flow)
                rm_node_list.append(nodeid)
        DG.remove_nodes_from(rm_node_list)


###########################
# node merging: connect all edges of nodeid2 to nodeid1, then delete nodeid2
# if there exists the same edges (reg_flow the same) before merging, merge reg_flow; otherwise copy reg_flow
def contracted_nodes(DG, nodeid1, nodeid2):
    if nodeid1 in DG.nodes and nodeid2 in DG.nodes:
        for pre_id in list(DG.predecessors(nodeid2)):
            if DG.has_edge(pre_id, nodeid1):
                DG[pre_id][nodeid1]['reg_flow'] = list(set(DG[pre_id][nodeid1]['reg_flow'] + DG[pre_id][nodeid2]['reg_flow']))
            else:
                DG.add_edge(pre_id, nodeid1, reg_flow = list(set(DG[pre_id][nodeid2]['reg_flow'])))

        for suc_id in list(DG.successors(nodeid2)):
            if DG.has_edge(nodeid1, suc_id):
                DG[nodeid1][suc_id]['reg_flow'] = list(set(DG[nodeid1][suc_id]['reg_flow'] + DG[nodeid2][suc_id]['reg_flow']))
            else:
                DG.add_edge(nodeid1, suc_id, reg_flow = list(set(DG[nodeid2][suc_id]['reg_flow'])))
        DG.remove_node(nodeid2)
    else:
        print('CHECK: contracted_nodes: nodeid1 = {} or nodeid2 = {} not exist in DG.nodes'.format(nodeid1, nodeid2))
        assert(0)


# compute the locations of item in seq, return a list of locations (allow returning multiple locations)
def duplicate_extract(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


# merge all cop of the same rtlop; pay attention: load/store are not completely merged so that the reg_flow can be clearly represented
# separately deal with load/store: check whether the connection are identical: pre_op -> load/store -> suc_op
# merge if connected to the same pre_op and suc_op, and then denote this relationship in the edge: multiple reg_flow in the same edge
def df_node_merge(DG, rop_dict):
    for rtl_name in rop_dict:
        rop = rop_dict[rtl_name]
        cop_list = sorted(list(rop.cop_set))
        
        if rop.optype in ['memory', 'io']:
            if rop.opcode == 'write':
                continue
            load_list = [cop for cop in cop_list if DG.nodes[cop]['opcode'] == 'load']
            store_list = [cop for cop in cop_list if DG.nodes[cop]['opcode'] == 'store']
            for i, target_list in enumerate([load_list, store_list]):
                if len(target_list) == 0:
                    continue
                neighbor_list = list()
                duplicate_list = list()
                for nodeid in target_list:
                    pre_list = list(DG.predecessors(nodeid))
                    suc_list = list(DG.successors(nodeid))
                    neighbor_list.append((pre_list, suc_list))
                for neighbor in neighbor_list:
                    duplicates = duplicate_extract(neighbor_list, neighbor)
                    if duplicates not in duplicate_list and len(duplicates) > 1:
                        duplicate_list.append(duplicates)
                for duplicates in duplicate_list:
                    node_list = list(itemgetter(*duplicates)(target_list))
                    for nodeid in node_list[1:]:
                        contracted_nodes(DG, node_list[0], nodeid)
                    set_graph_node_info(DG, node_list[0], node_id = node_list[0], merge_op_set = node_list)

        elif len(cop_list) > 1: # rop.optype not in ['memory', 'io']:
            if rop.opcode in df_bypass_op:
                continue
            if cop_list[0] not in DG.nodes:
                print('CHECK: df_node_merge: cop_list[0] = {} not in DG.nodes'.format(cop_list[0]))
                assert(0)
            for nodeid in cop_list[1:]:
                if nodeid in DG.nodes:
                    contracted_nodes(DG, cop_list[0], nodeid)
                else:
                    print('CHECK: df_node_merge: nodeid = {} not in DG.nodes'.format(nodeid))
                    assert(0)
            set_graph_node_info(DG, cop_list[0], node_id = cop_list[0], merge_op_set = cop_list)


###########################
def df_mem_path_find(DG):
    mem_path_dict = defaultdict(lambda: defaultdict(list))
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] in ['internal_mem', 'io_mem', 'io_port_in', 'io_port_out']:
            for suc_id_1 in list(DG.successors(nodeid)):
                if DG.nodes[suc_id_1]['opcode'] == 'load':
                    for suc_id_2 in list(DG.successors(suc_id_1)):
                        if DG.nodes[suc_id_2]['opcode'] in ['store', 'write']:
                            for suc_id_3 in list(DG.successors(suc_id_2)):
                                if DG.nodes[suc_id_3]['opcode'] in ['internal_mem', 'io_mem', 'io_port_in', 'io_port_out']:
                                    if (suc_id_1, suc_id_2) not in mem_path_dict[nodeid][suc_id_3]:
                                        mem_path_dict[nodeid][suc_id_3].append((suc_id_1, suc_id_2))
    return mem_path_dict


# check and merge: mem -> load -> store -> mem
#                  mem -> load -> write -> mem
def df_mem_path_merge(DG):
    mem_path_dict = df_mem_path_find(DG)
    for src_id in mem_path_dict:
        for dst_id in mem_path_dict[src_id]:
            if len(mem_path_dict[src_id][dst_id]) > 1:
                load_op_list = [mem_path_dict[src_id][dst_id][0][0]]
                store_op_list = [mem_path_dict[src_id][dst_id][0][1]]
                for id_1, id_2 in mem_path_dict[src_id][dst_id][1:]:
                    contracted_nodes(DG, mem_path_dict[src_id][dst_id][0][0], id_1)
                    load_op_list.append(id_1)
                    contracted_nodes(DG, mem_path_dict[src_id][dst_id][0][1], id_2)
                    store_op_list.append(id_2)

                set_graph_node_info(DG, mem_path_dict[src_id][dst_id][0][0], node_id = mem_path_dict[src_id][dst_id][0][0], merge_op_set = load_op_list)
                set_graph_node_info(DG, mem_path_dict[src_id][dst_id][0][1], node_id = mem_path_dict[src_id][dst_id][0][1], merge_op_set = store_op_list)


###########################
# remove unimportant connected_components (cc), which usually act as ctrl flows
def df_cc_trim(DG):
    rm_node_list = list()
    for subG in nx.weakly_connected_components(DG):
        find_critical_component = False
        for nodeid in subG:
            if DG.nodes[nodeid]['opcode'] in ['internal_mem', 'io_mem', 'io_port_in', 'io_port_out', 'fadd', 'fsub', 'fdiv', 'read', 'write']:
                find_critical_component = True
                break
        if find_critical_component == False:
            rm_node_list += list(subG)
    DG.remove_nodes_from(rm_node_list)


###########################
# complement rtl info of nodes 
def df_rtl_info_add(DG, cdfg_fsm_node_dict, rop_dict):
    for nodeid in DG.nodes:
        if nodeid in cdfg_fsm_node_dict:
            cdfg_fsm_node = cdfg_fsm_node_dict[nodeid]
            if cdfg_fsm_node.rtl_name in rop_dict:
                rop_node = rop_dict[cdfg_fsm_node.rtl_name]
                set_graph_node_info(DG, nodeid, node_id = nodeid, rtl_name = rop_node.rtl_name, rtl_id = rop_node.rtl_id, reg_name = rop_node.reg_name, 
                    opnd_num = rop_node.opnd_num, bw_out = rop_node.bw_out, bw_0 = rop_node.bw_0, bw_1 = rop_node.bw_1, bw_2 = rop_node.bw_2, 
                    lut = rop_node.lut, dsp = rop_node.dsp, bram = rop_node.bram, ff = rop_node.ff, mem_words = rop_node.mem_words,
                    mem_bits = rop_node.mem_bits, mem_banks = rop_node.mem_banks, mem_wxbitsxbanks = rop_node.mem_wxbitsxbanks, 
                    optype = rop_node.optype, cop_set = rop_node.cop_set, line_num_set = rop_node.line_num_set, 
                    latency_set = rop_node.latency_set, latency = rop_node.latency, delay = rop_node.delay)


###########################
def is_str_int(str_in):
    try:
        int(str_in)
        return 1
    except ValueError:
        return 0


# extract node activities
def df_node_tracer_gen(DG, info_dir, cdfg_fsm_node_dict, cop_dict, rop_dict):
    cop_node_dict = defaultdict(set)
    for nodeid in DG.nodes:
        node_id = DG.nodes[nodeid]['node_id']
        if 'rtl_name' in DG.nodes[nodeid]:
            rtl_name = DG.nodes[nodeid]['rtl_name']
            rtl_op_set = rop_dict[rtl_name].cop_set
            for op_id in rtl_op_set:
                cop_node_dict[op_id].add(nodeid)

        if 'merge_op_set' in DG.nodes[nodeid]:
            merge_op_set = DG.nodes[nodeid]['merge_op_set']
            for op_id in merge_op_set:
                cop_node_dict[op_id].add(nodeid)

        if is_str_int(node_id):
            cop_node_dict[node_id].add(nodeid)

    with open('{}/cop_node_dict.txt'.format(info_dir), 'w+', newline = '') as wfile:
        for cop_id in cop_node_dict:
            if cop_id not in cop_dict:
                if cop_id in cdfg_fsm_node_dict:
                    if cdfg_fsm_node_dict[cop_id].opcode == 'phi':
                        continue
                    cop = cdfg_fsm_node_dict[cop_id]
                    rtl_id = -1
                    rtl_name = 'not_exist'
                    optype = cop.opcode
                else:
                    continue
            else:
                cop = cop_dict[cop_id]
                rtl_name = cop.rtl_name
                rtl_id = cop.rtl_id
                optype = cop.optype

            node_id_list = str(list(cop_node_dict[cop_id])).strip('[').strip(']')
            wr_line = "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(cop.nodeid, cop.name, node_id_list, rtl_id, rtl_name, optype, 
                cop.opcode, cop.c_step, cop.latency, cop.opnd_num, cop.bw_out, cop.bw_0, cop.bw_1, cop.bw_2, cop.m_delay, cop.instruction)
            wfile.write(wr_line + '\n')


###########################
class edge_obj:
    def __init__(self, edge_id, edge_src_id, edge_dst_id, src_id, dst_id, latency, cop_set = None):
        self.edge_id = edge_id
        self.edge_src_id = edge_src_id
        self.edge_dst_id = edge_dst_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.latency = latency
        self.cop_set = set()
        if cop_set != None:
            self.add_cop(cop_set)

    def add_cop(self, cop_id):
        cop_id_list = [cop_id] if type(cop_id) == int else cop_id
        self.cop_set = set(list(self.cop_set) + cop_id_list)


# generate the file required by the IR tracer
# to find the edge acitivity in graph act annotate: edge_id | src_node_id | dst_node_id | cop_set (for cop)
# to add tracer for cop in IR: cop_id | edge_id_list | opcode | opnd_num | bw_out | bw_0 | bw_1 | bw_2 | instruction
# the same cop can appear in different edge_id because of the merging process above
def df_edge_tracer_gen(DG, info_dir, cdfg_fsm_node_dict, cop_dict, rop_dict):
    cop_edge_dict = defaultdict(list)
    edge_obj_dict = dict()
    for edge_id, (src_id, dst_id) in enumerate(DG.edges):
        edge_src_id = 2 * edge_id
        edge_dst_id = 2 * edge_id + 1
        DG[src_id][dst_id]['edge_id'] = edge_id
        DG[src_id][dst_id]['edge_src_id'] = edge_src_id
        DG[src_id][dst_id]['edge_dst_id'] = edge_dst_id

        # if id not found, it is phi/io_mem/internal_mem, should be bypassed later
        src_cop_list = [reg_flow[2] for reg_flow in DG[src_id][dst_id]['reg_flow']]
        for nodeid in src_cop_list:
            cop_edge_dict[nodeid].append(edge_src_id)
            if nodeid in cdfg_fsm_node_dict:
                latency = cdfg_fsm_node_dict[nodeid].latency
            else:
                latency = 0

        edge_obj_dict[edge_src_id] = edge_obj(edge_id = edge_src_id, edge_src_id = edge_src_id, edge_dst_id =  edge_dst_id,
            src_id = src_id, dst_id = dst_id, latency = latency, cop_set = src_cop_list)

        dst_cop_list = [reg_flow[3] for reg_flow in DG[src_id][dst_id]['reg_flow']]
        for nodeid in dst_cop_list:
            cop_edge_dict[nodeid].append(edge_dst_id)
            if nodeid in cdfg_fsm_node_dict:
                latency = cdfg_fsm_node_dict[nodeid].latency
            else:
                latency = 0
        
        edge_obj_dict[edge_dst_id] = edge_obj(edge_id = edge_dst_id, edge_src_id = edge_src_id, edge_dst_id =  edge_dst_id,
            src_id = src_id, dst_id = dst_id, latency = latency, cop_set = dst_cop_list)

    with open('{}/cop_edge_dict.txt'.format(info_dir), 'w+', newline = '') as wfile:
        for cop_id in cop_edge_dict:
            if cop_id not in cop_dict:
                if cop_id in cdfg_fsm_node_dict:
                    if cdfg_fsm_node_dict[cop_id].opcode == 'phi':
                        continue
                    cop = cdfg_fsm_node_dict[cop_id]
                    rtl_id = -1
                    rtl_name = 'not_exist'
                    optype = cop.opcode
                else:
                    continue
            else:
                cop = cop_dict[cop_id]
                rtl_name = cop.rtl_name
                rtl_id = cop.rtl_id
                optype = cop.optype

            edge_id_list = str(list(set(cop_edge_dict[cop_id]))).strip('[').strip(']')
            wr_line = "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(cop.nodeid, cop.name, edge_id_list, rtl_id, rtl_name, optype, 
                cop.opcode, cop.c_step, cop.latency, cop.opnd_num, cop.bw_out, cop.bw_0, cop.bw_1, cop.bw_2, cop.m_delay, cop.instruction)
            wfile.write(wr_line + '\n')

    with open('{}/edge_obj_dict.csv'.format(info_dir), 'w+', newline = '') as wfile:
        writer = csv.writer(wfile)
        writer.writerow(['edge_id', 'edge_src_id', 'edge_dst_id', 'src_id', 'dst_id', 'latency', 'cop_set'])
        for edge_id in edge_obj_dict:
            edge_src_id = edge_obj_dict[edge_id].edge_src_id
            edge_dst_id = edge_obj_dict[edge_id].edge_dst_id
            src_id = edge_obj_dict[edge_id].src_id
            dst_id = edge_obj_dict[edge_id].dst_id
            latency = edge_obj_dict[edge_id].latency
            cop_set = str(edge_obj_dict[edge_id].cop_set).strip('{').strip('}')
            writer.writerow([edge_id, edge_src_id, edge_dst_id, src_id, dst_id, latency, cop_set])


###########################
def df_graph_visualize(out_dir, save_name, act_plot = False):
    DG = nx.DiGraph(nx.drawing.nx_pydot.read_dot('{}/{}.dot'.format(out_dir, save_name)))
    gvz_graph = gvz.Digraph(format = 'png', filename = '{}/{}'.format(out_dir, save_name))
    gvz_graph.attr('node', fontsize = '50')
    gvz_graph.attr('edge', arrowsize = '2.4', fontsize = '30')
    for nodeid in DG.nodes:
        if DG.nodes[nodeid]['opcode'] in ['internal_mem', 'io_mem', 'io_port_in', 'io_port_out']:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '1')
        elif DG.nodes[nodeid]['opcode'] in ['fadd', 'fsub', 'fmul', 'fdiv']:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '2')
        elif DG.nodes[nodeid]['opcode'] in ['store']:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '4')
        elif DG.nodes[nodeid]['opcode'] in ['phi']:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '5')
        elif DG.nodes[nodeid]['opcode'] in ['mux', 'select']:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '8')
        elif DG.nodes[nodeid]['opcode'] in ['add']:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '6')
        else:
            gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '3')

        if DG.nodes[nodeid]['opcode'] in ['internal_mem', 'io_mem', 'io_port_in', 'io_port_out']:
            gvz_graph.node(str(nodeid), label = '{} - {}\n{}'.format(DG.nodes[nodeid]['node_id'], nodeid, DG.nodes[nodeid]['rtl_name']))
        else:
            gvz_graph.node(str(nodeid), label = '{} - {} \n{}'.format(DG.nodes[nodeid]['node_id'], nodeid, DG.nodes[nodeid]['opcode']))

    for edge in DG.edges:
        if act_plot == True:
            gvz_graph.edge(str(edge[0]), str(edge[1]), label = '{}-{}\n{:.4f}/{:.4f}\n{:.4f}/{:.4f}'.format(DG[edge[0]][edge[1]]['edge_src_id'], DG[edge[0]][edge[1]]['edge_dst_id'],
                float(DG[edge[0]][edge[1]]['src_activity'].strip('\"')), float(DG[edge[0]][edge[1]]['dst_activity'].strip('\"')),
                float(DG[edge[0]][edge[1]]['src_act_ratio'].strip('\"')), float(DG[edge[0]][edge[1]]['dst_act_ratio'].strip('\"'))))
        else:
            gvz_graph.edge(str(edge[0]), str(edge[1]), label = '{}-{}'.format(DG[edge[0]][edge[1]]['edge_src_id'], DG[edge[0]][edge[1]]['edge_dst_id']))
    try:
        gvz_graph.render(view = False)
    except:
        print("CHECK: df_graph_visualize: draw error")


###################################################################
def graph_visualize(out_dir, save_name, rop_name_dict, act_plot = False):
    DG = nx.DiGraph(nx.drawing.nx_pydot.read_dot('{}/{}.dot'.format(out_dir, save_name)))
    gvz_list = [gvz.Digraph(format = 'png', filename = '{}/{}_opcode'.format(out_dir, save_name)), 
        gvz.Digraph(format = 'png', filename = '{}/{}_rtlop'.format(out_dir, save_name))]
    
    for color_scheme, gvz_graph in enumerate(gvz_list):
        gvz_graph.attr('node', fontsize = '50')
        gvz_graph.attr('edge', arrowsize = '2.4', fontsize = '30')
        for nodeid in DG.nodes:
            if color_scheme == 0: ### color scheme 1: coloring different node opcode ###    
                if DG.nodes[nodeid]['comp_type'] in ['fadd', 'fmul']:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '2')
                elif DG.nodes[nodeid]['comp_type'] in ['io_mem', 'write', 'internal_mem']:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '1')
                elif DG.nodes[nodeid]['comp_type'] in ['reg', 'icmp', 'or']:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '3')
                elif DG.nodes[nodeid]['comp_type'] in ['mux', 'select']:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '4')
                elif DG.nodes[nodeid]['comp_type'] in ['add', 'mul']:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '6')
                elif DG.nodes[nodeid]['comp_type'] in ['load', 'store']:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '5')
                else:
                    gvz_graph.attr('node', style = '', colorscheme = 'set28', color = '8')
            elif color_scheme == 1: ### color scheme 2: emphasizing rtl node ###
                if 'comp_name' in DG.nodes[nodeid]:
                    if DG.nodes[nodeid]['rtl_name'] in rop_name_dict:
                        gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '2')
                    else:
                        gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '3')
                elif 'opcode' in DG.nodes[nodeid]:
                    if DG.nodes[nodeid]['opcode'] == 'const':
                        gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '1')
                    else:
                        gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '3')
                else:
                    gvz_graph.attr('node', style = 'filled', colorscheme = 'set28', color = '3')

            if DG.nodes[nodeid]['comp_type'] in ['io_mem']:
                gvz_graph.node(str(nodeid), label = '{}\n{}'.format(DG.nodes[nodeid]['compid'], DG.nodes[nodeid]['comp_name']))
            elif DG.nodes[nodeid]['comp_type'] in ['internal_mem']:
                gvz_graph.node(str(nodeid), label = '{}\n{}'.format(DG.nodes[nodeid]['compid'], DG.nodes[nodeid]['rtl_name']))
            elif 'opcode' in DG.nodes[nodeid]:
                if DG.nodes[nodeid]['opcode'] == 'const':
                    gvz_graph.node(str(nodeid), label = '{}\n{}'.format(DG.nodes[nodeid]['compid'], DG.nodes[nodeid]['opcode']))
                else:
                    gvz_graph.node(str(nodeid), label = '{}\n{}'.format(DG.nodes[nodeid]['compid'], DG.nodes[nodeid]['comp_type']))
            else:
                gvz_graph.node(str(nodeid), label = '{}\n{}'.format(DG.nodes[nodeid]['compid'], DG.nodes[nodeid]['comp_type']))

        for edge in DG.edges:
            if act_plot == True:
                gvz_graph.edge(str(edge[0]), str(edge[1]), label='{}\n{:.4f}'.format(DG[edge[0]][edge[1]]['pin_in_index'], float(DG[edge[0]][edge[1]]['activity'].strip("\""))))
            else:
                gvz_graph.edge(str(edge[0]), str(edge[1]), label='{}'.format(DG[edge[0]][edge[1]]['pin_in_index']))

        gvz_graph.render(view = False)


###########################
def df_graph_construct(info_dir, fsm_node_dict, cdfg_fsm_node_dict, df_reg_dict, cop_dict, rop_dict):
    DG = nx.DiGraph()
    df_node_edge_construct(DG, fsm_node_dict, cdfg_fsm_node_dict, df_reg_dict, cop_dict)
    df_mem_insert(DG, rop_dict, cdfg_fsm_node_dict)
    df_reg_insert(DG)
    df_io_port_insert(DG, rop_dict)
    df_mem_connect(DG)
    df_op_bypass(DG, df_bypass_op)
    df_node_merge(DG, rop_dict)
    df_mem_path_merge(DG)
    df_cc_trim(DG)
    df_rtl_info_add(DG, cdfg_fsm_node_dict, rop_dict)
    DG.remove_nodes_from(list(nx.isolates(DG)))
    DG = nx.convert_node_labels_to_integers(DG)
    df_node_tracer_gen(DG, info_dir, cdfg_fsm_node_dict, cop_dict, rop_dict)
    df_edge_tracer_gen(DG, info_dir, cdfg_fsm_node_dict, cop_dict, rop_dict)
    nx.nx_pydot.write_dot(DG, '{}/DG.dot'.format(info_dir))
    if DG.number_of_edges() < PLOT_EDGE_LIMIT:
        df_graph_visualize(info_dir, 'DG', act_plot = False)


###########################################################################################
def op_extract(info_dir, IR_file, FSMD_file, RSC_file):
    IRinfo = ET.parse(IR_file).getroot()
    FSMDinfo = ET.parse(FSMD_file).getroot()
    RSCinfo = ET.parse(RSC_file).getroot()

    cdfg_node_dict, op_set = cdfg_node_explore(info_dir, IRinfo)
    fsm_node_dict, op_node_mapping_dict = fsm_node_explore(info_dir, FSMDinfo)
    cdfg_fsm_node_dict = cdfg_fsm_node_explore(info_dir, cdfg_node_dict, fsm_node_dict)
    rsc_dict = component_rsc_explore(info_dir, RSCinfo)
    global_dict = global_explore(info_dir, RSCinfo)
    cop_dict, rop_dict, rtl_id = c_rtl_op_explore(info_dir, IRinfo, cdfg_fsm_node_dict, rsc_dict)
    df_reg_dict = get_df_reg(FSMDinfo, op_node_mapping_dict, cdfg_fsm_node_dict)

    df_graph_construct(info_dir, fsm_node_dict, cdfg_fsm_node_dict, df_reg_dict, cop_dict, rop_dict)
    print('total number of rtl_operators: {}'.format(rtl_id))


###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Extract c-operators and rtl-operators from HLS, construct cdfg graph', epilog = '')
    parser.add_argument('kernel_name', help = "input: kernel_name of HLS function")
    parser.add_argument('--src_path', required  = True, help = "directory path of input files", action = 'store')
    parser.add_argument('--dest_path', required  = True, help = "directory path of output files", action = 'store')

    args = parser.parse_args()
    IR_file = '{}/{}.adb'.format(args.src_path, args.kernel_name)
    FSMD_file = '{}/{}.adb.xml'.format(args.src_path, args.kernel_name)
    RSC_file = '{}/{}.verbose.rpt.xml'.format(args.src_path, args.kernel_name)
    # Net_file = '{}/{}.verbose.rpt'.format(args.src_path, args.kernel_name)

    op_extract(args.dest_path, IR_file, FSMD_file, RSC_file)
