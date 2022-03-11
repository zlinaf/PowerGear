import sys
import os
import time
import shutil
import argparse


#############################################################
def uncompress_prj(compress_dir, run_dir, kernel_name, prj_name):
    os.system('7z x {}/{}.7z -o{}/overall_run/{} > /dev/null'.format(compress_dir, prj_name, run_dir, kernel_name))


#############################################################
def rm_after_extract(run_dir, kernel_name, prj_name):
    shutil.rmtree('{}/overall_run/{}/{}'.format(run_dir, kernel_name, prj_name))


#############################################################
def feature_extract(prj_name, kernel_name, run_dir, out_dir, tb_dir):
    if 'io1' in prj_name:
        iob = 1
    elif 'io2' in prj_name:
        iob = 2
    elif 'io4' in prj_name:
        iob = 4
    elif 'io8' in prj_name:
        iob = 8
    else:
        print('CHECK: iob value exceed range: {}'.format(prj_name))

    print('##### Step 1: extract operator info #####')
    os.system('mkdir {}/{}/preprocess'.format(out_dir, prj_name))
    os.system('python {a}/preprocess/op_extract.py --src_path {a}/overall_run/{f}/{b}/info --dest_path {c}/{d}/preprocess {e}'.format(a = run_dir, 
        b = prj_name, c = out_dir, d = prj_name, e = kernel_name, f = kernel_name))

    print('##### Step 2: annotate IR for node and trace activities #####')
    os.system('mkdir {}/{}/ir_revise'.format(out_dir, prj_name))
    os.system('./../ir_revise/build/bin/ir_revise {a} ./{e}/{b}/info/a.o.3.bc {c}/{d}/ir_revise/annotated.bc {c}/{d}/preprocess/cop_node_dict.txt {c}/{d}/ir_revise/operand_info.txt 2> {c}/{d}/ir_revise/log_ir_revise.txt'.format(
        a = kernel_name, b = prj_name, c = out_dir, d = prj_name, e = kernel_name))
    os.system('mkdir {}/{}/act_trace'.format(out_dir, prj_name))
    os.system('clang++ -c {a}/{b}/ir_revise/annotated.bc -o {a}/{b}/act_trace/annotated.o'.format(a = out_dir, b = prj_name))
    os.system('/usr/bin/c++ -fopenmp -O3 -fPIC -std=c++11 -o {a}/{b}/act_trace/main.o -c {c}/io{d}/main.cpp'.format(a = out_dir, b = prj_name, c = tb_dir, d = iob))
    os.system('/usr/bin/c++ -fopenmp -O3 -fPIC -std=c++11 -o {a}/{b}/act_trace/act_trace {a}/{b}/act_trace/main.o {a}/{b}/act_trace/annotated.o {c}/act_trace/build/rtlop_tracer.o {c}/act_trace/build/tracer.o'.format(a=out_dir, b=prj_name, c=run_dir))
    os.system('{a}/{b}/act_trace/act_trace {a}/{b}/act_trace node > {a}/{b}/act_trace/act_trace.log'.format(a = out_dir, b = prj_name))

    print('##### Step 3: annotate IR for edge and trace activities #####')
    os.system('./../ir_revise/build/bin/ir_revise {a} ./{e}/{b}/info/a.o.3.bc {c}/{d}/ir_revise/annotated.bc {c}/{d}/preprocess/cop_edge_dict.txt 2> {c}/{d}/ir_revise/log_ir_revise_edge_trace.txt'.format(
        a = kernel_name, b = prj_name, c = out_dir, d = prj_name, e = kernel_name))
    os.system('clang++ -c {a}/{b}/ir_revise/annotated.bc -o {a}/{b}/act_trace/annotated.o'.format(a=out_dir, b=prj_name))
    os.system('/usr/bin/c++ -fopenmp -O3 -fPIC -std=c++11 -o {a}/{b}/act_trace/main.o -c {c}/io{d}/main.cpp'.format(a=out_dir, b=prj_name, c=tb_dir, d=iob))
    os.system('/usr/bin/c++ -fopenmp -O3 -fPIC -std=c++11 -o {a}/{b}/act_trace/act_trace {a}/{b}/act_trace/main.o {a}/{b}/act_trace/annotated.o {c}/act_trace/build/rtlop_tracer.o {c}/act_trace/build/tracer.o'.format(a=out_dir, b=prj_name, c=run_dir))
    os.system('{a}/{b}/act_trace/act_trace {a}/{b}/act_trace edge > {a}/{b}/act_trace/act_trace.log'.format(a = out_dir, b = prj_name))

    print('##### Step 4: embed node and edge activities in the graph sample #####')
    os.system('mkdir {}/{}/graph'.format(out_dir, prj_name))
    os.system('python {a}/preprocess/feature_embed.py --src_path {b}/{c}/preprocess --dest_path {b}/{c}/graph {d}'.format(a = run_dir, 
        b = out_dir, c = prj_name, d = kernel_name))
    

#############################################################
def run_routine(app_dir, prj_list, kernel_name):
    run_dir = os.path.abspath('..')
    compress_dir = '{}/{}/hls/compressed'.format(app_dir, kernel_name)
    out_dir = '{}/{}/generated'.format(app_dir, kernel_name)
    tb_dir = '{}/{}/testbench'.format(app_dir, kernel_name)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if os.path.isdir('{}/overall_run/{}'.format(run_dir, kernel_name)):
        shutil.rmtree('{}/overall_run/{}'.format(run_dir, kernel_name))
    os.mkdir('{}/overall_run/{}'.format(run_dir, kernel_name))

    prj_cnt = 0
    for prj_name in prj_list:
        if os.path.exists('{}/{}/graph/graph_node_feature.csv'.format(out_dir, prj_name)) and \
            os.path.exists('{}/{}/graph/graph_edge_feature.csv'.format(out_dir, prj_name)) :
            continue
        elif os.path.isdir('{}/{}'.format(out_dir, prj_name)):
            shutil.rmtree('{}/{}'.format(out_dir, prj_name))
        os.system('mkdir {}/{}'.format(out_dir, prj_name))
        
        prj_cnt = prj_cnt + 1
        print('######################## #{}/{} Start processing prj: {} {} ########################'.format(prj_cnt, len(prj_list), kernel_name, prj_name))
        uncompress_prj(compress_dir, run_dir, kernel_name, prj_name)
        feature_extract(prj_name, kernel_name, run_dir, out_dir, tb_dir)
        rm_after_extract(run_dir, kernel_name, prj_name)
        print('### Finish processing prj: {} {}, at time: {} ###'.format(kernel_name, prj_name, time.asctime(time.localtime(time.time()))))

    shutil.rmtree('{}/overall_run/{}'.format(run_dir, kernel_name))
    return prj_cnt


#############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Feature extraction flow for one kernel', epilog = '')
    parser.add_argument('kernel_name', help = "input: kernel_name for feature extraction")
    parser.add_argument('--app_dir', help = "location of benchmark for feature extraction", action = 'store', default = './')
    args = parser.parse_args()

    kernel_name = args.kernel_name
    app_dir = args.app_dir
    compress_dir = '{}/{}/hls/compressed'.format(app_dir, kernel_name)
    h_prj_list = [fname.split(".")[0] for fname in os.walk('{}'.format(compress_dir)).__next__()[2]]
    prj_list = sorted(h_prj_list, reverse = False)
    
    start_time = float(time.time())
    prj_cnt = run_routine(app_dir, prj_list, kernel_name)
    end_time = float(time.time())
    duration = (end_time - start_time) / 60
    print('##### Total prj num = {} #####'.format(prj_cnt))
    print('##### Total exe time = {} minutes #####'.format(duration))

    