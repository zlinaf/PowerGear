import sys
import os
import io
import shutil
import argparse


#############################################################
def gen_script(fw, src_dir, prj_name, kernel_name, iob, bp, lp1, lp2):
    fw.write('open_project -reset "{}"\n'.format(prj_name))
    fw.write('set_top {}\n'.format(kernel_name))
    fw.write('add_files {}/{}.h\n'.format(src_dir, kernel_name))
    fw.write('add_files {}/{}.c\n'.format(src_dir, kernel_name))
    fw.write('open_solution "solution"\n')
    fw.write('set_part {xczu9eg-ffvb1156-2-i}\n')
    fw.write('create_clock -period 10 -name default\n')
    
    # io partition
    fw.write('set_directive_resource -core RAM_1P "{}" A\n'.format(kernel_name))
    fw.write('set_directive_array_partition -type cyclic -factor {} -dim 2 "{}" A\n'.format(iob, kernel_name))
    fw.write('set_directive_resource -core RAM_1P "{}" x\n'.format(kernel_name))
    fw.write('set_directive_interface -mode ap_fifo "{}" y_out\n'.format(kernel_name))

    # buffer partition
    fw.write('set_directive_array_partition -type cyclic -factor {} -dim 2 "{}" buff_A\n'.format(iob, kernel_name))
    fw.write('set_directive_array_partition -type cyclic -factor {} -dim 1 "{}" tmp1\n'.format(iob, kernel_name))
    fw.write('set_directive_array_partition -type cyclic -factor {} -dim 1 "{}" buff_x\n'.format(iob, kernel_name))
    fw.write('set_directive_array_partition -type cyclic -factor {} -dim 1 "{}" buff_y_out\n'.format(iob, kernel_name))

    if bp == 'd1d2':
        fw.write('set_directive_array_partition -type cyclic -factor {} -dim 1 "{}" buff_A\n'.format(iob, kernel_name))

    # buff reading loop pipeline/unroll
    fw.write('set_directive_pipeline "{}/lprd_2"\n'.format(kernel_name))
    fw.write('set_directive_unroll -factor {} "{}/lprd_2"\n'.format(iob, kernel_name))
    fw.write('set_directive_pipeline "{}/lpwr_1"\n'.format(kernel_name))
    fw.write('set_directive_unroll -factor {} "{}/lpwr_1"\n'.format(iob, kernel_name))
    
    # loop pipeline/unroll    
    if lp1[0] == 'p':
        fw.write('set_directive_pipeline "{}/lp1"\n'.format(kernel_name))
        fw.write('set_directive_unroll -factor {} "{}/lp1"\n'.format(lp1[1], kernel_name))
    elif lp1[2] == 'p':
        fw.write('set_directive_pipeline "{}/lp2"\n'.format(kernel_name))
        fw.write('set_directive_unroll -factor {} "{}/lp2"\n'.format(lp1[3], kernel_name))
    else:
        fw.write('set_directive_unroll -factor {} "{}/lp2"\n'.format(lp1[3], kernel_name))

    if lp2[0] == 'p':
        fw.write('set_directive_pipeline "{}/lp3"\n'.format(kernel_name))
        fw.write('set_directive_unroll -factor {} "{}/lp3"\n'.format(lp2[1], kernel_name))
    elif lp2[2] == 'p':
        fw.write('set_directive_pipeline "{}/lp4"\n'.format(kernel_name))
        fw.write('set_directive_unroll -factor {} "{}/lp4"\n'.format(lp2[3], kernel_name))
    else:
        fw.write('set_directive_unroll -factor {} "{}/lp4"\n'.format(lp2[3], kernel_name))
    
    fw.write('csynth_design\n')
    fw.write('close_project\n')
    fw.write('\n')


#############################################################
def gen_strategy(partition_factor, unroll_factor):
    lp_d = []

    for iob in partition_factor: # io/buffer partition
        if iob == 1:
            buff_p = ['d2']
        else:
            buff_p = ['d2', 'd1d2']

        lp1_d = []
        for out_p in ['n', 'p']:  # lp1
            if out_p == 'p':
                for i in unroll_factor:
                    if i <= iob:
                        lp1_d.append([out_p, str(i), 'n', '1'])
            else:
                for in_p in ['n', 'p']:
                    for i in unroll_factor:
                        if i <= iob:
                            lp1_d.append(['n', '1', in_p, str(i)])

        lp2_d = []
        for out_p in ['n', 'p']:  # lp2
            if out_p == 'p':
                for i in unroll_factor:
                    if i <= iob:
                        lp2_d.append([out_p, str(i), 'n', '1'])
            else:
                for in_p in ['n', 'p']:
                    for i in unroll_factor:
                        if i <= iob:
                            lp2_d.append(['n', '1', in_p, str(i)])

        for lp1 in lp1_d:
            for lp2 in lp2_d:
                for bp in buff_p:
                    lp_d.append([iob, lp1[0], lp1[1], lp1[2], lp1[3], lp2[0], lp2[1], lp2[2], lp2[3], bp])
    
    lp_d = sorted(lp_d, key = lambda x: x[0], reverse = False)
    
    return lp_d


#############################################################
def run_routine(fw, prj_dir, src_dir, kernel_name, script_num):
    fprj = open('./prj_to_run.txt', 'w+')
    ftotal = open('./prj_total.txt', 'w+')
    prj_cnt = 0
    total_cnt = 0
    partition_factor = [1, 2, 4, 8]
    unroll_factor = [1, 2, 4, 8]

    lp_d = gen_strategy(partition_factor, unroll_factor)
    
    for lp in lp_d:
        iob = lp[0]
        lp1 = [lp[1], lp[2], lp[3], lp[4]]
        lp2 = [lp[5], lp[6], lp[7], lp[8]]
        bp = lp[9]
        prj_name = 'io{}_l1{}{}{}{}_l3{}{}{}{}'.format(lp[0], lp[1], lp[2], lp[3], lp[4], lp[5], lp[6], lp[7], lp[8])
        if bp == 'd1d2':
            prj_name = prj_name + '_' + bp
        
        total_cnt = total_cnt + 1
        ftotal.write('{}\n'.format(prj_name))
        file_num = sum([len(files) for r, d, files in os.walk('{}/{}/solution/syn/verilog/'.format(prj_dir, prj_name))])
        if not os.path.exists('{}/{}/solution/syn/verilog/{}.v'.format(prj_dir, prj_name, kernel_name)) or file_num <= 5:
            gen_script(fw[prj_cnt % script_num], src_dir, prj_name, kernel_name, iob, bp, lp1, lp2)
            prj_cnt = prj_cnt + 1
            fprj.write('{}\n'.format(prj_name))

        if os.path.exists('{}/{}'.format(prj_dir, prj_name)) and file_num <= 5:
            print('Priorly terminated project: {}'.format(prj_name))

    fprj.close()
    ftotal.close()
    return prj_cnt, total_cnt


#############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generate scripts to automatically run HLS design', epilog = '')
    parser.add_argument('kernel_name', help = 'kernel name to process')
    parser.add_argument('--script_num', help = "number of scripts to run parallely", action = 'store', default = 1)
    parser.add_argument('--src_dir', help = 'directory of hls src as input', action = 'store')
    args = parser.parse_args()
    kernel_name = args.kernel_name
    script_num = args.script_num
    src_dir = args.src_dir

    prj_dir = '{}/../prj'.format(src_dir)
    if os.path.isdir(prj_dir):
        shutil.rmtree(prj_dir)
    os.mkdir(prj_dir)

    fw = []
    for i in range(0, script_num):
        file = open('{}/script_{}.tcl'.format(prj_dir, i), 'w+')
        fw.append(file)

    prj_cnt, total_cnt = run_routine(fw, prj_dir, src_dir, kernel_name, script_num)
    print('To run prj num = {}'.format(prj_cnt))
    print('Total prj num = {}'.format(total_cnt))

    for i in range(0, script_num):
        fw[i].close()
