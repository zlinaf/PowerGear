import sys
import os
import io
import shutil
import argparse

#############################################################
def gen_script(fw, prj_dir, prj_name):
    fw.write('open_project {}/{}/prj.xpr\n'.format(prj_dir, prj_name))
    fw.write('open_run impl_exe\n')
    fw.write('report_utilization -hierarchical -file {}/{}/utilization.xls\n'.format(prj_dir, prj_name))
    fw.write('report_timing -file {}/{}/slack.xls -sort_by slack\n'.format(prj_dir, prj_name))
    fw.write('report_timing_summary -setup -hold -file {}/{}/timing.xls\n'.format(prj_dir, prj_name))
    fw.write('close_design\n')
    fw.write('close_project\n')
    fw.write('\n')

#############################################################
def run_routine(fw, prj_dir, prj_list, script_num = 1):
    prj_cnt = 0
    for prj_name in prj_list:
        if os.path.exists('{}/{}/prj.runs/impl_exe/wrapper.bit'.format(prj_dir, prj_name)) and (not os.path.exists('{}/{}/utilization.xls'.format(prj_dir, prj_name))):
            gen_script(fw[prj_cnt % script_num], prj_dir, prj_name)
            prj_cnt = prj_cnt + 1
        elif not os.path.exists('{}/{}/prj.runs/impl_exe/wrapper.bit'.format(prj_dir, prj_name)):
            print('Bitstream not exist: {}'.format(prj_name))
            # shutil.rmtree('{}/{}'.format(prj_dir, prj_name))
    return prj_cnt

#############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'automatic Vivado bitstream & report extraction', epilog = '')
    parser.add_argument('--script_num', help = "number of scripts to run parallely", action = 'store', default = 1)
    parser.add_argument('--prj_dir', help = "directory of the vivado projects", action = 'store')
    args = parser.parse_args()
    script_num = int(args.script_num)
    prj_dir = os.getcwd() + '/' + args.prj_dir
    if not os.path.exists('{}/../info_ext'.format(prj_dir)):
        os.system('mkdir {}/../info_ext'.format(prj_dir))

    fw = []
    for i in range(0, script_num):
        file = open('{}/../info_ext/script_{}.tcl'.format(prj_dir, i), 'w+')
        fw.append(file)

    prj_list = sorted(os.walk('{}'.format(prj_dir)).__next__()[1])
    prj_cnt = run_routine(fw, prj_dir, prj_list, script_num)
    print('Total prj num = {}'.format(prj_cnt))

    for i in range(0, script_num):
        fw[i].close()