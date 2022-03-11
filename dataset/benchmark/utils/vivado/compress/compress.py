import sys
import os
import io
import subprocess
import time
import datetime
import numpy as np 
import math
import shutil
import argparse


#############################################################
def remove_redundant(prj_dir, prj_name):
    os.system('cp {a}/{b}/prj.runs/impl_exe/wrapper.bit {a}/{b}/bitstream.bit'.format(a = prj_dir, b = prj_name))
    os.system('cp {a}/{b}/prj.runs/impl_exe/wrapper.ltx {a}/{b}/status_probe.ltx'.format(a = prj_dir, b = prj_name))
    if os.path.exists('{}/{}/prj.cache'.format(prj_dir, prj_name)):
        shutil.rmtree('{}/{}/prj.cache'.format(prj_dir, prj_name))
    if os.path.exists('{}/{}/prj.hw'.format(prj_dir, prj_name)):
        shutil.rmtree('{}/{}/prj.hw'.format(prj_dir, prj_name))
    if os.path.exists('{}/{}/prj.ip_user_files'.format(prj_dir, prj_name)):
        shutil.rmtree('{}/{}/prj.ip_user_files'.format(prj_dir, prj_name))
    if os.path.exists('{}/{}/prj.runs'.format(prj_dir, prj_name)):
        shutil.rmtree('{}/{}/prj.runs'.format(prj_dir, prj_name))
    if os.path.exists('{}/{}/prj.sim'.format(prj_dir, prj_name)):
        shutil.rmtree('{}/{}/prj.sim'.format(prj_dir, prj_name))
    if os.path.exists('{}/{}/prj.srcs'.format(prj_dir, prj_name)):
        shutil.rmtree('{}/{}/prj.srcs'.format(prj_dir, prj_name))
    os.remove('{}/{}/prj.xpr'.format(prj_dir, prj_name))
    

#############################################################
def compress(prj_dir, compress_dir, prj_name):
    os.system('7z a {c}/{a}.7z {b}/{a}'.format(a = prj_name, b = prj_dir, c = compress_dir))


#############################################################
def remove_project(prj_dir, prj_name):
    shutil.rmtree('{}/{}'.format(prj_dir, prj_name))
    

#############################################################
def run_routine(prj_list, prj_dir, compress_dir):
    prj_cnt = 0
    for prj_name in prj_list:
        if os.path.exists('{}/{}/prj.runs/impl_exe/wrapper.bit'.format(prj_dir, prj_name)) and \
            os.path.exists('{}/{}/prj.runs/impl_exe/wrapper.ltx'.format(prj_dir, prj_name)) and \
            os.path.exists('{}/{}/slack.xls'.format(prj_dir, prj_name)) and \
            os.path.exists('{}/{}/timing.xls'.format(prj_dir, prj_name)) and \
            os.path.exists('{}/{}/utilization.xls'.format(prj_dir, prj_name)):
            prj_cnt = prj_cnt + 1
            remove_redundant(prj_dir, prj_name)
            compress(prj_dir, compress_dir, prj_name)
            remove_project(prj_dir, prj_name)
        elif not os.path.exists('{}/{}/prj.runs/impl_exe/wrapper.bit'.format(prj_dir, prj_name)):
            print('Bitstream not exist: {}'.format(prj_name))
        elif not os.path.exists('{}/{}/prj.runs/impl_exe/wrapper.ltx'.format(prj_dir, prj_name)):
            print('Debug probe not exist: {}'.format(prj_name))
        else:
            print('Info not extract: {}'.format(prj_name))

    return prj_cnt


#############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'automatic Vivado designs compression script', epilog = '')
    parser.add_argument('--prj_dir', help = 'directory of Vivado project as input', action = 'store')
    args = parser.parse_args()
    prj_dir = os.getcwd() + '/' + args.prj_dir
    compress_dir = prj_dir + '/../compressed'
    if not os.path.exists(compress_dir):
        os.system('mkdir {}'.format(compress_dir))

    prj_list = sorted(os.walk('{}'.format(prj_dir)).__next__()[1])
    prj_cnt = run_routine(prj_list, prj_dir, compress_dir)

    print('Finished compressing vivado projects: project num = {}'.format(prj_cnt))
