import os
import argparse


################################################################
def compress_remove(prj_name, prj_dir, compress_dir):
    if not os.path.exists('{}/{}.7z'.format(compress_dir, prj_name)):
        os.system('mkdir {}/{}'.format(compress_dir, prj_name))
        os.system('mkdir {}/{}/info'.format(compress_dir, prj_name))
        os.system('cp -r {}/{}/solution/syn/verilog/ {}/{}'.format(prj_dir, prj_name, compress_dir, prj_name))
        os.system('cp -r {}/{}/solution/syn/report/ {}/{}'.format(prj_dir, prj_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/solution.directive {}/{}/report'.format(prj_dir, prj_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/solution.log {}/{}/report'.format(prj_dir, prj_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/.autopilot/db/a.o.3.bc {}/{}/info'.format(prj_dir, prj_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/.autopilot/db/{}.adb.xml {}/{}/info'.format(prj_dir, prj_name, kernel_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/.autopilot/db/{}.verbose.rpt.xml {}/{}/info'.format(prj_dir, prj_name, kernel_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/.autopilot/db/{}.adb {}/{}/info'.format(prj_dir, prj_name, kernel_name, compress_dir, prj_name))
        os.system('cp {}/{}/solution/.autopilot/db/{}.verbose.rpt {}/{}/info'.format(prj_dir, prj_name, kernel_name, compress_dir, prj_name))
        
        os.system('7z a {}/{}.7z {}/{}'.format(compress_dir, prj_name, compress_dir, prj_name))
        os.system('rm -rf {}/{}'.format(compress_dir, prj_name))


################################################################
def run_routine(prj_list, prj_dir, compress_dir):
    prj_cnt = 0
    
    for prj_name in prj_list:
        file_num = sum([len(files) for r, d, files in os.walk('{}/{}/solution/syn/verilog/'.format(prj_dir, prj_name))])
        if os.path.exists('{}/{}/solution/syn/verilog/{}.v'.format(prj_dir, prj_name, kernel_name)) and file_num >= 3 and not os.path.exists('{}/{}.7z'.format(compress_dir, prj_name)):
            compress_remove(prj_name, prj_dir, compress_dir)
            prj_cnt = prj_cnt + 1

    return prj_cnt


################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Automatic HLS designs compress script', epilog = '')
    parser.add_argument('kernel_name', help = 'kernel name to process')
    parser.add_argument('--in_dir', help = 'directory of hls project as input', action = 'store')
    parser.add_argument('--out_dir', help = "directory to store the compressed output", action = 'store')
    args = parser.parse_args()
    kernel_name = args.kernel_name
    prj_dir = os.getcwd() + '/' + args.in_dir
    compress_dir = os.getcwd() + '/' + args.out_dir + '/compressed'

    if not os.path.exists(compress_dir):
        os.system('mkdir {}'.format(compress_dir))

    prj_list = os.walk('{}'.format(prj_dir)).__next__()[1]
    prj_list = sorted(prj_list)
    prj_cnt = run_routine(prj_list, prj_dir, compress_dir)
    print('Finished compressing project num = {}'.format(prj_cnt))
