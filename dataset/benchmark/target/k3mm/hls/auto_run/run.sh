# automatically run the HLS projects
python auto_run.py <kernel_name> --src_dir <bench_directory>/hls/src
source <vivado_installation_directory>/settings64.sh
cd <bench_directory>/hls/prj
vivado_hls -f script_0.tcl