# automatically create and run multiple Vivado projects
python auto_run.py --hls_dir <bench_directory>/hls/prj --src_dir <bench_directory>/vivado/src --prj_dir <bench_directory>/vivado/prj
source <vivado_installation_directory>/settings64.sh
cd <bench_directory>/vivado/auto_run
vivado -mode batch -nojournal -nolog -notrace -source script_0.tcl