# automatically extract the timing and resource utilization information of the Vivado projects
python info_extract.py --prj_dir <bench_directory>/vivado/prj
source <vivado_installation_directory>/settings64.sh
cd <bench_directory>/vivado/info_ext
vivado -mode batch -nojournal -nolog -notrace -source script_0.tcl