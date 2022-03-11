# program the bitstream to run onboard
# set the hardware to IDLE STATUS 
# next, you can measure the power consumption with Zynq UltraScale+ MPSoC Power Advantage Tool 2018.1

open_hw
connect_hw_server
open_hw_target

current_hw_device [get_hw_devices xczu9_0]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices xczu9_0] 0]
current_hw_device [get_hw_devices arm_dap_1]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices arm_dap_1] 0]
current_hw_device [get_hw_devices xczu9_0]
set_property PROBES.FILE {<project_directory>/status_probe.ltx} [get_hw_devices xczu9_0]
set_property FULL_PROBES.FILE {<project_directory>/status_probe.ltx} [get_hw_devices xczu9_0]
set_property PROGRAM.FILE {<project_directory>/bitstream.bit} [get_hw_devices xczu9_0]
program_hw_devices [get_hw_devices xczu9_0]
refresh_hw_device [lindex [get_hw_devices xczu9_0] 0]

set_property OUTPUT_VALUE 0 [get_hw_probes probe_out_OBUF -of_objects [get_hw_vios -of_objects [get_hw_devices xczu9_0] -filter {CELL_NAME=~"vio_inst"}]]
commit_hw_vio [get_hw_probes {probe_out_OBUF} -of_objects [get_hw_vios -of_objects [get_hw_devices xczu9_0] -filter {CELL_NAME=~"vio_inst"}]]

close_hw
