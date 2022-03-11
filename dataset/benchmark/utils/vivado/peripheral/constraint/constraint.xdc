##-------------------------------------
## Clock Pinout
##-------------------------------------
# 300 MHz Clock on board
set_property IOSTANDARD DIFF_SSTL15 [get_ports clk_p]
set_property IOSTANDARD DIFF_SSTL15 [get_ports clk_n]
set_property PACKAGE_PIN AL8 [get_ports clk_p]
set_property PACKAGE_PIN AL7 [get_ports clk_n]

##-------------------------------------
## Pinin (SW)
##-------------------------------------
set_property PACKAGE_PIN AN14 [get_ports ap_rst]
set_property IOSTANDARD LVCMOS33 [get_ports ap_rst]

##-------------------------------------
## Pinout (LED)
##-------------------------------------
set_property PACKAGE_PIN AG14 [get_ports probe_out]
set_property IOSTANDARD LVCMOS33 [get_ports probe_out]
set_property PACKAGE_PIN AF13 [get_ports data_valid]
set_property IOSTANDARD LVCMOS33 [get_ports data_valid]
set_property PACKAGE_PIN AE13 [get_ports {data_out[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {data_out[0]}]
set_property PACKAGE_PIN AJ14 [get_ports {data_out[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {data_out[1]}]
set_property PACKAGE_PIN AJ15 [get_ports {data_out[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {data_out[2]}]
set_property PACKAGE_PIN AH13 [get_ports {data_out[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {data_out[3]}]
