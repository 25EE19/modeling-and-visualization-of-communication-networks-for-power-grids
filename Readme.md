# Modeling and visualization of communication networks for power grids

This program creates a communication network for a selected combination of lv and mv benchmark power grids.


## Description
Selected lv grids will be attatched to the loads of the chosen mv grid. The components of the created grid will be transferred into a model (DataFrame). Based on these components, a communication network will be generated. Therefore communication in the mv part of the grid will use wired technology and the lv components will get a wireless connection to generated base stations. Wired communication follows the topology of the power grid and can be interpreted as PLC of fiber optics. Due to the option of adjusting the range of base stations, different wireless technologies like cellular of LTE450 can be simulated. This program was mainly developed to handle the szenario 0 in SimBench grids with full switch representation. Communication grids can be generated for all other SimBench grids as well, but there might be inaccuracies.   


## Requirements
To install and run the project, three things are needed:
- python (developed with version 3.10.11)
- pandapower library 
- simbench library


## How to use
The Programm provides many parameters to customize the output. All parameter settings can be done in the params.py file. Here a brief overview what each parameter does:
- mv_grid_name => determines which mv grid to use. Make sure the chosen grid is available in "mv_grid_options"
- rural_lv_grid_name => sets the lv grid with rural characteristics. Make sure the chosen grid is available in "lv_grid_options"
- suburban_lv_grid_name => sets the lv grid with suburban characteristics. Make sure the chosen grid is available in "lv_grid_options"
- urban_lv_grid_name => sets the lv grid with urban characteristics. Make sure the chosen grid is available in "lv_grid_options"
- load_percentage => determines the percentage of mv-loads, that stay loads
- rural_percentage => determines the percentage of mv-loads, that will be replaced by a rural lv grid
- suburban_percentage => determines the percentage of mv-loads, that will be replaced by a suburban lv grid
- urban_percentage => determines the percentage of mv-loads, that will be replaced by a urban lv grid
- show_lv_buses => determines whether the lv buses are to be displayed
- show_loads => determines whether the loads are to be displayed
- show_generation => determines whether the generators are to be displayed
- show_storage => determines whether the storage units are to be displayed
- show_switches => determines whether the switches are to be displayed
- show_base_stations => determines whether the base stations are to be displayed
- show_base_station_connection = determines whether the connection to the base stations shall be visualized as a line
- show_power_grid => determines whether the power grid shall be shown instead of the communication network. Therefore other parameters will be automatically affected.
- create_small_world_network => determines whether the wired communication network consisting of mv components, shall be turned into a small world network
- small_world_network_degree => determines the maximal steps needed to reach the base node from any other node in the small world network
- color_buses => sets the color for bus nodes
- color_loads => sets the color for load nodes
- color_generation => sets the color for generation nodes
- color_storage => sets the color for storage nodes
- color_trafo => sets the color for trafo nodes
- color_switch => sets the color for switch nodes
- color_base_station => sets the color for base station nodes
- mv_node_size => defines the node size for mv nodes
- lv_node_size => defines the node size for lv nodes
- base_station_range => defines the range of the base stations
- show_radius => determines whether the cells around the base stations shall be shown as a circle
- seed => gives the possibility to generate the same network again. A random seed will cause a different arrangement of lv grids on every execution
- node_distance => determines the distance between nodes. For nodes without position, this distance is used in relation to the grid size to calculate their new position.
- recreate_lv_grids => determines whether the models of the lv grids shall be recreated and saved in csv-files

If the folder "lv_grids" is empty, make sure to set "recreate_lv_grids" in the params file to True. The generated model and graph will be saved to the folder "export" after each execution. 


## License
MIT License

Copyright (c) 2023, Benjamin Grabbert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


