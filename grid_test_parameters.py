""" Overwrites the parameter in the way they needed to be for testing the network generation """

import params as p


def set_parameters(mv):
    p.load_percentage = 100 if mv else 90
    p.rural_percentage = 0 if mv else 10
    p.suburban_percentage = 0
    p.urban_percentage = 0

    p.show_lv_buses = False
    p.show_loads = True
    p.show_generation = True
    p.show_storage = True
    p.show_switches = True
    p.show_base_stations = True
    p.show_base_station_connection = False
    p.show_power_grid = False
    p.create_small_world_network = False

    p.color_buses = 'blue'
    p.color_loads = 'green'
    p.color_generation = 'red'
    p.color_storage = 'purple'
    p.color_trafo = 'yellow'
    p.color_switch = 'pink'
    p.color_base_station = 'orange'

    p.mv_node_size = 50
    p.lv_node_size = 15

    p.base_station_range = 7.5
    p.show_radius = True

    p.recreate_lv_grids = False
    p.grid_test = False

    p.test_szenario = True
