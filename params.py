""" Defines every parameter needed to adjust the output """

mv_grid_options = ['cigre_mv_without', 'cigre_mv_pv_wind', 'cigre_mv_all', '1-MV-rural--0-sw', '1-MV-rural--0-no_sw',
                   '1-MV-rural--1-sw', '1-MV-rural--1-no_sw', '1-MV-rural--2-sw', '1-MV-rural--2-no_sw',
                   '1-MV-semiurb--0-sw', '1-MV-semiurb--0-no_sw', '1-MV-semiurb--1-sw', '1-MV-semiurb--1-no_sw',
                   '1-MV-semiurb--2-sw', '1-MV-semiurb--2-no_sw', '1-MV-urban--0-sw', '1-MV-urban--0-no_sw',
                   '1-MV-urban--1-sw', '1-MV-urban--1-no_sw', '1-MV-urban--2-sw', '1-MV-urban--2-no_sw',
                   '1-MV-comm--0-sw', '1-MV-comm--0-no_sw', '1-MV-comm--1-sw', '1-MV-comm--1-no_sw', '1-MV-comm--2-sw',
                   '1-MV-comm--2-no_sw', 'mv_oberrhein']
# chose from above
mv_grid_name = 'cigre_mv_all'

lv_grid_options = ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1', 'cigre_lv', '1-LV-rural1--0-sw',
                   '1-LV-rural1--0-no_sw', '1-LV-rural1--1-sw', '1-LV-rural1--1-no_sw', '1-LV-rural1--2-sw',
                   '1-LV-rural1--2-no_sw','1-LV-rural2--0-sw', '1-LV-rural2--0-no_sw', '1-LV-rural2--1-sw',
                   '1-LV-rural2--1-no_sw', '1-LV-rural2--2-sw', '1-LV-rural2--2-no_sw', '1-LV-rural3--0-sw',
                   '1-LV-rural3--0-no_sw', '1-LV-rural3--1-sw', '1-LV-rural3--1-no_sw', '1-LV-rural3--2-sw',
                   '1-LV-rural3--2-no_sw', '1-LV-semiurb4--0-sw', '1-LV-semiurb4--0-no_sw', '1-LV-semiurb4--1-sw',
                   '1-LV-semiurb4--1-no_sw', '1-LV-semiurb4--2-sw', '1-LV-semiurb4--2-no_sw', '1-LV-semiurb5--0-sw',
                   '1-LV-semiurb5--0-no_sw', '1-LV-semiurb5--1-sw', '1-LV-semiurb5--1-no_sw', '1-LV-semiurb5--2-sw',
                   '1-LV-semiurb5--2-no_sw', '1-LV-urban6--0-sw', '1-LV-urban6--0-no_sw', '1-LV-urban6--1-sw',
                   '1-LV-urban6--1-no_sw', '1-LV-urban6--2-sw', '1-LV-urban6--2-no_sw']
# chose from above
rural_lv_grid_name = 'rural_1'
suburban_lv_grid_name = 'suburb_1'
urban_lv_grid_name = '1-LV-urban6--0-sw'

# percentages need to sum up to 100
load_percentage = 70
rural_percentage = 10
suburban_percentage = 10
urban_percentage = 10

# display parameters
show_lv_buses = False
show_loads = True
show_generation = True
show_storage = True
show_switches = True
show_base_stations = True
show_base_station_connection = False

show_power_grid = False

create_small_world_network = False  # the bigger the mv-grid, the longer the calculation will take!
# how many steps should a network have maximally to reach their base node
small_world_network_degree = 6

# node color definition
color_buses = 'blue'
color_loads = 'green'
color_generation = 'red'
color_storage = 'purple'
color_trafo = 'yellow'
color_switch = 'pink'
color_base_station = 'orange'

# node size definition
mv_node_size = 50
lv_node_size = 15

# radius that base stations have. Can be used to model different technologies
base_station_range = 7.5   # 5 Units represent 1 km
show_radius = True

seed = 4416  # will be used for the arrangement of lv_grids
node_distance = 1 / 40  # will be multiplied with the size of the grid (is relevant for placing nodes)

# is only needed if .csv files are missing or the code has changed
recreate_lv_grids = False    # will take longer

# disables everything else and just checks for errors with any grid during generation
grid_test = False    # will take some time, dependent on the number of gids
test_szenario = False
