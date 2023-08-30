import ast
import json
import random
import time
import pandapower.networks as pn
import pandas as pd
import simbench as sb
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import math
import params
import exception
import grid_test_parameters


def create_grids(name):
    """
    creates the chosen grid and returns it

    :param name: name of the grid to create

    :returns: pandapower grid
    """

    def create_simbench_grid(title):
        return sb.get_simbench_net(name)

    if name not in params.mv_grid_options and name not in params.lv_grid_options:
        raise exception.ParameterException('The chosen mv grid doesn\'t exist')
    if name == 'cigre_mv_all':
        return pn.create_cigre_network_mv(with_der='all')
    elif name == 'cigre_mv_pv_wind':
        return pn.create_cigre_network_mv(with_der='pv_wind')
    elif name == 'cigre_mv_without':
        return pn.create_cigre_network_mv(with_der=False)
    elif name == 'mv_oberrhein':
        return pn.mv_oberrhein()
    elif name == 'rural_1':
        return pn.create_synthetic_voltage_control_lv_network('rural_1')
    elif name == 'rural_2':
        return pn.create_synthetic_voltage_control_lv_network('rural_2')
    elif name == 'village_1':
        return pn.create_synthetic_voltage_control_lv_network('village_1')
    elif name == 'village_2':
        return pn.create_synthetic_voltage_control_lv_network('village_2')
    elif name == 'suburb_1':
        return pn.create_synthetic_voltage_control_lv_network('suburb_1')
    elif name == 'cigre_lv':
        return pn.create_cigre_network_lv()
    else:
        return create_simbench_grid(name)


def get_grid_size(grid_name):
    """
    this method returns the grid size of a given grid, as noted in grid_sized.csv

    :param grid_name: name of the grid. Must be contained in grid_sized.csv

    :return: tuple with the maximal expansion along the x- and y-axis
    """

    path = './grid_sizes.csv'
    sizes = pd.read_csv(path)

    x_size = sizes[sizes['name'] == grid_name].size_x.iloc[0]
    y_size = sizes[sizes['name'] == grid_name].size_y.iloc[0]

    return x_size, y_size


# This method returns the expansion of the values in a given series of positions
def get_grid_expansion(pos):
    """
    this method identifies the lowest and highest x and y coordinates in a given position set

    :param pos: set of positions

    :return: floats: lowest and highest x and y coordinates
    """

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for p in pos:
        if p is not None:
            if max_x is None:
                max_x = p[0]
                min_x = p[0]
                max_y = p[1]
                min_y = p[1]
            else:
                max_x = max(max_x, p[0])
                min_x = min(min_x, p[0])
                max_y = max(max_y, p[1])
                min_y = min(min_y, p[1])

    return max_x, min_x, max_y, min_y


def resize_grid(pos, name):
    """
    this method normalizes a set of positions to the size of the given grid

    :param pos: set of positions
    :param name: name of the grid to identify it's size

    :return: set of normalized positions
    """

    size = get_grid_size(name)
    normalized_pos = []

    expansion = get_grid_expansion(pos)
    max_x, min_x, max_y, min_y = expansion[0], expansion[1], expansion[2], expansion[3]

    if max_x - min_x == 0 or max_y - min_y == 0:
        raise exception.ValueException('Can\'t divide by zero. Grid might has an expansion of 0.')

    for x, y in pos:
        normalized_pos.append(((x - min_x) / (max_x - min_x) * size[0], (y - min_y) / (max_y - min_y) * size[1]))

    return normalized_pos


def set_neighbor_node(pos, node, distance, steps):
    """
    this method assigns a position close to a given ''related'' node, that is still free (recursive)

    :param pos: set of positions being searched for a free position
    :param node: ''related'' node that determines the starting position for the search
    :param distance: defines the needed distance to all other nodes
    :param steps: defines how many possible positions are suggested for a given radius (distance)

    :return: free position for the new node
    """

    step = 360 / steps
    valid_positions = [p for p in pos if p is not None]

    for i in range(steps):
        # create a new position that is shifted by ''distance'' in the direction of step (degree)
        new_pos = node[0] + distance * math.cos(i * step), node[1] + distance * math.sin(i * step)
        # calculating the distance from the new position to every other position
        distances = np.linalg.norm(np.array(valid_positions) - np.array(new_pos), axis=1)

        if np.all(distances > distance):
            return new_pos

    return set_neighbor_node(pos, node, distance / 2, steps * 2)


def manipulate_positions(pos, load, first_bus):
    """
    this method moves the positions of a lv graph, so it's first element aligns with a given node

    :param pos: positions of components of a lv graph
    :param load: node to determine the new position of the lv grid. This is to be a mv load node
    :param first_bus: represents the first element of a lv grid

    :return: set of new positions for the lv grid
    """

    if pos is None:
        return pos

    x_shift = load[0] - first_bus[0]
    y_shift = load[1] - first_bus[1]

    new_pos = [(p[0] + x_shift, p[1] + y_shift) if isinstance(p, tuple) else None for p in pos]
    return new_pos


def get_lv_dataframe(name):
    """
    reads and returns the dataframe of a wanted lv grid. They are stored in .csv files in the folder lv_grids

    :param name: wanted grid

    :return: DataFrame of the wanted grid
    """

    lv_grid_files = {
        'rural': params.rural_lv_grid_name,
        'suburban': params.suburban_lv_grid_name,
        'urban': params.urban_lv_grid_name
    }
    try:
        if name in lv_grid_files:
            df = pd.read_csv('./lv_grids/' + lv_grid_files[name] + '.csv')
            df.drop(columns=df.columns[0], inplace=True)
            df['pos'] = df['pos'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df['connection'] = df['connection'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and ',' in x else x)  # only tuples need eval

        else:
            df = None
    except FileNotFoundError:
        raise exception.ParameterException('No csv file for that grid. Check your spelling or recreate lv-grids.')

    return df


def modify_connections(df, index):
    """
    modifies the names of the stored connections. This is necessary, because the names need to be unique. If the same
    lv grid is appended twice or more, there is a need to append an index not only to the names, but also to the stored
    connections.

    :param df: model of the grid where the connections need to be indexed
    :param index: given index

    :return: modified model
    """

    for i, j in df.iterrows():
        if pd.notna(j.connection):
            if isinstance(j.connection, tuple):
                df.at[i, 'connection'] = (j.connection[0] + '_' + str(index), j.connection[1] + '_' + str(index))
            else:
                df.at[i, 'connection'] = j.connection + '_' + str(index)

    return df


def create_small_world_network(gr, df):
    """
    adds connections to the graph and model if the base node is not reachable from the current node within a defined
    number of steps. In the end the model is stored, because these were the last changes on it.

    :param gr: graph where edges can be added
    :param df: model where connections can be added

    :return: modified graph
    """

    base_node = list(gr.nodes)[0]
    for i in range(len(gr.nodes)):
        try:
            if nx.astar_path_length(gr, base_node, list(gr.nodes)[i]) > params.small_world_network_degree:
                gr.add_edge(base_node, list(gr.nodes)[i])
                connection = base_node, list(gr.nodes)[i]
                new_entry = pd.DataFrame({'name': 'small_world_connection_' + str(i), 'pos': None, 'type': 'line',
                                          'connection': [connection], 'voltage': None})
                df = pd.concat([df, new_entry], ignore_index=True)
        except nx.NetworkXNoPath:
            pass  # nothing you can do about if there is a node without connection in a given network

    df.to_csv('./export/dataframe.csv')
    return gr


# This method creates a dataframe for a given mv-grid containing lv-grids on the loads set before
def create_dataframe(grid, name, is_mv):
    """
    creates the model (DataFrame) for a given pandapower grid. This core method adds buses, their position at first.
    Lines, transformers, and switches will be added with their connection to other components. Loads, generation and
    storage will be added afterwards with a generated position close to their connected bus. If the grid is a mv grid, loads
    will be substituted with lv-grids as defined in params.py

    :param grid: pandapower grid
    :param name: name of the grid
    :param is_mv: defines the voltage of the given grid

    :return: model (DataFrame) of the grid
    """

    size = get_grid_size(name)
    model = pd.DataFrame()
    model['name'] = grid.bus.name
    model['pos'] = list(zip(grid.bus_geodata.x, grid.bus_geodata.y))
    model['type'] = 'bus'
    model['connection'] = None

    if is_mv:
        voltage = 'mv'
        # create a list with as many entries as mv loads, containing the grid types to attach assigned by percentage
        grid_list = list()
        if params.rural_percentage + params.suburban_percentage + params.urban_percentage + \
                params.load_percentage != 100:
            raise exception.ParameterException('Percentages for the attached lv grids don\'t add up to 100')
        grid_list.extend(['rural'] * int(params.rural_percentage / 100 * grid.load.index.size))
        grid_list.extend(['suburban'] * int(params.suburban_percentage / 100 * grid.load.index.size))
        grid_list.extend(['urban'] * int(params.urban_percentage / 100 * grid.load.index.size))
        grid_list.extend(['load'] * (grid.load.index.size - len(grid_list)))
        # using a seed for recreating the same result, if wanted
        if not isinstance(params.seed, int):
            raise exception.ParameterException('The seed has to be an integer')
        random.Random(params.seed).shuffle(grid_list)
    else:
        voltage = 'lv'
    model['voltage'] = voltage

    def add_to_model(n, p, t, c, v, l):
        """ method to add a component to the model """
        new_entry = pd.DataFrame({'name': n, 'pos': p, 'type': t, 'connection': c, 'voltage': v,
                                  'connection_length': l})
        return pd.concat([model, new_entry], ignore_index=True)

    def check_existing_lines(line_list, node1, node2):
        """ method to check if a line already exists """
        for (i, j) in line_list.iterrows():
            if j.type == 'line' and set(j.connection) == {node1, node2}:
                return j
            elif j.type == 'trafo' and (j[1] == node1 or j[1] == node2):
                if j.connection[0] == node1 or j.connection[0] == node2:
                    j.connection = j.connection[1]
                elif j.connection[1] == node1 or j.connection[1] == node2:
                    j.connection = j.connection[0]
                return j
        return None

    # create unique names where names were None
    for (i, j) in model.iterrows():
        if j['name'] is None:
            model['name'][i] = 'bus' + str(i)
        if not is_mv and grid.bus.vn_kv[i] >= 20:
            model['type'][i] = 'first_bus'

    # normalize the grid size
    model['pos'] = resize_grid(model['pos'], name)
    model.reset_index(inplace=True)

    # include lines
    for i, j in grid.line.iterrows():
        # using ['index'] instead of normal index, because in the pandapower grids they jumped some indices but still
        # use it as only reference. So this way iteration is easier.
        connection = model[model['index'] == j.from_bus].name.iloc[0], model[model['index'] == j.to_bus].name.iloc[0]
        name = 'line' + str(i) if j[0] is None else j[0]
        model = add_to_model(name, None, 'line', [connection], voltage, j.length_km)

    # include transformers
    for i, j in grid.trafo.iterrows():
        pos_bus1 = model[model['index'] == j.hv_bus].pos.iloc[0]
        pos_bus2 = model[model['index'] == j.lv_bus].pos.iloc[0]
        # place the trafo exactly between the two buses it is between
        trafo_pos = pos_bus2[0] + (pos_bus1[0] - pos_bus2[0]) / 2, pos_bus2[1] + (pos_bus1[1] - pos_bus2[1]) / 2
        name = 'trafo' + str(i) if j[0] is None else j[0]
        connection = (model[model['index'] == j.hv_bus].name.iloc[0],
                      model[model['index'] == j.lv_bus].name.iloc[0])
        model = add_to_model(name, [trafo_pos], 'trafo', [connection], voltage, None)

    # include switches
    lines = model[(model['type'] == 'line') | (model['type'] == 'trafo')]
    # switches were mostly just placed between two components, so they had no real connection. Later they will be
    # connected with the two components they are between of and the original connection will be removed.
    lines_to_remove = set([])

    for i, j in grid.switch.iterrows():
        name = 'switch' + str(i) if j[5] is None else j[5]
        pos_bus1 = model[model['index'] == j.bus].pos.iloc[0]
        pos_bus2 = tuple
        con = tuple

        if j.et == 'b':        # switch is between two buses
            pos_bus2 = model[model['index'] == j.element].pos.iloc[0]
            con = (model[model['index'] == j.element].name.iloc[0],
                   model[model['index'] == j.bus].name.iloc[0])
            line = check_existing_lines(lines, con[0], con[1])
            if line is not None and (not is_mv or params.show_switches):
                lines_to_remove.add(line[1])
        elif j.et == 't':      # switch is between a bus and a trafo
            trafos = model[model['type'] == 'trafo']
            trafos.reset_index(drop=True, inplace=True)
            pos_bus2 = trafos.pos[j.element]
            con = (model[model['index'] == j.bus].name.iloc[0], trafos.name[j.element])
            line = check_existing_lines(lines, con[0], con[1])
            if line is not None and (not is_mv or params.show_switches):
                model.loc[model['name'] == line[1], 'connection'] = line.connection
        elif j.et == 'l':      # switch is on a line
            bus1 = grid.line.from_bus[j.element]
            bus2 = grid.line.to_bus[j.element]
            pos_bus1 = model[model['index'] == bus1].pos.iloc[0]
            pos_bus2 = model[model['index'] == bus2].pos.iloc[0]
            con = (model[model['index'] == bus1].name.iloc[0], model[model['index'] == bus2].name.iloc[0])
            line = check_existing_lines(lines, con[0], con[1])
            if line is not None and (not is_mv or params.show_switches):
                lines_to_remove.add(line[1])
        # place the switch exactly between the two components it is between
        switch_pos = (pos_bus2[0] + (pos_bus1[0] - pos_bus2[0]) / 2, pos_bus2[1] + (pos_bus1[1] - pos_bus2[1]) / 2)
        model = add_to_model(name, [switch_pos], 'switch', [con], voltage, None)

    # remove lines that would be there twice
    for i in lines_to_remove:
        model = model.drop(model[model['name'] == i].index)

    # include loads
    for i in range(grid.load.index.size):
        # loads have no position, but for representation they get one close to the bus they are connected with
        tmp_pos = set_neighbor_node(model['pos'], model[model['index'] == grid.load.bus[i]].pos.iloc[0],
                                    size[0] * params.node_distance, 4)
        name = 'load' + str(i) if grid.load.name[i] is None else grid.load.name[i]
        model = add_to_model(name, [tmp_pos], 'load', model.name[grid.load.bus[i]], voltage, None)
        # if it is a mv grid attach the lv grids to the loads
        if is_mv:
            df = get_lv_dataframe(grid_list[i])
            if df is not None:
                # add an index to the names, so they won't appear twice by appending the grid twice
                df['name'] = df['name'] + '_' + str(i + 1)
                first_bus = df.loc[df['type'] == 'first_bus'].pos.iloc[0]
                df['pos'] = manipulate_positions(df.pos, tmp_pos, first_bus)
                # modify the connections in the same way as the names, so the stored components still exist
                df = modify_connections(df, i + 1)
                model = pd.concat([model, df], ignore_index=True)

    # include generation
    for i in range(grid.sgen.index.size):
        # generators have no position, but for representation they get one close to the bus they are connected with
        tmp_pos = set_neighbor_node(model['pos'], model[model['index'] == grid.sgen.bus[i]].pos.iloc[0],
                                    size[0] * params.node_distance, 4)
        name = 'gen' + str(i) if grid.sgen.name[i] is None else grid.sgen.name[i]
        model = add_to_model(name, [tmp_pos], 'gen', model.name[grid.sgen.bus[i]], voltage, None)

    # include storage
    for i in range(grid.storage.index.size):
        # storages have no position, but for representation they get one close to the bus they are connected with
        tmp_pos = set_neighbor_node(model['pos'], model[model['index'] == grid.storage.bus[i]].pos.iloc[0],
                                    size[0] * params.node_distance, 4)
        name = 'storage' + str(i) if grid.storage.name[i] is None else grid.storage.name[i]
        model = add_to_model(name, [tmp_pos], 'storage', model.name[grid.storage.bus[i]], voltage, None)

    # the index column is not needed anymore
    model.drop(columns=model.columns[0], inplace=True)

    return model


def recreate_lv_grids():
    """ recreates the model of every lv grid and saves it into a .csv file in the folder lv_grids """

    for i in params.lv_grid_options:
        df = create_dataframe(create_grids(i), i, False)
        df.to_csv('lv_grids/' + i + '.csv')


# This method generates a graph (including positions, colors and node size) out of the given dataframe
def create_graph(df):
    """
    generates a graph including positions, colors and node sizes from the given dataframe. Components that shall not be
    in the graph, will be removed from the model at first. For every other component, necessary nodes and edges will be
    added to the graph. If the creation of additional connections in the mv grid is wanted, this will be done after
    all mv components were added to the graph.

    :param df: model of the grid

    :return: an networkX graph, positions of all nodes, colors of the nodes, and size of the nodes
    """

    gr = nx.Graph()
    pos = dict()
    c_map = []
    node_size = []
    if params.create_small_world_network:
        sw_marker = True
    else:
        sw_marker = False

    if not params.show_lv_buses:
        df.drop(df[(df['type'] == 'bus') & (df['voltage'] == 'lv')].index, inplace=True)
    if not params.show_loads:
        df.drop(df[df['type'] == 'load'].index, inplace=True)
    if not params.show_generation:
        df.drop(df[df['type'] == 'gen'].index, inplace=True)
    if not params.show_storage:
        df.drop(df[df['type'] == 'storage'].index, inplace=True)
    if not params.show_switches:
        df.drop(df[df['type'] == 'switch'].index, inplace=True)

    def create_edge(node1, node2):
        """ adds an edge to the graph """
        if gr.has_node(node1) and gr.has_node(node2):
            gr.add_edge(node1, node2)

    def create_node(j, color):
        """ adds a node to the graph. Also saves its position, color and size. """
        if not gr.has_node(j[0]):
            gr.add_node(j[0])
            pos[j[0]] = j.pos
            c_map.append(color)
            node_size.append(params.lv_node_size if j.voltage == 'lv' else params.mv_node_size)

    for (i, j) in df.iterrows():
        # if additional connections are wanted, they will be generated before the first lv component is added
        if sw_marker and j.voltage == 'lv':
            gr = create_small_world_network(gr, df)
            sw_marker = False
        if j.type == 'line' or j.type == 'wireless_connection':
            if j.voltage == 'mv' or j.voltage is None or params.show_power_grid:
                create_edge(j.connection[0], j.connection[1])
        else:
            if not gr.has_node(j[0]):
                if j.type == 'bus':
                    create_node(j, params.color_buses)
                elif j.type == 'load':
                    create_node(j, params.color_loads)
                    if isinstance(j.connection, str) and (j.voltage == 'mv' or params.show_power_grid):
                        create_edge(j[0], j.connection)
                elif j.type == 'gen':
                    create_node(j, params.color_generation)
                    if isinstance(j.connection, str) and (j.voltage == 'mv' or params.show_power_grid):
                        create_edge(j[0], j.connection)
                elif j.type == 'storage':
                    create_node(j, params.color_storage)
                    if isinstance(j.connection, str) and (j.voltage == 'mv' or params.show_power_grid):
                        create_edge(j[0], j.connection)
                elif j.type == 'trafo':
                    create_node(j, params.color_trafo)
                    if isinstance(j.connection, tuple) and (j.voltage == 'mv' or params.show_power_grid):
                        create_edge(j[0], j.connection[0])
                        create_edge(j[0], j.connection[1])
                    elif isinstance(j.connection, str) and (j.voltage == 'mv' or params.show_power_grid):
                        create_edge(j[0], j.connection)
                elif j.type == 'base_station':
                    create_node(j, params.color_base_station)
                elif j.type == 'switch':
                    create_node(j, params.color_switch)
                    if isinstance(j.connection, tuple) and (j.voltage == 'mv' or params.show_power_grid):
                        create_edge(j[0], j.connection[0])
                        create_edge(j[0], j.connection[1])

    if sw_marker:
        gr = create_small_world_network(gr, df)  # if no lv grids are attached
    # if no additional connections are wanted, no further changes will be made on the model, so it can be saved
    if not params.create_small_world_network:
        df.to_csv('./export/dataframe.csv')

    return gr, pos, c_map, node_size


def draw_graph(gr, pos, c_map, node_size, base_stations):
    """
    visualizes the given graph with all other parameters. If chosen, the radius of the base stations is shown as well.
    Afterwards the plot will be saved as .png and .graphml file.

    :param gr: graph to visualize
    :param pos: positions for the nodes of the graph
    :param c_map: colors for the nodes of the graph
    :param node_size: size for the nodes of the graph
    :param base_stations: DataFrame with base stations to add the radii to the plot
    """

    plt.figure(figsize=(10, 10))
    nx.draw(gr, pos, node_color=c_map, with_labels=False, node_size=node_size, font_size=10)
    # include circles to represent the range of the base stations
    if params.show_radius and params.show_base_stations:
        for i, j in base_stations.iterrows():
            if j.type == 'base_station':
                radius = plt.Circle(j.pos, params.base_station_range, color='black', fill=False)
                plt.gca().add_patch(radius)
    # create and plot a legend
    legend_colors = dict({params.color_buses: 'buses', params.color_trafo: 'transformers', params.color_loads: 'loads',
                          params.color_generation: 'generators', params.color_switch: 'switches',
                          params.color_storage: 'storages', params.color_base_station: 'base stations'})
    legend_nodes = [plt.Line2D([], [], marker='o', color=color, label=label) for color, label in legend_colors.items()]
    legend_nodes.append(plt.scatter([], [], s=params.mv_node_size, marker='o', color='black', label='MV component'))
    legend_nodes.append(plt.scatter([], [], s=params.lv_node_size, marker='o', color='black', label='LV component'))
    plt.legend(handles=legend_nodes)
    # export the graph
    plt.savefig('export/graph.png')
    nx.write_graphml(gr, 'export/graph.graphml')
    with open('./export/graph.json', 'w+') as f:
        json.dump(json_graph.node_link_data(gr), f)


def create_base_stations(df):
    """
    creates a DataFrame with base stations. It is created a area wide network of base stations. Afterwards the base
    stations without a lv component in their range are removed again from the DataFrame.

    :param df: model of the grid to check for lv components within the range of base stations

    :return: DataFrame with all needed base stations
    """

    def distance(pos1, pos2):
        """ calculates the distance between two positions """
        x_diff = pos1[0] - pos2[0]
        y_diff = pos1[1] - pos2[1]
        dist = math.sqrt(x_diff ** 2 + y_diff ** 2)
        return dist

    # find the size of the grid
    expansion = get_grid_expansion(df.pos)

    # calculate the distance between the base stations (horizontally and vertically)
    radius = params.base_station_range * 1.5
    height = math.sqrt(params.base_station_range ** 2 - (params.base_station_range / 2) ** 2)

    station_counter = 0
    x_counter = expansion[1]
    # needed to indicate the offset between columns
    shifted = False

    base_stations = []

    # as long as the current position (x_counter) is smaller than the size of the grid, base stations are still needed.
    while x_counter - height <= expansion[0]:
        y_counter = expansion[3]
        if shifted:
            y_counter += radius
        while y_counter - radius / 1.5 <= expansion[2]:
            new_entry = pd.DataFrame(
                {'name': 'basestation' + str(station_counter), 'pos': [(x_counter, y_counter)], 'type': 'base_station',
                 'connection': None, 'voltage': 'mv'})
            base_stations.append(new_entry)
            station_counter += 1
            y_counter += 2 * radius

        x_counter += height
        if shifted:
            shifted = False
        else:
            shifted = True

    new_df = pd.concat(base_stations, ignore_index=True)

    drop_list = []
    wireless_connection_list = []
    for i in range(len(new_df)):
        is_needed = False
        for k in range(len(df.pos)):
            if new_df.pos[i] is not None and df.pos[k] is not None:
                # base stations are only relevant if a lv component is in its range
                if distance(new_df.pos[i], df.pos[k]) < params.base_station_range and df.voltage[k] == 'lv':
                    is_needed = True
                    # if it is wanted to display the connections, lines will be added to the DataFrame
                    if params.show_base_station_connection:
                        name = 'base_station_line' + str(k)
                        new_entry = pd.DataFrame(
                            {'name': name, 'pos': None, 'type': 'wireless_connection',
                             'connection': [(df.name[k], new_df.name[i])],
                             'voltage': None})
                        wireless_connection_list.append(new_entry)
        if not is_needed:
            drop_list.append(i)

    # drop all base stations that are not needed
    for i in drop_list:
        new_df = new_df.drop(i)
    if params.show_base_station_connection:
        params.show_base_stations = True
        wireless_connection_df = pd.concat(wireless_connection_list, ignore_index=True)
        new_df = pd.concat([new_df, wireless_connection_df], ignore_index=True)

    return new_df


def grid_test():
    grid_test_parameters.set_parameters(True)
    for i in params.mv_grid_options:
        params.mv_grid_name = i
        try:
            main()
            print('Generation of grid ' + i + ' successful')
            plt.close('all')
        except Exception:
            raise exception.GridException('There was an error during generation of grid ' + i)

    print('mv grid test successful')

    params.mv_grid_name = params.mv_grid_options[0]
    params.recreate_lv_grids = True
    main()
    params.recreate_lv_grids = False
    print('recreation of lv grids successful')

    grid_test_parameters.set_parameters(False)
    for i in params.lv_grid_options:
        params.rural_lv_grid_name = i
        try:
            main()
            print('Generation of grid ' + i + ' successful')
            plt.close('all')
        except Exception:
            raise exception.GridException('There was an error during generation of grid ' + i)

    print('lv grid test successful')
    print('grid test passed')


def main():
    """
    main method, that controls the procedure of creating the grids, creating the dataframe, creating the graph and
    visualize it. This method also measures the time needed.
    """

    if params.grid_test:
        params.test_szenario = True
        grid_test()
    else:
        start = time.time()
        if params.recreate_lv_grids:
            recreate_lv_grids()
        # to display the power grid properly, it has to be ensured, that some parameters have a certain value
        if params.show_power_grid:
            params.show_lv_buses = True
            params.show_radius = False
            params.show_base_station_connection = False
            params.show_base_stations = False
            params.show_switches = True
        recreation = time.time() - start
        mv_grid = create_grids(params.mv_grid_name)

        df = create_dataframe(mv_grid, params.mv_grid_name, True)
        if params.show_base_stations:
            base_stations = create_base_stations(df)
            df = pd.concat([df, base_stations], ignore_index=True)
        else:
            base_stations = None

        creation_df = time.time() - start
        tmp = create_graph(df)
        graph = tmp[0]
        positions = tmp[1]
        color_map = tmp[2]
        node_size = tmp[3]
        draw_graph(graph, positions, color_map, node_size, base_stations)
        total = time.time() - start
        if params.test_szenario:
            plt.show(block=False)
        else:
            plt.show()
        print('recreation time: ' + str(recreation), 'df creation time: ' + str(creation_df), 'total time: ' +
              str(total))


main()
