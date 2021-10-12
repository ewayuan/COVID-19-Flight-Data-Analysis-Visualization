import gzip
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Geo, Timeline
from pyecharts.globals import ChartType, SymbolType
import matplotlib.pyplot as plt
import os.path as osp
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import networkx as nx
from tqdm import tqdm


def open_csvgz(csvgz_path):
    with gzip.open(csvgz_path) as f:
        aircraft_data = pd.read_csv(f)
    return aircraft_data


def draw_aircraft_line(csvgz_path, target_airport=None):
    aircraft_data = open_csvgz(csvgz_path)
    aircraft_data = aircraft_data[['origin', 'destination', 'day']]

    total_rows = len(aircraft_data)
    aircraft_data.dropna(axis=0, inplace=True)
    print('%d rows of nan values were successfully dropped.' % (total_rows - len(aircraft_data)))

    ident_map = pd.read_csv('airports.csv')
    ident_map = ident_map[['ident', 'name', 'latitude_deg', 'longitude_deg']]
    ident_map.dropna(axis=0, inplace=True)
    ident_map.set_index(keys='ident', inplace=True)

    days = sorted(list(set(aircraft_data['day'])))
    tl = Timeline()
    for date in days:
        print('Rendering %s ...' % date)
        sub_aircraft_data = aircraft_data[aircraft_data['day'] == date]
        airport_set = list(set(sub_aircraft_data['origin']) | set(sub_aircraft_data['destination']))
        sub_ident_map = ident_map.loc[airport_set, :]

        geo = Geo().add_schema(
            maptype="world",
            itemstyle_opts=opts.ItemStyleOpts(color="#323c48", border_color="#111"),
        )

        for row_index, row in sub_ident_map.iterrows():
            geo.add_coordinate(row_index, row['longitude_deg'], row['latitude_deg'])
        geo.add(
            "Airports",
            [(row_index, i) for i, row_index in enumerate(sub_ident_map.index)],
            type_=ChartType.SCATTER,
            color="red",
        )
        if target_airport is None:
            geo.add(
                "Aircraft Line",
                [(row['origin'], row['destination']) for _, row in sub_aircraft_data.iterrows()],
                type_=ChartType.LINES,
                effect_opts=opts.EffectOpts(
                    symbol=SymbolType.ARROW, symbol_size=1, color="blue"
                ),
                linestyle_opts=opts.LineStyleOpts(curve=0.2),
            )
        else:
            sub_aircraft_data_from = sub_aircraft_data[sub_aircraft_data['origin'] == target_airport]
            sub_aircraft_data_to = sub_aircraft_data[sub_aircraft_data['destination'] == target_airport]

            geo.add(
                "Aircraft Line From %s" % target_airport,
                [(row['origin'], row['destination']) for _, row in sub_aircraft_data_from.iterrows()],
                type_=ChartType.LINES,
                effect_opts=opts.EffectOpts(
                    symbol=SymbolType.ARROW, symbol_size=5, color="blue"
                ),
                linestyle_opts=opts.LineStyleOpts(curve=0.2, color='blue'),
            )
            geo.add(
                "Aircraft Line To %s" % target_airport,
                [(row['origin'], row['destination']) for _, row in sub_aircraft_data_to.iterrows()],
                type_=ChartType.LINES,
                effect_opts=opts.EffectOpts(
                    symbol=SymbolType.ARROW, symbol_size=5, color="green"
                ),
                linestyle_opts=opts.LineStyleOpts(curve=0.2, color='green'),
            )

        geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False)) \
            .set_global_opts(title_opts=opts.TitleOpts(title="Aircraft Line %s" % date))
        tl.add(geo, date)

    tl.render("geo_lines_background.html")


def airport_distribution(path):
    data_path = [osp.join(path, p) for p in os.listdir(path) if p.endswith('gz')]
    data_path = sorted(data_path)

    ident_map = pd.read_csv('airports.csv')
    ident_map = ident_map[['ident', 'name', 'latitude_deg', 'longitude_deg']]
    ident_map.dropna(axis=0, inplace=True)
    ident_map.set_index(keys='ident', inplace=True)

    tl = Timeline()

    for csvgz_path in data_path:
        date = os.path.split(csvgz_path)[-1]
        date = date.split('_')[1]
        aircraft_data = open_csvgz(csvgz_path)
        aircraft_data = aircraft_data[['origin', 'destination']]

        total_rows = len(aircraft_data)
        aircraft_data.dropna(axis=0, inplace=True)
        print('Rendering %s ...' % date)
        print('%d rows of nan values were successfully dropped.' % (total_rows - len(aircraft_data)))

        aircraft_num = aircraft_data.groupby('origin').agg('count')

        geo = Geo().add_schema(
            maptype="world",
            itemstyle_opts=opts.ItemStyleOpts(color="#323c48", border_color="#111"),
        )
        aircraft_num_list = []

        for row_index, row in aircraft_num.iterrows():
            try:
                ident_row = ident_map.loc[row_index]
                aircraft_num_list.append([str(row_index),int(row['destination'])])
                geo.add_coordinate(row_index, ident_row['longitude_deg'], ident_row['latitude_deg'])
            except KeyError:
                continue

        geo.add(
            "Airports",
            aircraft_num_list
        )


        geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False)) \
            .set_global_opts(visualmap_opts=opts.VisualMapOpts(min_=0,max_=150),title_opts=opts.TitleOpts(title="Airports %s" % date))
        tl.add(geo, date)

    tl.render("airports.html")



def flights_count_distribution(path):
    data_path = [osp.join(path, p) for p in os.listdir(path) if p.endswith('gz')]
    data_path = sorted(data_path)
    date_list = []
    aircraft_num = []
    for p in data_path:
        date = p.split('_')[-2]
        date = date[:6]
        print('Processing %s...' % date)
        date_list.append(date)
        data = open_csvgz(p)
        aircraft_num.append(len(data))

    plt.plot(date_list, aircraft_num, '-*')
    plt.xlabel('Date')
    plt.ylabel('Aircraft Number')
    plt.xticks(rotation=90)
    plt.show()


def shotest_path_distribution(csvgz_path):
    aircraft_data = open_csvgz(csvgz_path)
    aircraft_data = aircraft_data[['origin', 'destination', 'day']]

    total_rows = len(aircraft_data)
    aircraft_data.dropna(axis=0, inplace=True)
    print('%d rows of nan values were successfully dropped.' % (total_rows - len(aircraft_data)))

    date = csvgz_path.split('_')[-2]
    date = date[:6]

    graph = nx.Graph()
    for _, row in tqdm(aircraft_data.iterrows(), total=len(aircraft_data)):
        origin = row['origin']
        destination = row['destination']
        graph.add_edge(origin, destination)
    dist_list = []
    for source, target_dict in tqdm(nx.shortest_path_length(graph), total=graph.number_of_nodes()):
        for target, dist in target_dict.items():
            dist_list.append(dist)
    dist_list = np.asarray(dist_list)
    x, x_count = np.unique(dist_list, return_counts=True)
    plt.title('Shortest Path Distribution of %s' % date)
    plt.bar(x, x_count,color=['r','g','b','c','m','y','k'])
    plt.xlabel('Distance')
    plt.ylabel('Number')
    plt.show()


if __name__ == '__main__':
    data_root = r'E:\DownLoad\flights'
    airport_distribution(data_root)

    draw_aircraft_line(osp.join(data_root, 'flightlist_20190101_20190131.csv.gz'),'YLVK')
    flights_count_distribution(data_root)
    shotest_path_distribution(osp.join(data_root, 'flightlist_20190501_20190531.csv.gz'))
    shotest_path_distribution(osp.join(data_root, 'flightlist_20200501_20200531.csv.gz'))
