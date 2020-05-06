import networkx as nx
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request
from text_tools import TextTools
from prepare_data import PrepareData
from clustering_manipulator import ClusteringManipulator
from webservice_config import WebserviceConfig
from config import config

app = Flask(__name__, static_url_path='')


@app.route('/document-clustering/', methods=['POST'])
def document_clustering():
    wc = WebserviceConfig(config)
    tt = TextTools()
    pr = PrepareData()
    cm = ClusteringManipulator()

    data = request.get_json()
    if data is None:
        return jsonify('No json Body is provided!')

    # validate input
    is_valid, validation_message = wc.validate_input(data)
    if not is_valid:
        return jsonify(validation_message)

    # get parameters
    parameters = wc.get_parameters(request.args)

    start_time = datetime.now()

    messages_df = pd.DataFrame(data['messages'])
    messages_df.drop(messages_df.columns.difference(config['data_columns']), axis=1, inplace=True)
    messages_df['id'] = messages_df['id'].astype('int64')
    messages_df.reset_index(inplace=True)
    messages_df.rename(columns={'id': 'origin_id', 'index': 'id'}, inplace=True)
    messages_df['id'] = messages_df['id'].astype(int)

    edges, nodes = tt.create_graph(messages_df.drop(['origin_id'], axis=1).copy())

    data_sim = pr.jaccard_sim(edges, nodes)
    data_sim, sim_df = pr.sim_nodes_detector(data_sim)

    # prune edges
    data_sim = data_sim[data_sim['jaccard_sim'] >= parameters['prune_thresh']]

    G = nx.from_pandas_edgelist(data_sim, source='source', target='target', edge_attr=True)

    partitions = cm.clustering_graph(G.copy(), noise_deletion=parameters['noise_deletion'], eps=parameters['eps'],
                                     min_samples=parameters['min_samples'])

    # add similar messages
    sim_df = sim_df.merge(partitions, how='left', left_on='id', right_index=True)
    sim_df.drop(['id'], axis=1, inplace=True)
    partitions = partitions.append(sim_df)

    messages_df = messages_df.merge(partitions, how='inner', left_on='id', right_index=True)
    messages_df.drop(['id'], axis=1, inplace=True)
    messages_df.rename(columns={'origin_id': 'id'}, inplace=True)
    messages_dic = messages_df.to_dict('records')

    result = {'number_of_clusters': len(partitions['class'].unique()), 'duration': str(datetime.now() - start_time),
              'messages': messages_dic}

    return jsonify(result)


@app.route('/similar-messages-prune/', methods=['POST'])
def similar_messages_prune():
    wc = WebserviceConfig(config)
    tt = TextTools()
    pr = PrepareData()

    data = request.get_json()
    if data is None:
        return jsonify('No json Body is provided!')

    # validate input
    is_valid, validation_message = wc.validate_input(data)
    if not is_valid:
        return jsonify(validation_message)

    # get parameters
    parameters = wc.get_parameters(request.args)

    start_time = datetime.now()

    messages_df = pd.DataFrame(data['messages'])
    messages_df.drop(messages_df.columns.difference(config['data_columns']), axis=1, inplace=True)
    messages_df['id'] = messages_df['id'].astype('int64')
    messages_df.reset_index(inplace=True)
    messages_df.rename(columns={'id': 'origin_id', 'index': 'id'}, inplace=True)
    messages_df['id'] = messages_df['id'].astype(int)

    edges, nodes = tt.create_graph(messages_df.drop(['origin_id'], axis=1).copy())

    data_sim = pr.jaccard_sim(edges, nodes)

    sim_list = pr.sim_messages(data_sim, nodes, parameters['sim_thresh'])

    messages_df = messages_df[~messages_df['id'].isin(sim_list)]
    messages_df.drop(['id'], axis=1, inplace=True)
    messages_df.rename(columns={'origin_id': 'id'}, inplace=True)
    messages_dic = messages_df.to_dict('records')

    result = {'duration': str(datetime.now() - start_time), 'messages': messages_dic}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
