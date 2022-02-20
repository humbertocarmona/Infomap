# %%
from curses.ascii import controlnames
from xml.etree.ElementTree import Comment
from infomap import Infomap
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # outputs clu, tree, ftree, newick, json, csv, network, states
    im = Infomap()
    G = nx.DiGraph()

    df = pd.read_csv('./nodelist.csv', comment='#')
    pos = dict()
    pos2 = dict()
    for i, r in df.iterrows():
        pos[i] = np.array([r.x, r.y])
        pos2[i] = np.array([r.x, r.y+0.14])

    df = pd.read_csv('./edgelist.csv', comment='#')
    for i, r in df.iterrows():
        im.add_link(r.s, r.d, weight=r.w)
        G.add_edge(r.s, r.d, weight=r.w)

    im.run(flow_model="directed",
           two_level=True,
           ftree=True,
           output=['ftree', 'network'],
           out_name="./test.tree",
        #    to_nodes=False,
           seed=123
           )
    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    group = dict()
    flow = dict()
    for node in im.nodes:
        group[node.node_id] = node.module_id
        flow[node.node_id] = node.flow

    nx.set_node_attributes(G, group, 'group')
    nx.set_node_attributes(G, flow, 'flow')

    # im.write_json('test.json', states=True)

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axis('equal')

    desc = dict()
    labels = dict()
    color = np.zeros(G.number_of_nodes())
    total_flow = 0
    for n in G.nodes(data=True):
        i = int(n[0])
        gr = n[1]['group']
        fl = n[1]['flow']
        total_flow += fl
        labels[i] = f"{i}"
        desc[i] = f"{fl:0.3f}"
        color[i] = gr
    print(f'total flow = {total_flow:0.3f}')
    # edge_labels = dict()

    edge_color = [e[2]['weight'] for e in G.edges(data=True)]

    width = [min(w['weight'], 5) for u, v, w in G.edges(data=True)]
    nx.draw_networkx(G, pos=pos,  ax=ax,
                     node_color=color, edgecolors="k",
                     vmin=0, vmax=3,
                     cmap='jet',
                     labels=labels,
                     width=width,
                     edge_color=edge_color,
                     edge_cmap=plt.cm.jet,
                     connectionstyle="arc3,rad=0.4")
    nx.draw_networkx_labels(G, pos2, labels=desc, clip_on=False)

    # nx.draw_networkx_edge_labels(G,pos,
    #                              label_pos=0.8, edge_labels=edge_labels)
    # # plt.axis('off')

    return 0


if __name__ == "__main__":
    main()

    # %%
# %%

# %%
