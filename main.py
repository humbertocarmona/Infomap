# %%

from infomap import Infomap
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main(tau=0.15, y_shift=-0.08):
    """

    d pᵢ
    --- = -(1-τ) ∑ⱼ Wʳᵢⱼ pᵢ - N τ pᵢ + (1-τ) ∑ⱼ Wʳⱼᵢ pⱼ + τ = 0     (1)
    dt

    onde N é o número de nós e Wʳᵢⱼ é o peso normalizado (veja abaixo) .
    sujeito a
        ∑ᵢ pᵢ = 1                                             (2)
        ∑ᵢⱼ Wʳᵢⱼ pᵢ = 1 ⇒ Wʳᵢⱼ = Wᵢⱼ/<Wᵢⱼ>                            (3)
            com
                <Wᵢⱼ> = ∑ᵢⱼ Wᵢⱼ pᵢ                                (4)

    o "fluxo" é calculado com

        ϕᵢ = ∑ⱼ Wʳᵢⱼ pᵢ                                           (5)

    para τ=1 a eq(1) implica pᵢ = 1/N, great

    nesse caso <Wᵢⱼ> =(1/N)∑ᵢⱼ Wᵢⱼ


    para τ=0

                    ∑ⱼ Wᵢⱼ pᵢ =  ∑ⱼ Wⱼᵢ pⱼ


    """

    # outputs clu, tree, ftree, json, csv, network, states
    im = Infomap()
    G = nx.DiGraph()

    df = pd.read_csv("./nodelist.csv", comment="#")
    pos = dict()
    pos2 = dict()

    for i in range(len(df)):
        id = df.id[i]
        x = df.x[i]
        y = df.y[i]
        pos[id] = np.array([x, y])
        pos2[id] = np.array([x, y + y_shift])

    df = pd.DataFrame(pd.read_csv("./edgelist.csv", comment="#"))
    for i, r in df.iterrows():
        im.add_link(r.s, r.d, weight=r.w)
        G.add_edge(r.s, r.d, weight=r.w)

    im.run(
        flow_model="directed",
        teleportation_probability=tau,
        seed=123,
        num_trials=1,
    )
    print(im.get_dataframe())
    # im.writeJsonTree('teste.json')
    # print(f"modules:{im.num_top_modules} modules codelength:{im.codelength}")

    edge_flow = dict()
    for link in im.flow_links:
        s = link[0]
        d = link[1]
        f = link[2]
        edge_flow[(s, d)] = f

    group = dict()
    flow = dict()
    n_flow = []
    for node in im.nodes:
        group[node.node_id] = node.module_id
        flow[node.node_id] = node.flow
        n_flow.append(node.flow)

    nx.set_node_attributes(G, group, "group")
    nx.set_node_attributes(G, flow, "flow")
    nx.set_edge_attributes(G, edge_flow, "flow")

    im.write_json("test.json", states=True)
    # nx.set_edge_attributes(G,nx.edge_betweenness(G),"bc")

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.axis("equal")

    desc = dict()
    labels = dict()
    color = np.zeros(G.number_of_nodes())
    total_flow = 0
    for n in G.nodes(data=True):
        i = int(n[0])
        gr = n[1]["group"]
        fl = n[1]["flow"]
        total_flow += fl
        labels[i] = f"{i}"
        desc[i] = f"{fl:0.3f}"
        color[i - 1] = gr
    print(f"total flow = {total_flow:0.3f}")
    # edge_labels = dict()

    edge_color = [e[2]["flow"] for e in G.edges(data=True)]

    width = [min(w["weight"], 5) for _, _, w in G.edges(data=True)]
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_color=color,
        edgecolors="k",
        vmin=0,
        vmax=max(1.2 * color),
        cmap="jet",
        labels=labels,
        width=width,
        edge_color=edge_color,
        edge_cmap=plt.cm.jet,
        connectionstyle="arc3,rad=0.4",
    )
    nx.draw_networkx_labels(G, pos2, labels=desc, clip_on=False)
    # ax.text(0.4, -0.27, "w12=5")
    # ax.text(0.4, 0.15, "w21=1")
    # ax.text(0.15, -0.4, "w13=1")
    # ax.text(0.4, 0.4, f"$\\tau$={tau}")
    # nx.draw_networkx_edge_labels(G, pos, label_pos=0.8,
    #                              edge_labels=edge_labels)
    # plt.axis("off")
    plt.show()
    return G


if __name__ == "__main__":
    G = main(tau=0.0)


# %%
