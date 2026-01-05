import os
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import csv


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_lcc(path):
    G = nx.read_gml(path)
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def degree_distribution(G):
    degrees = [d for _, d in G.degree()]
    return Counter(degrees)


def plot_degree_distribution(G, title, outpath):
    dist = degree_distribution(G)
    x = list(dist.keys())
    y = list(dist.values())

    plt.figure()
    plt.scatter(x, y)
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Frequency (log)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def graph_stats(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2 * m / (n * (n - 1))
    global_clustering = nx.transitivity(G)
    mean_local_clustering = nx.average_clustering(G)
    return n, m, density, global_clustering, mean_local_clustering


def plot_degree_vs_clustering(G, title, outpath):
    deg = dict(G.degree())
    clust = nx.clustering(G)

    x = [deg[v] for v in G.nodes()]
    y = [clust[v] for v in G.nodes()]

    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel("Degree")
    plt.ylabel("Local Clustering Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def run_q2():
    networks = {
        "Caltech36": "data/Caltech36.gml",
        "MIT8": "data/MIT8.gml",
        "JohnsHopkins55": "data/Johns Hopkins55.gml",
    }

    out_dir = "results/q2"
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    rows = []
    for name, path in networks.items():
        G = load_lcc(path)

        dd_path = os.path.join(fig_dir, f"{name}_degree_distribution.png")
        plot_degree_distribution(G, f"{name} - Degree Distribution (LCC)", dd_path)

        dc_path = os.path.join(fig_dir, f"{name}_degree_vs_clustering.png")
        plot_degree_vs_clustering(G, f"{name} - Degree vs Local Clustering (LCC)", dc_path)

        n, m, density, gcc, mlcc = graph_stats(G)
        rows.append([name, n, m, density, gcc, mlcc])

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["network", "n_nodes_lcc", "m_edges_lcc", "density", "global_clustering", "mean_local_clustering"])
        writer.writerows(rows)

    print(f"Saved figures in: {fig_dir}")
    print(f"Saved metrics table: {csv_path}")


if __name__ == "__main__":
    run_q2()
