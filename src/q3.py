import os
import csv
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


ATTRIBUTES = ["student_fac", "major_index", "dorm", "gender"]
OUT_DIR = "results/q3"
FIG_DIR = os.path.join(OUT_DIR, "figures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_lcc(path):
    G = nx.read_gml(path)
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def numeric_assortativity(G):
    return nx.degree_assortativity_coefficient(G)


def categorical_assortativity(G, attr):
    values = nx.get_node_attributes(G, attr)
    if len(values) == 0:
        return None
    return nx.attribute_assortativity_coefficient(G, attr)


def list_gml_files(data_dir):
    return [f for f in sorted(os.listdir(data_dir)) if f.endswith(".gml")]


def process_one_graph(data_dir, fname):
    path = os.path.join(data_dir, fname)
    name = fname.replace(".gml", "")

    G = load_lcc(path)
    n = G.number_of_nodes()

    row = {"network": name, "n": n, "degree": numeric_assortativity(G)}
    for attr in ATTRIBUTES:
        row[attr] = categorical_assortativity(G, attr)

    return row


def plot_results(rows):
    ensure_dir(FIG_DIR)

    for attr in ["degree"] + ATTRIBUTES:
        xs = []
        ys = []

        for r in rows:
            if r.get(attr) is None:
                continue
            xs.append(r["n"])
            ys.append(r[attr])

        if len(ys) == 0:
            continue

        plt.figure()
        plt.scatter(xs, ys, alpha=0.6)
        plt.axhline(0, linestyle="--")
        plt.xscale("log")
        plt.xlabel("Network size (log scale)")
        plt.ylabel(f"{attr} assortativity")
        plt.title(f"{attr.capitalize()} assortativity vs network size")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{attr}_assortativity_vs_size.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.hist(ys, bins=20)
        plt.axvline(0, linestyle="--")
        plt.xlabel(f"{attr} assortativity")
        plt.ylabel("Frequency")
        plt.title(f"{attr.capitalize()} assortativity distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{attr}_assortativity_distribution.png"), dpi=200)
        plt.close()


def run_q3(data_dir="data", graph=None, limit=None, no_plots=False, workers=0):
    ensure_dir(OUT_DIR)

    files = list_gml_files(data_dir)

    if graph is not None:
        if graph.endswith(".gml"):
            target = graph
        else:
            target = graph + ".gml"

        if target not in files:
            raise FileNotFoundError(f"{target} not found in {data_dir}/")

        files = [target]

    if limit is not None:
        files = files[:limit]

    rows = []

    if len(files) == 1:
        rows = [process_one_graph(data_dir, files[0])]
    else:
        if workers == 0:
            workers = max(1, (os.cpu_count() or 2) - 1)

        if workers == 1:
            for fname in tqdm(files, desc="Processing graphs", unit="graph"):
                rows.append(process_one_graph(data_dir, fname))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(process_one_graph, data_dir, fname) for fname in files]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing graphs", unit="graph"):
                    rows.append(fut.result())

    csv_path = os.path.join(OUT_DIR, "assortativity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved assortativity table: {csv_path}")

    if not no_plots:
        plot_results(rows)
        print(f"Saved figures in: {FIG_DIR}")

    if len(rows) == 1:
        r = rows[0]
        print("\n--- Single graph results ---")
        print(f"network: {r['network']}")
        print(f"n (LCC): {r['n']}")
        for k in ["degree"] + ATTRIBUTES:
            print(f"{k} assortativity: {r.get(k)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--graph", default=None, help='Example: "Caltech36.gml" or "Caltech36"')
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--workers", type=int, default=0, help="0=auto, 1=serial")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_q3(
        data_dir=args.data_dir,
        graph=args.graph,
        limit=args.limit,
        no_plots=args.no_plots,
        workers=args.workers
    )


