import os
import csv
import argparse
import random
import numpy as np
import networkx as nx
import torch
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results/q5"
DEFAULT_GRAPHS = [
    "Duke14",
    "Caltech36",
    "Amherst41",
    "Reed98",
    "Simmons81",
    "Haverford76",
    "Auburn71",
    "Vassar85",
    "Johns Hopkins55",
    "MIT8"
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_lcc(path):
    G = nx.read_gml(path)
    if nx.is_empty(G):
        raise ValueError(f"Graph at {path} is empty")
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()

class LabelPropagation:
    def __init__(self, adj_matrix, device=DEVICE):
        self.device = device
        self.n_nodes = adj_matrix.shape[0]
        
        degs = np.array(adj_matrix.sum(axis=1)).flatten()
        degs[degs == 0] = 1 
        inv_degs = 1.0 / degs
        
        coo = adj_matrix.tocoo()
        values = coo.data * inv_degs[coo.row] 
        
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(values).float()
        
        self.adj_norm = torch.sparse_coo_tensor(
            indices, values, torch.Size(coo.shape)
        ).to(self.device)

    def fit_predict(self, labels, mask_idx, n_classes, max_iter=200, tol=1e-3):
        Y = torch.zeros(self.n_nodes, n_classes, device=self.device)
        known_labels = labels[mask_idx]
        Y[mask_idx, known_labels] = 1.0
        Y_static = Y.clone()
        
        for i in range(max_iter):
            Y_old = Y.clone()
            Y = torch.sparse.mm(self.adj_norm, Y)
            Y[mask_idx] = Y_static[mask_idx]
            
            diff = torch.norm(Y - Y_old)
            if diff < tol:
                break
        
        return torch.argmax(Y, dim=1)

def get_clean_attribute_data(G, attr):
    data = []
    nodes = []
    node_attr = nx.get_node_attributes(G, attr)
    
    for n in G.nodes():
        val = node_attr.get(n, 0)
        if val != 0 and val is not None:
            nodes.append(n)
            data.append(val)
    return nodes, data

def run_experiment_on_graph(graph_path, attributes, fractions, seed):
    try:
        G = load_lcc(graph_path)
    except Exception as e:
        return []

    network_name = os.path.splitext(os.path.basename(graph_path))[0]
    adj = nx.adjacency_matrix(G, weight=None)
    lp_model = LabelPropagation(adj)
    node_to_idx = {n: i for i, n in enumerate(G.nodes())}
    results = []

    for attr in attributes:
        valid_nodes, raw_labels = get_clean_attribute_data(G, attr)
        
        if len(valid_nodes) < 100:
            continue
            
        le = LabelEncoder()
        encoded_labels = le.fit_transform(raw_labels)
        n_classes = len(le.classes_)
        full_labels_tensor = torch.zeros(G.number_of_nodes(), dtype=torch.long, device=DEVICE)
        valid_indices = [node_to_idx[n] for n in valid_nodes]
        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=DEVICE)
        encoded_labels_tensor = torch.tensor(encoded_labels, dtype=torch.long, device=DEVICE)
        full_labels_tensor[valid_indices_tensor] = encoded_labels_tensor

        rng = random.Random(seed)
        
        for frac in fractions:
            n_valid = len(valid_indices)
            n_remove = int(n_valid * frac)
            
            perm = rng.sample(range(n_valid), n_valid)
            remove_idx_rel = perm[:n_remove]
            keep_idx_rel = perm[n_remove:]
            
            remove_mask = valid_indices_tensor[remove_idx_rel]
            keep_mask = valid_indices_tensor[keep_idx_rel]
            
            preds = lp_model.fit_predict(full_labels_tensor, keep_mask, n_classes)
            y_pred = preds[remove_mask]
            y_true = full_labels_tensor[remove_mask]
            
            correct = (y_pred == y_true).sum().item()
            total = len(y_true)
            accuracy = correct / total if total > 0 else 0.0
            
            results.append({
                "network": network_name,
                "attribute": attr,
                "fraction_removed": frac,
                "accuracy": round(accuracy, 4),
                "n_removed": total,
                "n_classes": n_classes
            })
            
    return results

def print_summary_tables(all_rows):
    duke_rows = [r for r in all_rows if r["network"] == "Duke14"]
    if duke_rows:
        print("\n" + "="*60)
        print(f" TABLEAU 1 : Résultats détaillés pour Duke14 (Requis Q5d)")
        print("="*60)
        print(f"{'Attribute':<10} | {'Fraction Removed':<18} | {'Accuracy':<10}")
        print("-" * 45)
        duke_rows.sort(key=lambda x: (x["attribute"], x["fraction_removed"]))
        
        current_attr = ""
        for r in duke_rows:
            attr_display = r["attribute"] if r["attribute"] != current_attr else ""
            print(f"{attr_display:<10} | {r['fraction_removed']:<18} | {r['accuracy']:.4f}")
            current_attr = r["attribute"]

    agg = defaultdict(list)
    for r in all_rows:
        key = (r["attribute"], r["fraction_removed"])
        agg[key].append(r["accuracy"])
    
    print("\n" + "="*60)
    print(f" TABLEAU 2 : Moyenne Globale sur {len(set(r['network'] for r in all_rows))} Graphes (Pour Conclusion Q5e)")
    print("="*60)
    print(f"{'Attribute':<10} | {'Fraction Removed':<18} | {'Mean Accuracy':<15} | {'Std Dev':<10}")
    print("-" * 60)
    
    sorted_keys = sorted(agg.keys())
    current_attr = ""
    for attr, frac in sorted_keys:
        accs = agg[(attr, frac)]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        attr_display = attr if attr != current_attr else ""
        print(f"{attr_display:<10} | {frac:<18} | {mean_acc:.4f}          | {std_acc:.4f}")
        current_attr = attr
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graphs", nargs="*", default=None, help="Specific graphs to run. If empty, runs default set.")
    parser.add_argument("--fractions", nargs="*", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(OUT_DIR)
    
    target_graphs = []
    
    if args.graphs:
        for g in args.graphs:
            fname = g if g.endswith(".gml") else g + ".gml"
            target_graphs.append(fname)
    else:
        available_files = set(os.listdir(args.data_dir))
        for g_name in DEFAULT_GRAPHS:
            fname = g_name + ".gml"
            if fname in available_files:
                target_graphs.append(fname)
        
        if not target_graphs:
            all_gmls = sorted([f for f in available_files if f.endswith(".gml")])
            target_graphs = all_gmls[:10]

    print(f"Processing {len(target_graphs)} graphs...")
    
    all_rows = []
    attributes = ["dorm", "major", "gender"]
    
    for fname in target_graphs:
        path = os.path.join(args.data_dir, fname)
        rows = run_experiment_on_graph(path, attributes, args.fractions, args.seed)
        all_rows.extend(rows)

    if not all_rows:
        return

    csv_path = os.path.join(OUT_DIR, "label_propagation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["network", "attribute", "fraction_removed", "accuracy", "n_removed", "n_classes"])
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Results saved to: {csv_path}")
    print_summary_tables(all_rows)

if __name__ == "__main__":
    main()