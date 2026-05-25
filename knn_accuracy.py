from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(".")
OUT = Path("CSV")
OUT.mkdir(exist_ok=True)

N_REPEATS = 30
TEST_SIZE = 0.30
N_NEIGHBORS = 5
RANDOM_SEED = 1
NORMALIZE_COUNTS = True



#convert the file name in the name of the parameters 
def clean_label(path):
    name = path.stem.replace("hierarchy_gene_list_", "")
    return name.replace("n_is_", "n=").replace("_pct_", ", pct=").replace("0pnt", "0.")


#take the unique gene 
def read_gene_set(path):
    genes = set()
    with path.open(newline="") as f:
        for row in csv.reader(f):
            for cell in row:
                genes.update(g.strip() for g in str(cell).split(";") if g.strip())
    return genes



gene_files = sorted(DATA_DIR.glob("hierarchy_gene_list_*.csv"))
if not gene_files:
    raise FileNotFoundError("No hierarchy_gene_list_*.csv files found.")

#read the important file 
genes = pd.read_csv(DATA_DIR / "genes.tsv", sep="\t", header=None, names=["gene_id", "gene_symbol"])
barcodes = pd.read_csv(DATA_DIR / "barcodes.tsv", sep="\t", header=None, names=["barcode"])
labels_df = pd.read_csv(DATA_DIR / "clustering_result_from_seurat.tsv", sep="\t", header=None, names=["barcode", "cell_type"])


#load matrix and align cells with Seurat labels
X = mmread(DATA_DIR / "matrix.mtx").tocsr().T
barcode_to_pos = {bc: i for i, bc in enumerate(barcodes["barcode"])}
kept = labels_df[labels_df["barcode"].isin(barcode_to_pos)].copy()
X = X[[barcode_to_pos[bc] for bc in kept["barcode"]], :]
y = kept["cell_type"].to_numpy()


#normalize counts because scRNA-seq preprocessing
if NORMALIZE_COUNTS:
    lib_size = np.asarray(X.sum(axis=1)).ravel()
    lib_size[lib_size == 0] = 1.0
    X = X.multiply((10000.0 / lib_size)[:, None]).tocsr()
    X.data = np.log1p(X.data)


#link gene symbols to matrix columns
symbol_to_idx = {}
for i, symbol in enumerate(genes["gene_symbol"].astype(str)):
    symbol_to_idx.setdefault(symbol, i)


#repeated stratified train/test evaluation
splitter = StratifiedShuffleSplit(n_splits=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_SEED)
rows = []


#evaluation with the parameters settings
for path in gene_files:
    setting = clean_label(path)
    marker_genes = read_gene_set(path)
    gene_idx = [symbol_to_idx[g] for g in sorted(marker_genes) if g in symbol_to_idx]

    if not gene_idx:
        raise ValueError(f"No marker genes from {setting} were found in genes.tsv")

    X_sub = X[:, gene_idx]

    for repeat, (train_idx, test_idx) in enumerate(splitter.split(X_sub, y), start=1):
        model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        model.fit(X_sub[train_idx], y[train_idx])
        pred = model.predict(X_sub[test_idx])

        rows.append({
            "parameter_setting": setting,
            "repeat": repeat,
            "n_marker_genes_in_file": len(marker_genes),
            "n_marker_genes_found_in_matrix": len(gene_idx),
            "accuracy": accuracy_score(y[test_idx], pred),
            "macro_f1": f1_score(y[test_idx], pred, average="macro"),
        })

#summarise KNN
results = pd.DataFrame(rows)
order = sorted(results["parameter_setting"].unique())

summary = results.groupby("parameter_setting").agg(
    mean_accuracy=("accuracy", "mean"),
    sd_accuracy=("accuracy", "std"),
    mean_macro_f1=("macro_f1", "mean"),
    sd_macro_f1=("macro_f1", "std"),
    n_marker_genes_in_file=("n_marker_genes_in_file", "first"),
    n_marker_genes_found_in_matrix=("n_marker_genes_found_in_matrix", "first"),
    n_repeats=("repeat", "count"),
).reindex(order)

results.to_csv(OUT / "fig3_knn_accuracy_results.csv", index=False)
summary.to_csv(OUT / "fig3_knn_accuracy_summary.csv")

print("Saved raw KNN CSV files in", OUT)
print(summary)
