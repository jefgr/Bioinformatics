from pathlib import Path
import csv
import pandas as pd

DATA_DIR = Path(".")
OUT = Path("CSV")
OUT.mkdir(exist_ok=True)



#convert the file name in the name of the parameters 
def clean_label(path):
    name = path.stem.replace("hierarchy_gene_list_", "")
    return name.replace("n_is_", "n=").replace("_pct_", ", pct=").replace("0pnt", "0.")


#etract all marker gene from a hierarchy_file
def read_gene_set(path):
    genes = set()
    with path.open(newline="") as f:
        for row in csv.reader(f):
            for cell in row:
                genes.update(g.strip() for g in str(cell).split(";") if g.strip())
    return genes


#do the jaccard similarity between two marker-gene sets
def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


files = sorted(DATA_DIR.glob("hierarchy_gene_list_*.csv"))
if not files:
    raise FileNotFoundError("No hierarchy_gene_list_*.csv files found.")

labels = [clean_label(p) for p in files]
gene_sets = {label: read_gene_set(path) for label, path in zip(labels, files)}

#Compare every parameter setting with every other setting
jaccard_matrix = [
    [jaccard(gene_sets[a], gene_sets[b]) for b in labels]
    for a in labels
]

pd.DataFrame(jaccard_matrix, index=labels, columns=labels).to_csv(
    OUT / "fig1_global_gene_jaccard_matrix.csv"
)

pd.DataFrame({
    "parameter_setting": labels,
    "n_unique_marker_genes": [len(gene_sets[label]) for label in labels],
}).to_csv(OUT / "fig1_global_gene_counts.csv", index=False)

