from rdkit import Chem
from rdkit.Chem import BRICS
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_fragments(
    smiles_list,
    min_frag_size=3,
    max_frag_size=40,
    exclude_small_frags=True,
    repeat_threshold=5
)
    
    all_frags = []
    filtered_frags = []

    for smi in smiles_list
        mol = Chem.MolFromSmiles(smi)
        if mol
            frags = list(BRICS.BRICSDecompose(mol))
            frag_counts = Counter(frags)

            for frag, cnt in frag_counts.items()
                fmol = Chem.MolFromSmiles(frag)
                if not fmol
                    continue
                n_atoms = fmol.GetNumAtoms()

              
                if n_atoms  min_frag_size and exclude_small_frags
                    continue
                if n_atoms  max_frag_size
                    continue

                # === 重复过滤 ===
                if cnt  repeat_threshold
                    all_frags.extend([frag]  cnt)
                else
                    filtered_frags.append(frag)

    return Counter(all_frags), set(filtered_frags)



def compute_z_score(fragment, N_high, N_total, count_high, count_total)
    mean = N_high  (count_total  N_total)
    std = np.sqrt(N_high  (count_total  N_total)  (1 - count_total  N_total)  ((N_total - N_high)  (N_total - 1)))
    if std == 0
        return 0
    return (count_high - mean)  std



def zscore_fragment_analysis(high_perf_smiles, all_smiles, frag_params)
    high_counts, high_filtered = get_fragments(high_perf_smiles, frag_params)
    total_counts, total_filtered = get_fragments(all_smiles, frag_params)

    N_high = len(high_perf_smiles)
    N_total = len(all_smiles)
    z_scores = []

    for frag in total_counts.keys()
        count_high = high_counts.get(frag, 0)
        count_total = total_counts.get(frag, 0)
        z = compute_z_score(frag, N_high, N_total, count_high, count_total)
        is_filtered = frag in high_filtered or frag in total_filtered
        z_scores.append((frag, count_high, count_total, z, is_filtered))

    df = pd.DataFrame(z_scores, columns=[Fragment_SMILES, Count_in_High, Count_in_Total, Z_score, Filtered])
    return df.sort_values(by=Z_score, ascending=False)


# === Step 4 加载数据 ===
def clean_smiles_column(series)
    return series.dropna().astype(str).str.strip()


print(📂 正在读取 fragment.xlsx ...)
data = pd.read_excel(fragment.xlsx)

high_perf_smiles = clean_smiles_column(data.iloc[, 0])
all_smiles = clean_smiles_column(data.iloc[, 1])





df_result = zscore_fragment_analysis(high_perf_smiles, all_smiles, frag_params)

output_file = fragment_zscore.xlsx
df_result.to_excel(output_file, index=False)


topN = 20
plt.figure(figsize=(10, 6))
plt.barh(df_result.head(topN)[Fragment_SMILES], df_result.head(topN)[Z_score], color=teal)
plt.gca().invert_yaxis()
plt.xlabel(Z-score)
plt.title(fTop {topN} Enriched Fragments (BRICS Controlled))
plt.tight_layout()
plt.show()