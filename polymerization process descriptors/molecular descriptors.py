import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('smiles.csv')
smiles_list = df.iloc[:, ].astype(str).tolist()

descriptor_names = [desc[0] for desc in Descriptors._descList]


descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

def calc_descriptors(smiles):
 
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(descriptor_names)
        return descriptor_calculator.CalcDescriptors(mol)
    except Exception as e:
        return [np.nan] * len(descriptor_names)

results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
    delayed(calc_descriptors)(smi) for smi in tqdm(smiles_list, desc="running")
)
df_desc = pd.DataFrame(results, columns=descriptor_names)
df_out = pd.concat([df.reset_index(drop=True), df_desc.reset_index(drop=True)], axis=1)