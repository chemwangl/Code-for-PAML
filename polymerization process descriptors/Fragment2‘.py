

import pandas as pd
from rdkit import Chem

## -OH -OH [Re]
# -OH -NH2 [O] [N]
## -NH2 -NH2  [Rn]
## -SH [Rb]

def replace_substructure(molecule, substructure, replacement):
    return Chem.ReplaceSubstructs(molecule, substructure, replacement)[0]

def perform_batch_replacements1(molecules, replacement_rules):  
    replaced_molecules = []
    for smiles in molecules:
        result_smiles = Chem.MolFromSmiles(smiles)
        for rule in replacement_rules:
            substructure = Chem.MolFromSmiles(rule['substructure'])
            replacement = Chem.MolFromSmiles(rule['replacement'])
            result_smiles = replace_substructure(result_smiles, substructure, replacement)        
        replaced_molecules.append(Chem.MolToSmiles(result_smiles))
    return replaced_molecules

def perform_batch_replacements2(molecules, rep_list, replacement_rule):
    replaced_molecules = []
    for smiles, replacement in zip(molecules, rep_list):            
        molecule = Chem.MolFromSmiles(smiles)
        replacement = Chem.MolFromSmiles(replacement)
        for rule in replacement_rule:
            substructure = Chem.MolFromSmiles(rule)
            result_smiles = replace_substructure(molecule, substructure, replacement)
            for rule in replacement_rule:
                substructure = Chem.MolFromSmiles(rule)
                result_smiles = replace_substructure(result_smiles, substructure, replacement)   
        replaced_molecules.append(Chem.MolToSmiles(result_smiles))
    return replaced_molecules



def polymer_reaction_alcohol(polyol, Diisocyanate, alcohol):
    replacement_rules = [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}, {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}]
    result_molecules = perform_batch_replacements1(Diisocyanate, replacement_rules)
    pre = perform_batch_replacements2(polyol, result_molecules, ['C[Re]'])
    pol_alcohol = perform_batch_replacements2(alcohol, pre, ['C[Re]'])
    return pol_alcohol

def polymer_reaction_amine(polyol, Diisocyanate, amine):
    result_molecules = perform_batch_replacements1(Diisocyanate, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}])
    pre = perform_batch_replacements2(polyol, result_molecules, ['C[Re]'])
    result_molecules1 = perform_batch_replacements1(pre, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'}, {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'}])
    pol_amine = perform_batch_replacements2(amine, result_molecules1, ['C[Rn]'])
    return pol_amine

def polymer_reaction_alcohol_amine(polyol, Diisocyanate, alcohol_amine):
    result_molecules = perform_batch_replacements1(Diisocyanate, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}])
    pre = perform_batch_replacements2(polyol, result_molecules, ['C[Re]'])
    
    result_molecules1 = perform_batch_replacements1(pre, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'}, {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'}])
    pre2 = perform_batch_replacements2(polyol, result_molecules, ['C[Re]'])
    result_molecules2 = perform_batch_replacements1(pre2, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}, {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}])
    
    replaced_molecules = [] 
    for smiles, rep_N in zip(alcohol_amine, result_molecules1):            
        molecule = Chem.MolFromSmiles(smiles)
        result_smiles = replace_substructure(molecule, Chem.MolFromSmiles('C[N]'), Chem.MolFromSmiles(rep_N))                  
        replaced_molecules.append(Chem.MolToSmiles(result_smiles))
    dpol = replaced_molecules
    
    polymer_aa = []
    for smiles, rep_O in zip(dpol, result_molecules2):
        molecule = Chem.MolFromSmiles(smiles)
        result_smiles = replace_substructure(molecule, Chem.MolFromSmiles('C[O]'), Chem.MolFromSmiles(rep_O))
        polymer_aa.append(Chem.MolToSmiles(result_smiles))
        
    return polymer_aa

def polymer_reaction_mercaptan(polyol, Diisocyanate, mercaptan):
    result_molecules = perform_batch_replacements1(Diisocyanate, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}])
    pre = perform_batch_replacements2(polyol, result_molecules, ['C[Re]'])
    result_molecules1 = perform_batch_replacements1(pre, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)SC'}])
    pol_mercaptan1 = perform_batch_replacements2(mercaptan, result_molecules1, ['C[Rb]'])
    pol_mercaptan = perform_batch_replacements1(pol_mercaptan1, [{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}, {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}])
    return pol_mercaptan

def reaction_poly2(file):
    df = pd.read_csv(file)
    X = df.iloc[:, 0]
    polyol = list(X)
    y = df.iloc[:, 1]
    Diisocyanate = list(y)
    
    df = df.drop(['extender1'], axis=1)
    index = list(df.index)
    row_index1, row_index2, row_index3, row_index4 = [], [], [], []
    
    for a in index:
        k = df.iloc[a, 2]
        if '[Re]' in k: row_index1.append(a)
        elif '[Rn]' in k: row_index2.append(a)
        elif '[N]' in k: row_index3.append(a)
        elif '[Rb]' in k: row_index4.append(a)

    df1, df2, df3, df4 = df.iloc[row_index1].copy(), df.iloc[row_index2].copy(), df.iloc[row_index3].copy(), df.iloc[row_index4].copy()
    
    if not df1.empty:
        df1['pol'] = polymer_reaction_alcohol(list(df1.iloc[:, 0]), list(df1.iloc[:, 1]), list(df1['extender2']))
    if not df2.empty:
        df2['pol'] = polymer_reaction_amine(list(df2.iloc[:, 0]), list(df2.iloc[:, 1]), list(df2['extender2']))
    if not df3.empty:
        df3["pol"] = polymer_reaction_alcohol_amine(list(df3.iloc[:, 0]), list(df3.iloc[:, 1]), list(df3['extender2']))
    if not df4.empty:
        df4["pol"] = polymer_reaction_mercaptan(list(df4.iloc[:, 0]), list(df4.iloc[:, 1]), list(df4['extender2']))
        
    result = pd.concat([df1, df2, df3, df4])
    result.sort_index(inplace=True)
    return result 


def reaction_bools(file):
    df = pd.read_csv(file)
    polyol = list(df['polyol'])
    Diisocyanate = list(df['Diisocyanate'])
    c1_list = list(df['extender1'])
    

    part_A = polymer_reaction_alcohol(polyol, Diisocyanate, polyol)

    unit2 = reaction_poly2(file)
    part_B = list(unit2['pol'])


    after_trans_A = [s[:1] + "O" + s[2:] for s in part_A]
    after_trans_B = [s[:1] + "O" + s[2:] for s in part_B]

    polymer_unit = []
    for c1_smiles, A_smiles, B_smiles in zip(c1_list, after_trans_A, after_trans_B):
        c1_mol = Chem.MolFromSmiles(c1_smiles)
        A_mol = Chem.MolFromSmiles(A_smiles)
        B_mol = Chem.MolFromSmiles(B_smiles)
        

        tags = []
        for t in ['[Re]', '[Rn]', '[N]', '[O]', '[Rb]']:
            if t in c1_smiles:
                count = c1_smiles.count(t)
                for _ in range(count):
                    tags.append('C' + t)
                    
        if len(tags) >= 2:
            tag1, tag2 = tags[0], tags[1]

            step1_mol = replace_substructure(c1_mol, Chem.MolFromSmiles(tag1), A_mol)
            final_mol = replace_substructure(step1_mol, Chem.MolFromSmiles(tag2), B_mol)
            polymer_unit.append(Chem.MolToSmiles(final_mol))
        else:
            polymer_unit.append(c1_smiles)
            
    df["polymer-unit"] = polymer_unit
    df.to_csv(file, index=False)

if __name__ =='__main__':

    pass