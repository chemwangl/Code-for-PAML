#!/usr/bin/env python
# coding: utf-8
# Author: Qxk
# Time: 


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
        result_smiles = smiles
        result_smiles = Chem.MolFromSmiles(result_smiles)
        
        for rule in replacement_rules:
            substructure = Chem.MolFromSmiles(rule['substructure'])
            replacement = Chem.MolFromSmiles(rule['replacement'])
            result_smiles = replace_substructure(result_smiles, substructure, replacement)        
        replaced_molecules.append(Chem.MolToSmiles(result_smiles))

    return replaced_molecules

def perform_batch_replacements2(molecules, rep_list,replacement_rule):
    replaced_molecules = []
    
    for smiles,replacement in zip(molecules,rep_list):            
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

def perform_batch_replacements3(molecules, rep_list,replacement_rule):
    replaced_molecules = [] 
    
    for smiles,replacement in zip(molecules,rep_list):            
            molecule = Chem.MolFromSmiles(smiles)
            replacement = Chem.MolFromSmiles(replacement)
            
            for rule in replacement_rule:
                substructure = Chem.MolFromSmiles(rule)
                result_smiles = replace_substructure(molecule, substructure, replacement)                  
            replaced_molecules.append(Chem.MolToSmiles(result_smiles))
            
    return replaced_molecules

# return  prepolymer list   input list
def  prepolymer_reaction(polyol,Diisocyanate):  
    molecule_list = Diisocyanate
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
        ,{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
    ]
    add_1 = perform_batch_replacements1(molecule_list, replacement_rules)

    molecule_list = polyol
    replacement_list=add_1
    
    replacement_rule=['C[Re]']
    prepolymer = perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    return prepolymer

def polymer_reaction_alcohol(polyol,Diisocyanate,alcohol) :
    
    molecule_list = Diisocyanate
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
        ,{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
    ]
    result_molecules = perform_batch_replacements1(molecule_list, replacement_rules)
    molecule_list =polyol
    replacement_list=result_molecules 
    replacement_rule=['C[Re]']
    pre=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    molecule_list =alcohol
    replacement_list=pre
    replacement_rule= ['C[Re]']
    pol_alcohol=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    return pol_alcohol

def polymer_reaction_amine(polyol,Diisocyanate,amine):
    
    molecule_list = Diisocyanate
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'},    
    ]
    result_molecules = perform_batch_replacements1(molecule_list, replacement_rules)
    
    molecule_list = polyol
    replacement_list=result_molecules
    replacement_rule=['C[Re]']
    pre=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    molecule_list = pre
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'},
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'}
    ]
    result_molecules1 = perform_batch_replacements1(molecule_list, replacement_rules)
    
    molecule_list =amine
    replacement_list=result_molecules1
    replacement_rule= ['C[Rn]']
    pol_amine=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    return  pol_amine

def polymer_reaction_alcohol_amine(polyol,Diisocyanate,alcohol_amine):
    
    molecule_list = Diisocyanate
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
    ]
    result_molecules = perform_batch_replacements1(molecule_list, replacement_rules)
    
    add=result_molecules
    molecule_list = polyol
    replacement_list=add
    replacement_rule=['C[Re]']
    pre=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    molecule_list = pre
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'},
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)NC'}
    ]
    result_molecules1 = perform_batch_replacements1(molecule_list, replacement_rules)

    molecule_list = polyol
    replacement_list=add
    replacement_rule=['C[Re]']
    pre2=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    molecule_list = pre2
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'},
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
    ]
    result_molecules2 = perform_batch_replacements1(molecule_list, replacement_rules)
    
    molecule_list =alcohol_amine
    replacement_list=result_molecules1
    replacement_rule= ['C[N]']
    dpol=perform_batch_replacements3(molecule_list,replacement_list,replacement_rule)
    
    molecule_list =dpol
    replacement_list=result_molecules2
    replacement_rule= ['C[O]']
    polymer_reaction_alcohol_amine=perform_batch_replacements3(molecule_list,replacement_list,replacement_rule)
    
    return polymer_reaction_alcohol_amine

def polymer_reaction_mercaptan(polyol,Diisocyanate,mercaptan):
    
    molecule_list = Diisocyanate
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'},    
    ]
    result_molecules = perform_batch_replacements1(molecule_list, replacement_rules)

    molecule_list = polyol
    replacement_list=result_molecules
    replacement_rule=['C[Re]']
    pre=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    
    molecule_list = pre
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)SC'},
        ]
    result_molecules1 = perform_batch_replacements1(molecule_list, replacement_rules)

    molecule_list =mercaptan
    replacement_list=result_molecules1
    replacement_rule= ['C[Rb]']
    pol_mercaptan1=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)
    molecule_list = pol_mercaptan1
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
        ,{'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
        
    ]
    pol_mercaptan= perform_batch_replacements1(molecule_list, replacement_rules)

    return  pol_mercaptan

#prepoly react with extender1  return  dataframe
def  reaction_poly1(file):

    import pandas as pd
    df= pd.read_csv(file)
    X=df.iloc[:,0]
    polyol=list(X)
    y=df.iloc[:,1]
    Diisocyanate=list(y)
    df=df.drop(['extender2'],axis=1)
    index=list(df.index)
    row_index4=[]
    row_index1=[]
    row_index2=[]
    row_index3=[]

    for a in index :
        k=df.iloc[a, 2]
        if  '[Re]' in  k:
            row_index1.append(a)
        elif  '[Rn]' in  k : 
            row_index2.append(a)
        elif  '[N]' in k :
            row_index3.append(a)
        elif '[Rb]' in k : 
            row_index4.append(a)

    selected_rows1 = df.iloc[row_index1]
    df1=pd.DataFrame(selected_rows1)
    selected_rows2 = df.iloc[row_index2]
    df2=pd.DataFrame(selected_rows2)
    selected_rows3 = df.iloc[row_index3]
    df3=pd.DataFrame(selected_rows3)
    selected_rows4 = df.iloc[row_index4]
    df4=pd.DataFrame(selected_rows4)

    if df1.empty ==False :
        X1=df1.iloc[:,1]
        Diisocyanate1=list(X1)
        y1=df1.iloc[:,0]
        polyol1=list(y1)
        z1=df1['extender1']
        alcohol1=list(z1)
        df1['pol']=polymer_reaction_alcohol(polyol1,Diisocyanate1,alcohol1)

    if df2.empty ==False:
        X2=df2.iloc[:,1]
        Diisocyanate2=list(X2)
        y2=df2.iloc[:,0]
        polyol2=list(y2)
        z2=df2['extender1']
        amine=list(z2)
        df2['pol']=polymer_reaction_amine(polyol2,Diisocyanate2,amine)

    if df3.empty ==False:
        X3=df3.iloc[:,1]
        Diisocyanate3=list(X3)
        y3=df3.iloc[:,0]
        polyol3=list(y3)
        z3=df3['extender1']
        alcohol_amine=list(z3)
        df3["pol"] = polymer_reaction_alcohol_amine(polyol3,Diisocyanate3,alcohol_amine)

    if df4.empty == False:
        X4=df4.iloc[:,1]
        Diisocyanate4=list(X4)
        y4=df4.iloc[:,0]
        polyol4=list(y4)
        z4=df4['extender1']
        mercaptan=list(z4)    
        df4["pol"] = polymer_reaction_mercaptan(polyol4,Diisocyanate4,mercaptan)
        
    result = pd.concat([df1, df2, df3,df4])
    result.sort_index(inplace=True)
    
    return result

##prepoly react with extender2    return  dataframe
def  reaction_poly2(file):
    
    import pandas as pd
    df= pd.read_csv(file)
    X=df.iloc[:,0]
    polyol=list(X)
    y=df.iloc[:,1]
    Diisocyanate=list(y)
    df=df.drop(['extender1'],axis=1)
    index=list(df.index)
    row_index4=[]
    row_index1=[]
    row_index2=[]
    row_index3=[]
    for a in index :
        k=df.iloc[a, 2]
        if  '[Re]' in  k:
            row_index1.append(a)
        elif  '[Rn]' in  k : 
            row_index2.append(a)
        elif  '[N]' in k :
            row_index3.append(a)
        elif '[Rb]' in k : 
            row_index4.append(a)

    selected_rows1 = df.iloc[row_index1]
    df1=pd.DataFrame(selected_rows1)
    selected_rows2 = df.iloc[row_index2]
    df2=pd.DataFrame(selected_rows2)
    selected_rows3 = df.iloc[row_index3]
    df3=pd.DataFrame(selected_rows3)
    selected_rows4 = df.iloc[row_index4]
    df4=pd.DataFrame(selected_rows4)
    
    if df1.empty ==False :
        X1=df1.iloc[:,1]
        Diisocyanate1=list(X1)
        y1=df1.iloc[:,0]
        polyol1=list(y1)
        z1=df1['extender2']
        alcohol1=list(z1)
        df1['pol']=polymer_reaction_alcohol(polyol1,Diisocyanate1,alcohol1)
        
    if df2.empty ==False:
        X2=df2.iloc[:,1]
        Diisocyanate2=list(X2)
        y2=df2.iloc[:,0]
        polyol2=list(y2)
        z2=df2['extender2']
        amine=list(z2)
        df2['pol']=polymer_reaction_amine(polyol2,Diisocyanate2,amine)
        
    if df3.empty ==False:
        X3=df3.iloc[:,1]
        Diisocyanate3=list(X3)
        y3=df3.iloc[:,0]
        polyol3=list(y3)
        z3=df3['extender2']
        alcohol_amine=list(z3)
        df3["pol"] = polymer_reaction_alcohol_amine(polyol3,Diisocyanate3,alcohol_amine)
        
    if df4.empty ==False:
        X4=df4.iloc[:,1]
        Diisocyanate4=list(X4)
        y4=df4.iloc[:,0]
        polyol4=list(y4)
        z4=df4['extender2']
        mercaptan=list(z4)    
        df4["pol"] = polymer_reaction_mercaptan(polyol4,Diisocyanate4,mercaptan)
        
    result = pd.concat([df1, df2, df3,df4])
    result.sort_index(inplace=True)
    
    return result 

#pu_unit reaction
def  reaction_bools(file):
    
    unit1=reaction_poly1(file)
    unit2=reaction_poly2(file)

    trans1=list(unit1['pol'])
    after_trans1= []
    for input_string in trans1:
        new_letter = "O"
        modified_string = input_string[:1]+ new_letter +input_string[2:]
        after_trans1.append(modified_string)

    trans2=list(unit2['pol'])
    after_trans2= []
    for input_string in trans2:
        new_letter = "O"
        modified_string = input_string[:1]+ new_letter +input_string[2:]
        after_trans2.append(modified_string)
        
    import pandas as pd
    df= pd.read_csv(file)
    X=df['polyol']
    polyol=list(X)

    molecule_list = polyol
    replacement_list=after_trans1
    replacement_rule=['C[Re]']
    add=perform_batch_replacements3(molecule_list,replacement_list,replacement_rule)
    
    molecule_list = add
    replacement_list=after_trans2
    replacement_rule=['C[Re]']
    polymer_unit=perform_batch_replacements3(molecule_list,replacement_list,replacement_rule)
    
    df["polymer-unit"] =  polymer_unit
    df.to_csv(file, index=False)

#single reaction
def  single_reaction_poly1(polyol,Diisocyanate,extender1,extender2):

    bool1=[polyol,Diisocyanate,extender1,extender2]
    df=pd.DataFrame(bool1)
    df=df.T
    X=df.iloc[:,0]
    polyol=list(X)
    y=df.iloc[:,1]
    Diisocyanate=list(y)
    df=df.drop([3],axis=1)

    index=list(df.index)
    row_index4=[]
    row_index1=[]
    row_index2=[]
    row_index3=[]
    for a in index :
        k=df.iloc[a, 2]
        if  '[Re]' in  k:
            row_index1.append(a)
        elif  '[Rn]' in  k : 
            row_index2.append(a)
        elif  '[N]' in k :
            row_index3.append(a)
        elif '[Rb]' in k : 
            row_index4.append(a)

    selected_rows1 = df.iloc[row_index1]
    df1=pd.DataFrame(selected_rows1)
    selected_rows2 = df.iloc[row_index2]
    df2=pd.DataFrame(selected_rows2)
    selected_rows3 = df.iloc[row_index3]
    df3=pd.DataFrame(selected_rows3)
    selected_rows4 = df.iloc[row_index4]
    df4=pd.DataFrame(selected_rows4)

    if df1.empty ==False :
        X1=df1.iloc[:,1]
        Diisocyanate1=list(X1)
        y1=df1.iloc[:,0]
        polyol1=list(y1)
        z1=df1.iloc[:,2]
        alcohol1=list(z1)
        df1['pol']=polymer_reaction_alcohol(polyol1,Diisocyanate1,alcohol1)

    if df2.empty ==False:
        X2=df2.iloc[:,1]
        Diisocyanate2=list(X2)
        y2=df2.iloc[:,0]
        polyol2=list(y2)
        z2=df2.iloc[:,2]
        amine=list(z2)
        df2['pol']=polymer_reaction_amine(polyol2,Diisocyanate2,amine)

    if df3.empty ==False:
        X3=df3.iloc[:,1]
        Diisocyanate3=list(X3)
        y3=df3.iloc[:,0]
        polyol3=list(y3)
        z3=df3.iloc[:,2]
        alcohol_amine=list(z3)
        df3["pol"] = polymer_reaction_alcohol_amine(polyol3,Diisocyanate3,alcohol_amine)

    if df4.empty == False:
        X4=df4.iloc[:,1]
        Diisocyanate4=list(X4)
        y4=df4.iloc[:,0]
        polyol4=list(y4)
        z4=df4.iloc[:,2]
        mercaptan=list(z4)    
        df4["pol"] = polymer_reaction_mercaptan(polyol4,Diisocyanate4,mercaptan)
        
    result = pd.concat([df1, df2, df3,df4])
    result.sort_index(inplace=True)
    
    return result

##prepoly react with extender2    return  dataframe
def  single_reaction_poly2(polyol,Diisocyanate,extender1,extender2):

    bool1=[polyol,Diisocyanate,extender1,extender2]
    df=pd.DataFrame(bool1)
    df=df.T
    X=df.iloc[:,0]
    polyol=list(X)
    y=df.iloc[:,1]
    Diisocyanate=list(y)
    df=df.drop([2],axis=1)

    index=list(df.index)
    row_index4=[]
    row_index1=[]
    row_index2=[]
    row_index3=[]
    for a in index :
        k=df.iloc[a, 2]
        if  '[Re]' in  k:
            row_index1.append(a)
        elif  '[Rn]' in  k : 
            row_index2.append(a)
        elif  '[N]' in k :
            row_index3.append(a)
        elif '[Rb]' in k : 
            row_index4.append(a)

    selected_rows1 = df.iloc[row_index1]
    df1=pd.DataFrame(selected_rows1)
    selected_rows2 = df.iloc[row_index2]
    df2=pd.DataFrame(selected_rows2)
    selected_rows3 = df.iloc[row_index3]
    df3=pd.DataFrame(selected_rows3)
    selected_rows4 = df.iloc[row_index4]
    df4=pd.DataFrame(selected_rows4)

    if df1.empty ==False :
        X1=df1.iloc[:,1]
        Diisocyanate1=list(X1)
        y1=df1.iloc[:,0]
        polyol1=list(y1)
        z1=df1.iloc[:,2]
        alcohol1=list(z1)
        df1['pol']=polymer_reaction_alcohol(polyol1,Diisocyanate1,alcohol1)

    if df2.empty ==False:
        X2=df2.iloc[:,1]
        Diisocyanate2=list(X2)
        y2=df2.iloc[:,0]
        polyol2=list(y2)
        z2=df2.iloc[:,2]
        amine=list(z2)
        df2['pol']=polymer_reaction_amine(polyol2,Diisocyanate2,amine)

    if df3.empty ==False:
        X3=df3.iloc[:,1]
        Diisocyanate3=list(X3)
        y3=df3.iloc[:,0]
        polyol3=list(y3)
        z3=df3.iloc[:,2]
        alcohol_amine=list(z3)
        df3["pol"] = polymer_reaction_alcohol_amine(polyol3,Diisocyanate3,alcohol_amine)

    if df4.empty == False:
        X4=df4.iloc[:,1]
        Diisocyanate4=list(X4)
        y4=df4.iloc[:,0]
        polyol4=list(y4)
        z4=df4.iloc[:,2]
        mercaptan=list(z4)    
        df4["pol"] = polymer_reaction_mercaptan(polyol4,Diisocyanate4,mercaptan)
        
    result = pd.concat([df1, df2, df3,df4])
    result.sort_index(inplace=True)
    
    return result

#pu_unit reaction
def  single_reaction_bool(polyol,Diisocyanate,extender1,extender2):
    
    unit1=single_reaction_poly1(polyol,Diisocyanate,extender1,extender2)
    unit2=single_reaction_poly2(polyol,Diisocyanate,extender1,extender2)

    trans1=list(unit1['pol'])
    after_trans1= []
    for input_string in trans1:
        new_letter = "O"
        modified_string = input_string[:1]+ new_letter +input_string[2:]
        after_trans1.append(modified_string)
        
    trans2=list(unit2['pol'])
    after_trans2= []
    for input_string in trans2:
        new_letter = "O"
        modified_string = input_string[:1]+ new_letter +input_string[2:]
        after_trans2.append(modified_string)

    bool1=[polyol,Diisocyanate,extender1,extender2]
    df=pd.DataFrame(bool1)
    df=df.T
    X=df.iloc[:,0]
    polyol=list(X)
    
    molecule_list = polyol
    replacement_list=after_trans1
    replacement_rule=['C[Re]']
    add=perform_batch_replacements3(molecule_list,replacement_list,replacement_rule)
    
    molecule_list = add
    replacement_list=after_trans2
    replacement_rule=['C[Re]']
    polymer_unit=perform_batch_replacements3(molecule_list,replacement_list,replacement_rule)
    
    df["polymer-unit"] =  polymer_unit
    polymer_unit_out= list(df["polymer-unit"])
    return polymer_unit_out

def  prepolymer_reaction_bools(file):

    df=pd.read_csv(file)
    X=df.iloc[:,0]
    polyol=list(X)
    y=df.iloc[:,1]
    Diisocyanate=list(y)

    molecule_list = Diisocyanate
    replacement_rules = [
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'},
        {'substructure': 'CN=C=O', 'replacement': 'CNC(=O)OC'}
    ]
    add_1 = perform_batch_replacements1(molecule_list, replacement_rules)

    molecule_list = polyol
    replacement_list=add_1
    replacement_rule=['C[Re]']
    prepolymer=perform_batch_replacements2(molecule_list,replacement_list,replacement_rule)

    df["prepolymer"] =  prepolymer
    df.to_csv(file, index=False)



if __name__ =='__main__':
    pass
    

