from rdkit import Chem
sdfs = Chem.SDMolSupplier(r'/home/jike/dejunjiang/charge_prediction/outputs/hello_charge/3cl-min_new_e4.sdf')
for i, mol in enumerate(sdfs):
    # 修正后的原子电荷的总和, 整数
    print(float(mol.GetProp('charge')))
    for atom in mol.GetAtoms():
        # 修正后的原子电荷
        print(float(atom.GetProp('molFileAlias')))

# Chem.MolToMolFile(sdfs[0], r'G:\05Coding\ChargePredction\ligand_min_new\test.sdf')
