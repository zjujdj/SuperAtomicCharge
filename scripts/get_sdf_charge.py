from rdkit import Chem
sdfs = Chem.SDMolSupplier('../outputs/hello_charge/3cl-min_new_e4.sdf')
for i, mol in enumerate(sdfs):
    # get the summation of predicted charge in a molecule
    print(float(mol.GetProp('charge')))
    for atom in mol.GetAtoms():
        # get the predicted charge of each atom
        print(float(atom.GetProp('molFileAlias')))
