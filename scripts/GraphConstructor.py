from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import numpy as np
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_implicit_valence_one_hot, atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    atom_chiral_tag_one_hot, one_hot_encoding, bond_is_conjugated, atom_formal_charge, atom_num_radical_electrons, bond_is_in_ring, bond_stereo_one_hot
import pickle
import copy
import sys
import os
from dgl.data.utils import save_graphs, load_graphs
import pandas as pd
from torch.utils.data import Dataset
from dgl.data.chem import mol_to_bigraph
from dgl.data.chem import BaseBondFeaturizer
from functools import partial
import warnings
from dgl.data.utils import save_graphs, load_graphs
import multiprocessing as mp
warnings.filterwarnings('ignore')
from torchani import SpeciesConverter, AEVComputer

converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'])


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Si'], encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                                                                               Chem.rdchem.BondStereo.STEREOANY,
                                                                                                               Chem.rdchem.BondStereo.STEREOZ,
                                                                                                               Chem.rdchem.BondStereo.STEREOE], encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles)*0.01, np.sum(Angles)*0.01, np.mean(Angles)*0.01, np.max(Areas), np.sum(Areas), np.mean(Areas),
                np.max(Distances)*0.1, np.sum(Distances)*0.1, np.mean(Distances)*0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()


def graph_from_mol(m, add_self_loop=False, add_3D=False):
    """
    :param m: molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return: 
    complex: graphs contain m1
    """
    # small molecule
    # new_order = rdmolfiles.CanonicalRankAtoms(m)
    # mol = rdmolops.RenumberAtoms(m, new_order)
    mol = m
    # construct graph
    g = dgl.DGLGraph()  # small molecule

    # add nodes
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = AtomFeaturizer(mol)['h']
    # 'charge'
    charges = [float(mol.GetAtomWithIdx(i).GetProp('molFileAlias')) for i in range(num_atoms)]
    g.ndata['charge'] = torch.tensor(charges, dtype=torch.float).unsqueeze(dim=1)

    # assign edge features
    # 'e', edge features
    efeats = BondFeaturizer(mol)['e']  # 重复的边存在！
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])

    # 'd', distance
    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)

    #'e', total features for edges
    # g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)

    if add_3D:
        g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
        g.ndata['pos'] = mol.GetConformers()[0].GetPositions()

        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    return g


def graph_from_mol_for_prediction(m):
    """
    :param m: molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return:
    complex: graphs contain m1
    """
    # small molecule
    # new_order = rdmolfiles.CanonicalRankAtoms(m)
    # mol = rdmolops.RenumberAtoms(m, new_order)
    mol = m
    # construct graph
    g = dgl.DGLGraph()  # small molecule

    # add nodes
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)

    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = AtomFeaturizer(mol)['h']
    # 'charge'
    charges = [float(0) for i in range(num_atoms)]
    g.ndata['charge'] = torch.tensor(charges, dtype=torch.float).unsqueeze(dim=1)

    # assign edge features
    # 'e', edge features
    efeats = BondFeaturizer(mol)['e']  # 重复的边存在！
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])

    # 'd', distance
    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)

    #'e', total features for edges
    # g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)

    g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
    g.ndata['pos'] = mol.GetConformers()[0].GetPositions()

    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    return g


# acsf descriptor = 65
def graph_from_mol_new(data_dir, key, cache_path, path_marker):
    # small molecule
    # new_order = rdmolfiles.CanonicalRankAtoms(m)
    # mol = rdmolops.RenumberAtoms(m, new_order)
    add_self_loop = False
    mol = Chem.MolFromMolFile(data_dir, removeHs=False)
    # construct graph
    g = dgl.DGLGraph()  # small molecule

    # add nodes
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = AtomFeaturizer(mol)['h']
    # 'charge'
    charges = [float(mol.GetAtomWithIdx(i).GetProp('molFileAlias')) for i in range(num_atoms)]
    g.ndata['charge'] = torch.tensor(charges, dtype=torch.float).unsqueeze(dim=1)

    # assign edge features
    # 'e', edge features
    efeats = BondFeaturizer(mol)['e']  # 重复的边存在！
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])

    # 'd', distance
    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)

    #'e', total features for edges
    # g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)

    g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
    g.ndata['pos'] = mol.GetConformers()[0].GetPositions()

    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)

    # acsf 计算
    AtomicNums = []
    for i in range(num_atoms):
        AtomicNums.append(mol.GetAtomWithIdx(i).GetAtomicNum())
    Corrds = mol.GetConformer().GetPositions()
    AtomicNums = torch.tensor(AtomicNums, dtype=torch.long)
    Corrds = torch.tensor(Corrds, dtype=torch.float64)
    AtomicNums = torch.unsqueeze(AtomicNums, dim=0)
    Corrds = torch.unsqueeze(Corrds, dim=0)
    res = converter((AtomicNums, Corrds))
    pbsf_computer = AEVComputer(Rcr=6.0, Rca=6.0, EtaR=torch.tensor([4.00]), ShfR=torch.tensor([3.17]),
                                EtaA=torch.tensor([3.5]), Zeta=torch.tensor([8.00]),
                                ShfA=torch.tensor([0]), ShfZ=torch.tensor([3.14]), num_species=10)
    outputs = pbsf_computer((res.species, res.coordinates))
    if torch.any(torch.isnan(outputs.aevs[0].float())):
        print(mol)
        status = False
    ligand_atoms_aves = outputs.aevs[0].float()
    # acsf features
    g.ndata['acsf'] = ligand_atoms_aves

    save_graphs(cache_path + path_marker + key, [g])


# # test
# m = Chem.MolFromMolFile('F:\\01ReData\\BigData\\ChargeData\\ChargeData\\e4\\all\\0.sdf', removeHs=False)
# g = graph_from_mol_new(m)


class GraphDataset(object):
    def __init__(self, data_dirs, cache_file_path, add_3D):
        self.data_dirs = data_dirs
        self.retained_dirs = []
        self.cache_file_path = cache_file_path
        self.add_3D = add_3D
        self._pre_process()

    def _pre_process(self):
        # for i, data_dir in enumerate(self.data_dirs):
        #     m = Chem.MolFromMolFile(data_dir, removeHs=False)
        #     atom_num = m.GetNumAtoms()
        #     if (atom_num >= 0) and (atom_num <= 65):
        #         self.retained_dirs.append(data_dir)
        if os.path.exists(self.cache_file_path):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
        else:
            print('Generate complex graph...')
            self.graphs = []
            for i, data_dir in enumerate(self.data_dirs):
                m = Chem.MolFromMolFile(data_dir, removeHs=False)
                atom_num = m.GetNumAtoms()
                if (atom_num >= 0) and (atom_num <= 65):
                    print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                    g = graph_from_mol(m, add_3D=self.add_3D)
                    self.graphs.append(g)
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)

    def __getitem__(self, indx):
        return self.graphs[indx]

    def __len__(self):
        # return len(self.data_dirs)
        return len(self.graphs)


class GraphDatasetNew(object):
    """
    created in 20210706
    """
    def __init__(self, data_dirs, data_keys, cache_bin_file, tmp_cache_path, path_marker='/', num_process=8):
        self.data_dirs = data_dirs
        self.data_keys = data_keys
        self.cache_bin_file = cache_bin_file
        self.num_process = num_process
        self.tmp_cache_path = tmp_cache_path
        self.path_marker = path_marker
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_bin_file):
            print('Loading previously saved dgl graphs...')
            self.graphs = load_graphs(self.cache_bin_file)[0]

        else:
            print('Generate complex graph...')
            if not os.path.exists(self.tmp_cache_path):
                cmdline = 'mkdir -p %s' % self.tmp_cache_path
                os.system(cmdline)

            pool = mp.Pool(self.num_process)
            self.graphs = pool.starmap(partial(graph_from_mol_new, cache_path=self.tmp_cache_path, path_marker=self.path_marker),
                                       zip(self.data_dirs, self.data_keys))
            pool.close()
            pool.join()
            self.graphs = []
            # load the saved individual graphs
            for key in self.data_keys:
                self.graphs.append(load_graphs(self.tmp_cache_path + self.path_marker + key)[0][0])
            save_graphs(self.cache_bin_file, self.graphs)
            cmdline = 'rm -rf %s' % self.tmp_cache_path
            os.system(cmdline)

    def __getitem__(self, indx):
        return self.graphs[indx], self.data_keys[indx]

    def __len__(self):
        return len(self.graphs)


def collate_fn(data_batch):
    graphs = data_batch
    bg = dgl.batch(graphs)
    return bg


def collate_fn_new(data_batch):
    graphs, keys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    return bg, keys
