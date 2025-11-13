import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

from pubchemfp import GetPubChemFPs


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        # mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        # try:
        #     # Add conformer to molecule
        #     AllChem.EmbedMolecule(mol, randomSeed=42)
        #     AllChem.MMFFOptimizeMolecule(mol)
        # except:
        #     # If embedding fails, use ETKDG method as fallback
        #     try:
        #         AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        #         AllChem.MMFFOptimizeMolecule(mol)
        #     except:
        #         # If all methods fail, create a dummy conformer with zeros
        #         conformer = Chem.Conformer(mol.GetNumAtoms())
        #         for i in range(mol.GetNumAtoms()):
        #             conformer.SetAtomPosition(i, (0, 0, 0))
        #         mol.AddConformer(conformer)

        # Extract 3D coordinates - add error handling for Bad Conformer Id
        # try:
        #     conformer = mol.GetConformer()
        # except:
        #     # If we still can't get a conformer, create a dummy one
        #     if mol.GetNumAtoms() > 0:
        #         conformer = Chem.Conformer(mol.GetNumAtoms())
        #         for i in range(mol.GetNumAtoms()):
        #             conformer.SetAtomPosition(i, (0, 0, 0))
        #         mol.AddConformer(conformer)
        #         conformer = mol.GetConformer()
        #     else:
        #         # Handle case where molecule has no atoms
        #         pos = torch.zeros((0, 3), dtype=torch.float)
        #         # Return a minimal valid data structure
        #         x = torch.zeros((0, 2), dtype=torch.long)
        #         # z = torch.zeros((0, 1), dtype=torch.long)
        #         edge_index = torch.zeros((2, 0), dtype=torch.long)
        #         edge_attr = torch.zeros((0, 2), dtype=torch.long)
        #         data_i = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #         FP = torch.zeros((1, 2048), dtype=torch.float32)  # Assuming fingerprint size
        #         data_fp = Data(x=FP)
        #         return data_i, data_fp
        # pos = []
        # for i in range(conformer.GetNumAtoms()):
        #     position = conformer.GetAtomPosition(i)
        #     pos.append([position.x, position.y, position.z])
        # pos = torch.tensor(np.array(pos), dtype=torch.float)

        fp_list = []
        fp = []
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        fp_pubcfp = GetPubChemFPs(mol)
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp.extend(fp_pubcfp)
        fp_list.append(fp)
        FP = torch.tensor(np.array(fp_list), dtype=torch.float32)
        
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        # z = torch.cat([x1], dim=-1)
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # random mask a subgraph of the molecule
        num_mask_nodes = max([1, math.floor(0.25*N)])
        num_mask_edges = max([0, math.floor(0.25*M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]

        x_i = deepcopy(x)
        # z_i = deepcopy(z)
        # pos_i = deepcopy(pos)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
            # z_i[atom_idx] = torch.tensor([len(ATOM_LIST)])
            # pos_i[atom_idx,:] = torch.tensor([0, 0, 0])
        edge_index_i = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:,count] = edge_index[:,bond_idx]
                edge_attr_i[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        data_fp = Data(x=FP)
        return data_i, data_fp

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
