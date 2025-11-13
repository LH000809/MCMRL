import os
import csv
import math
import time
import random
import numpy as np
from pubchemfp import GetPubChemFPs

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
import os.path as osp
from typing import List, Dict, Any, Optional, Tuple


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path: str, target: str, task: str) -> Tuple[List[str], List[Any]]:
    smiles_data: List[str] = []
    labels: List[Any] = []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


def process_molecule(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Process a single molecule to generate 3D coordinates and features.
    This function is designed to be used with multiprocessing.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # mol = Chem.AddHs(mol)
    
    # # Generate 3D coordinates using more efficient ETKDGv3
    # try:
    #     # Add conformer to molecule using ETKDGv3 which is generally faster
    #     params = AllChem.ETKDGv3()
    #     params.randomSeed = 42
    #     AllChem.EmbedMolecule(mol, params)
    #     AllChem.MMFFOptimizeMolecule(mol)
    # except:
    #     # If embedding fails, try standard ETKDG as fallback
    #     try:
    #         AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    #         AllChem.MMFFOptimizeMolecule(mol)
    #     except:
    #         # If all methods fail, create a dummy conformer with zeros
    #         conformer = Chem.Conformer(mol.GetNumAtoms())
    #         for i in range(mol.GetNumAtoms()):
    #             conformer.SetAtomPosition(i, (0, 0, 0))
    #         mol.AddConformer(conformer)

    # # Extract 3D coordinates - add error handling for Bad Conformer Id
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
    #         return None
    
    # # Extract coordinates using most efficient method with NumPy
    # coords = np.zeros((conformer.GetNumAtoms(), 3))
    # for i in range(conformer.GetNumAtoms()):
    #     pos = conformer.GetAtomPosition(i)
    #     coords[i] = [pos.x, pos.y, pos.z]
    # pos = coords
    
    # Generate fingerprints more efficiently
    fp = []
    fp.extend(AllChem.GetMACCSKeysFingerprint(mol))
    fp.extend(AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1))
    fp.extend(GetPubChemFPs(mol))
    fp_list = [fp]
    
    # Generate atom features more efficiently using list comprehension
    atoms = mol.GetAtoms()
    type_idx = [ATOM_LIST.index(atom.GetAtomicNum()) for atom in atoms]
    chirality_idx = [CHIRALITY_LIST.index(atom.GetChiralTag()) for atom in atoms]
    atomic_number = [atom.GetAtomicNum() for atom in atoms]
    
    # Generate bond features
    row: List[int] = []
    col: List[int] = []
    edge_feat: List[List[int]] = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
    
    return {
        'smiles': smiles,
        # 'pos': pos,
        'fp': fp_list,
        'type_idx': type_idx,
        'chirality_idx': chirality_idx,
        'atomic_number': atomic_number,
        'row': row,
        'col': col,
        'edge_feat': edge_feat
    }


class MolTestDataset(Dataset):
    def __init__(self, data_path: str, target: str, task: str, precompute: bool = True, cache_path: Optional[str] = None):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        self.cache_path = cache_path
        
        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')
        
        # Precompute molecular features if requested
        if precompute:
            self.precomputed_data = self._precompute_features()
        else:
            self.precomputed_data = None

    def _precompute_features(self):
        """Precompute molecular features using multiprocessing"""
        # Check if cached data exists
        if self.cache_path and osp.exists(self.cache_path):
            print(f"Loading precomputed features from {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"Precomputing features for {len(self.smiles_data)} molecules using multiprocessing...")
        start_time = time.time()
        
        # Use multiprocessing to process molecules in parallel
        num_processes = min(cpu_count(), len(self.smiles_data))
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_molecule, self.smiles_data)
        
        # Convert results to a dictionary for fast lookup
        precomputed_data = {}
        for i, result in enumerate(results):
            if result is not None:
                precomputed_data[self.smiles_data[i]] = result
        
        # Save to cache if path provided
        if self.cache_path:
            print(f"Saving precomputed features to {self.cache_path}")
            with open(self.cache_path, 'wb') as f:
                pickle.dump(precomputed_data, f)
        
        end_time = time.time()
        print(f"Precomputation completed in {end_time - start_time:.2f} seconds")
        return precomputed_data

    def __getitem__(self, index):
        smiles = self.smiles_data[index]
        
        # Use precomputed data if available
        if self.precomputed_data and smiles in self.precomputed_data:
            # Retrieve precomputed data
            data_dict = self.precomputed_data[smiles]
            
            # Convert to tensors
            # pos = torch.tensor(data_dict['pos'], dtype=torch.float)
            FP = torch.tensor(np.array(data_dict['fp']), dtype=torch.float32)
            
            x1 = torch.tensor(data_dict['type_idx'], dtype=torch.long).view(-1,1)
            x2 = torch.tensor(data_dict['chirality_idx'], dtype=torch.long).view(-1,1)
            x = torch.cat([x1, x2], dim=-1)
            
            # z = torch.tensor(data_dict['atomic_number'], dtype=torch.long)
            
            edge_index = torch.tensor([data_dict['row'], data_dict['col']], dtype=torch.long)
            edge_attr = torch.tensor(np.array(data_dict['edge_feat']), dtype=torch.long)
            
            if self.task == 'classification':
                y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
            elif self.task == 'regression':
                y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
                
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            data_fp = Data(x=FP, y=y)
            return data, data_fp
        # else:
        #     # Fallback to original computation if precomputed data is not available
        #     mol = Chem.MolFromSmiles(smiles)
        #     # mol = Chem.AddHs(mol)

        #     # Generate 3D coordinates using more efficient ETKDGv3
        #     try:
        #         # Add conformer to molecule using ETKDGv3 which is generally faster
        #         params = AllChem.ETKDGv3()
        #         params.randomSeed = 42
        #         AllChem.EmbedMolecule(mol, params)
        #         AllChem.MMFFOptimizeMolecule(mol)
        #     except:
        #         # If embedding fails, try standard ETKDG as fallback
        #         try:
        #             AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        #             AllChem.MMFFOptimizeMolecule(mol)
        #         except:
        #             # If all methods fail, create a dummy conformer with zeros
        #             conformer = Chem.Conformer(mol.GetNumAtoms())
        #             for i in range(mol.GetNumAtoms()):
        #                 conformer.SetAtomPosition(i, (0, 0, 0))
        #             mol.AddConformer(conformer)

        #     # Extract 3D coordinates - add error handling for Bad Conformer Id
        #     try:
        #         conformer = mol.GetConformer()
        #     except:
        #         # If we still can't get a conformer, create a dummy one
        #         if mol.GetNumAtoms() > 0:
        #             conformer = Chem.Conformer(mol.GetNumAtoms())
        #             for i in range(mol.GetNumAtoms()):
        #                 conformer.SetAtomPosition(i, (0, 0, 0))
        #             mol.AddConformer(conformer)
        #             conformer = mol.GetConformer()
        #         else:
        #             # Handle case where molecule has no atoms
        #             pos = torch.zeros((0, 3), dtype=torch.float)
        #             # Return a minimal valid data structure
        #             x = torch.zeros((0, 2), dtype=torch.long)
        #             z = torch.zeros((0, 1), dtype=torch.long)
        #             edge_index = torch.zeros((2, 0), dtype=torch.long)
        #             edge_attr = torch.zeros((0, 2), dtype=torch.long)
        #             data_i = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, z=z)
        #             FP = torch.zeros((1, 2048), dtype=torch.float32)  # Assuming fingerprint size
        #             data_fp = Data(x=FP)
        #             return data_i, data_fp
        #     pos = []
        #     for i in range(conformer.GetNumAtoms()):
        #         position = conformer.GetAtomPosition(i)
        #         pos.append([position.x, position.y, position.z])
        #     pos = torch.tensor(np.array(pos), dtype=torch.float)

            # N = mol.GetNumAtoms()
            # M = mol.GetNumBonds()
            # # Generate fingerprints more efficiently
            # fp = []
            # fp.extend(AllChem.GetMACCSKeysFingerprint(mol))
            # fp.extend(AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1))
            # fp.extend(GetPubChemFPs(mol))
            # FP = torch.tensor(np.array([fp]), dtype=torch.float32)

            # type_idx = []
            # chirality_idx = []
            # atomic_number = []
            # for atom in mol.GetAtoms():
            #     type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            #     chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            #     atomic_number.append(atom.GetAtomicNum())

            # x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
            # x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
            # x = torch.cat([x1, x2], dim=-1)
            
            # # For SchNet, we need atomic numbers (z) and positions (pos)
            # z = torch.tensor(atomic_number, dtype=torch.long)

            # row, col, edge_feat = [], [], []
            # for bond in mol.GetBonds():
            #     start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            #     row += [start, end]
            #     col += [end, start]
            #     edge_feat.append([
            #         BOND_LIST.index(bond.GetBondType()),
            #         BONDDIR_LIST.index(bond.GetBondDir())
            #     ])
            #     edge_feat.append([
            #         BOND_LIST.index(bond.GetBondType()),
            #         BONDDIR_LIST.index(bond.GetBondDir())
            #     ])

            # edge_index = torch.tensor([row, col], dtype=torch.long)
            # edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
            # if self.task == 'classification':
            #     y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
            # elif self.task == 'regression':
            #     y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
            # data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            # data_fp = Data(x=FP, y=y)
            # return data, data_fp

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size: int, num_workers: int, valid_size: float, test_size: float, 
        data_path: str, target: str, task: str, splitting: str
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Generate cache path based on data path
        cache_dir = osp.dirname(self.data_path)
        cache_filename = f"precomputed_features_{osp.basename(self.data_path)}.pkl"
        cache_path = osp.join(cache_dir, cache_filename)
        
        train_dataset = MolTestDataset(
            data_path=self.data_path, 
            target=self.target, 
            task=self.task,
            precompute=True,
            cache_path=cache_path
        )
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset: MolTestDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
