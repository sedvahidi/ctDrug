"""
Implementation of a SMILES dataset.
"""

import torch
import torch.utils.data as tud


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, smiles_list, target_list, vocabulary):
        self._vocabulary = vocabulary
        self._smiles_list = list(smiles_list)
        self._target_list = None
        if target_list is not None:
            self._target_list = list(target_list)
            
    def __getitem__(self, i):
        smi = self._smiles_list[i]
        encoded = self._vocabulary.encode(smi, add_bos=True, add_eos=True)
        if self._target_list is not None:
            target = self._target_list[i]
            return [torch.tensor(encoded, dtype=torch.long), target] # pylint: disable=E1102
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self._smiles_list)
        
    @staticmethod
    def collate_fn(encoded_seqs):
        """Converts a list of encoded sequences into a padded tensor"""
        targets = None
        if len(encoded_seqs[0])==2:
            encoded_seqs, targets = list(zip(*encoded_seqs))
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        if targets is not None:
            targets = [tmp for tmp in targets]
            return (collated_arr, targets)
        return collated_arr
