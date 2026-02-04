# coding=utf-8

"""
Vocabulary helper class
"""

import re
import numpy as np
    
class tokens_struct():
    def __init__(self):
        self.tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                       '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                       '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']

        self.tokens_length = len(self.tokens)
        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    @property
    def bos(self):
        return self.tokens_vocab['<']

    @property
    def eos(self):
        return self.tokens_vocab['>']

    @property
    def pad(self):
        return self.tokens_vocab[' ']

    def get_default_tokens(self):
        """Default SMILES tokens"""
        return self.tokens

    def get_tokens_length(self):
        """Default SMILES tokens length"""
        return self.tokens_length

    def encode(self, char_list, add_bos=False, add_eos=False):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.tokens_vocab[char]
        if add_bos:
            smiles_matrix = np.insert(smiles_matrix, 0, self.bos)
        if add_eos:
            smiles_matrix = np.append(smiles_matrix, self.eos)
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.tokens_vocab['>']: break
            if i == self.tokens_vocab['<']: continue
            chars.append(self.reversed_tokens_vocab[i])
        smiles = "".join(chars)
        return smiles

    def pad_sequence(self, sentence, sen_len=140, pad_index=0):
        # 将每个sentence变成一样的长度
        if len(sentence) > sen_len:
            sentence_tensor = torch.FloatTensor(sentence[:sen_len])
        else:
            sentence_tensor = torch.ones(sen_len) * pad_index
            sentence_tensor[:len(sentence)] = torch.FloatTensor(sentence)
        assert len(sentence_tensor) == sen_len
        return sentence_tensor

import selfies as sf
import torch
class selfies_tokens_struct():
    def __init__(self):
        self.tokens = ['[nop]', '<', '>', '[#Branch1]', '[#Branch2]', '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[=Branch1]',
                       '[=Branch2]', '[=C-1]', '[=C]', '[=N+1]', '[=N-1]', '[=N]', '[=O+1]', '[=O]', '[=Ring1]',
                       '[=Ring2]', '[=S+1]', '[=SH1]', '[=S]', '[B-1]', '[B]', '[Br+1]', '[Br]', '[Branch1]',
                       '[Branch2]', '[C-1]', '[CH0]', '[CH1+1]', '[CH2+1]', '[CH2-1]', '[C]', '[Cl+1]', '[Cl]', '[F+1]',
                       '[F]', '[H]', '[I]', '[N+1]', '[N-1]', '[NH1]', '[N]', '[O+1]', '[O-1]', '[OH0]', '[O]', '[P]',
                       '[Ring1]', '[Ring2]', '[S+1]', '[SH1]', '[S]']

        self.tokens_length = len(self.tokens)
        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    @property
    def bos(self):
        return self.tokens_vocab['<']

    @property
    def eos(self):
        return self.tokens_vocab['>']

    @property
    def pad(self):
        return self.tokens_vocab['[nop]']

    def get_default_tokens(self):
        #Default SMILES tokens
        return self.tokens

    def get_tokens_length(self):
        #Default SMILES tokens length
        return self.tokens_length

    def encode(self, char_list, add_bos=False, add_eos=False):
        #Takes a list of characters (eg '[NH]') and encodes to array of indices
        label, _ = sf.selfies_to_encoding(
            selfies=char_list,
            vocab_stoi=self.tokens_vocab,
            enc_type="both"
        )
        smiles_matrix = np.array(label, dtype=np.float32)
        if add_bos:
            smiles_matrix = np.insert(smiles_matrix, 0, self.bos)
        if add_eos:
            smiles_matrix = np.append(smiles_matrix, self.eos)
        return smiles_matrix

    def seq2tensor(self, seqs):
        pad_to_len = max(sf.len_selfies(s) for s in seqs)
        tensor = torch.zeros((len(seqs), pad_to_len))
        for i, seq in enumerate(seqs):
            label, _ = sf.selfies_to_encoding(
                selfies=seq,
                vocab_stoi=self.tokens_vocab,
                pad_to_len=pad_to_len,
                enc_type="both"
            )
            tensor[i, :] = label
        return tensor

    def seq2list(self, seqs):
        results = []
        for i, seq in enumerate(seqs):
            label, _ = sf.selfies_to_encoding(
                selfies=seq,
                vocab_stoi=self.tokens_vocab,
                enc_type="both"
            )
            results.append(label)
        return results

    def decode(self, matrix):
        #Takes an array of indices and returns the corresponding SMILES
        chars = []
        for i in matrix:
            if i == self.tokens_vocab['>']: break
            if i == self.tokens_vocab['<']: continue
            chars.append(self.reversed_tokens_vocab[i])
        smiles = "".join(chars)
        return smiles

    def pad_sequence(self, sentence, sen_len=140, pad_index=0):\
        if len(sentence) > sen_len:
            sentence_tensor = torch.FloatTensor(sentence[:sen_len])
        else:
            sentence_tensor = torch.ones(sen_len) * pad_index
            sentence_tensor[:len(sentence)] = torch.FloatTensor(sentence)
        assert len(sentence_tensor) == sen_len
        return sentence_tensor

    def convertSmiles(self, smiles):
        encoded_selfies = []
        for i in smiles:
            encoded_selfies.append(sf.encoder(i))
        return encoded_selfies
