#!/bin/bash
# Run ChemBERTa + SMILES

python models/main/chemberta_smiles.py --mode finetune --epoch 100 --batch_size 512 
