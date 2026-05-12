#!/bin/bash
# Run ProtT5 + SMILES

python models/main/prot5_smiles.py --mode finetune --epoch 100 --batch_size 512 --path human_cross.pth
