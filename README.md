# ctDrug: Conditional Target-Aware Drug Design

This repository contains the official implementation of **ctDrug**, a ligand-based autoregressive framework for target-aware de novo molecular generation.
## Reproduction of Results

### Main Results (Tables 2-3)
#### Train
python models/main/prott5_smiles.py --batch_size 512 --mode train 
python models/main/prott5_selfies.py --batch_size 512 --mode train 
#### Fine-tune
| Model | Command |
|-------|---------|
| ProtT5 + SMILES | `python models/main/prott5_smiles.py --mode finetune` |
| ProtT5 + SELFIES | `python models/main/prott5_selfies.py --mode finetune` |
| ChemBERTa + SMILES | `python models/main/chemberta_smiles.py --mode finetune` |
| ChemBERTa + SELFIES | `python models/main/chemberta_selfies.py --mode finetune` |

### Ablation Study Results (Table 6)

| Mechanism | Command |
|-----------|---------|
| Cross-Attention | `python models/ablation/cross_attention.py --mode pretrain --subsample 0.2` |
| Concatenation | `python models/ablation/concatenation.py --mode pretrain --subsample 0.2` |
| FiLM | `python models/ablation/film.py --mode pretrain --subsample 0.2` |

### Inference (Generate Molecules)

DRD2 (P14416)

HTR1A (P08908)

```bash
python models/main/prott5_smiles.py --mode infer --target P14416 --num 10000
