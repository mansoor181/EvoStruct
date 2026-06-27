# EvoStruct: Structural Adapter for PLM-Based CDR Sequence Design

EvoStruct bridges a frozen ESM-2 protein language model with 3D structural context from an E(3)-equivariant GNN via a cross-attention adapter. 

## Training

3-phase progressive unfreezing schedule:

| Phase | Epochs | LR | What trains |
|-------|--------|-----|-------------|
| Phase 1 | up to 50 | 1e-4 | Adapter + GNN + seq_head (ESM frozen) |
| Phase 2 | 40 | 5e-5 | All Phase 1 + top 4 ESM-2 layers |
| Phase 3 | 30 | 1e-5 | Everything (joint low-LR polish) |

### Loss Function

5 loss terms + R-Drop consistency regularization:

| Loss | Weight | Description |
|------|--------|-------------|
| Sequence (CE) | 1.0 | Cross-entropy on predicted CDR sequence |
| Coordinate | 1.376 (alpha) | Smooth L1 on backbone coords |
| Pairing | 0.525 (beta) | Contrastive CDR-antigen pairing (InfoNCE) |
| Docking | 0.5 (gamma) | Min CA distance from predicted CDR to epitope (8A cutoff) |
| Shadow | 0.3 (delta) | |pred_dist_matrix - true_dist_matrix| for CDR-epitope pairs |
| R-Drop | 1.0 | (seq_loss_pass1 - seq_loss_pass2)^2 |



## Results

### CDR-H3, temporal split

| Method | AAR | CAAR | PPL | RMSD | Fnat | DockQ |
|--------|-----|------|-----|------|------|-------|
| **EvoStruct** | **0.43** | 0.22 | **1.88** | 1.84 | 0.61 | 0.70 |
| RAAD | 0.37 | 0.21 | 3.27 | 1.75 | 0.56 | 0.70 |
| MEAN | 0.37 | **0.24** | 3.10 | 1.84 | 0.57 | 0.69 |
| dyMEAN | 0.37 | 0.22 | 3.29 | 2.22 | 0.53 | 0.65 |
| DiffAb | 0.23 | 0.14 | -- | 2.49 | 0.59 | 0.65 |
| AbFlowNet | 0.23 | 0.14 | -- | 2.38 | 0.60 | 0.66 |
| RefineGNN | 0.21 | 0.10 | 8.46 | 2.86 | 0.65 | 0.73 |


## Directory Structure

```
code/
  trainer.py              # Hydra-based training + test inference
  chimera_evaluate.py     # Standalone 12-metric evaluation
  chimera_train.sh        # Shell script to reproduce WandB run
  utils.py                # Logging, seeding utilities
  conf/                   # Hydra config hierarchy
    config.yaml           # Top-level (defaults + paths)
    model/model.yaml      # Architecture + loss weights + ESM + adapter
    training/training.yaml # 3-phase schedule + training strategies
    dataset/dataset.yaml  # Split, CDR type
    callbacks/callbacks.yaml
    wandb/wandb.yaml
  model/
    core.py               # DockDesigner model (EvoStruct architecture)
    modules.py            # RelationEGNN, E(3)-equivariant layers
  data/
    dataset.py            # EquiAACDataset (RAAD-format data loading)
    pdb_utils.py          # PDB parsing, AAComplex, VOCAB
    preprocess.py           # CHIMERA -> RAAD native format conversion
```

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Hydra (`pip install hydra-core omegaconf`)
- ESM-2 (`pip install fair-esm`) -- 650M model auto-downloads on first run
- torch_scatter
- CHIMERA-Bench dataset preprocessed (see `preprocess.py`)
- Shared config: `conf/shared_config.yaml` with data paths
- Shared utils: `chimera_utils.py`

## Usage

### Preprocessing

Converts CHIMERA complex features into RAAD's native PDB/pkl format. 

```bash
cd code
python preprocess.py
```

Output goes to `trans_baselines/raad/` (JSONL + pkl cache + index mappings).

### Training (single run)

Reproduce the paper results (CDR-H3, epitope_group split):

```bash
cd code
bash chimera_train.sh --gpu 0
```

Other CDRs or splits:

```bash
bash chimera_train.sh --gpu 0 --cdr 1 --split antigen_fold
```

### Training (all CDRs x splits)

```bash
bash chimera_train.sh --gpu 0 --all
```

Trains 9 runs (3 CDR types x 3 splits) sequentially. Results written to `evostruct_results.csv`.

### Test-only inference

Load a checkpoint and run test evaluation without training:

```bash
bash chimera_train.sh --test-only --checkpoint /path/to/best.pt
```

### Direct trainer usage (Hydra)

The trainer uses Hydra for configuration. All hyperparameters are overridable via command line:

```bash
python trainer.py \
    dataset.split=epitope_group \
    dataset.cdr_type=\"3\" \
    training.gpu=0 \
    training.rdrop_alpha=1.0 \
    training.max_epoch=50 \
    training.run_name=my_experiment
```

Available training strategies (combinable via Hydra overrides):

| Override | Description |
|----------|-------------|
| `training.rdrop_alpha=1.0` | R-Drop regularization (default for paper) |
| `training.ema=true` | EMA weight averaging (decay 0.999) |
| `training.label_smoothing=0.1` | Label smoothing on seq CE |
| `training.lora=true` | LoRA adapters on ESM-2 instead of full unfreezing |
| `training.swa=true` | Stochastic weight averaging in Phase 3 |
| `training.neftune_alpha=5` | NEFTune noise on ESM embeddings |
| `training.disc_lr_decay=0.7` | Discriminative LR decay per ESM layer |
| `training.l2sp_lambda=0.01` | L2-SP regularization toward pretrained weights |
| `training.es_metric=aar` | Early stop on val AAR instead of val loss |

To dump the resolved config without running:

```bash
python trainer.py --cfg job
```

### Standalone evaluation

```bash
# Evaluate a single CDR prediction directory:
python chimera_evaluate.py --predictions /path/to/predictions/ --split epitope_group

# Aggregate all CDR runs for a split:
python chimera_evaluate.py --aggregate --split epitope_group
```

### WandB

Logging is enabled by default (project: `chimera`). Disable with `wandb.enabled=false`.


## CHIMERA-Bench Integration

This model follows the CHIMERA 5-file pattern:

| File | Purpose |
|------|---------|
| `conf/` | Hydra config hierarchy (model, training, dataset, callbacks, wandb) |
| `preprocess.py` | CHIMERA -> RAAD native format (shared with RAAD baseline) |
| `trainer.py` | Hydra-based training + test inference with all 12 CHIMERA metrics |
| `chimera_evaluate.py` | Standalone evaluation (`--predictions` or `--aggregate`) |
| `chimera_train.sh` | Orchestration script with exact WandB run config |

### Prediction format

Per-complex `.pt` files saved to `results_dir/{run_name}/predictions/`:

```python
{
    "complex_id": str,
    "cdr_type": str,         # "H1", "H2", or "H3"
    "pred_sequence": str,    # one-letter AA sequence
    "true_sequence": str,
    "pred_coords": np.array, # (L, 3) CA coordinates
    "true_coords": np.array,
    "ppl": float,            # per-residue perplexity
}
```

### 12 evaluation metrics

Sequence: AAR, CAAR, PPL | Structure: RMSD, TM-score | Interface: Fnat, iRMSD, DockQ | Epitope: F1 | Liabilities: n_liab 

## Key Design Decisions

1. **Sequence head in ESM space (1280D), not GNN space (256D)**. The GNN provides structural context *into* ESM representations via the adapter.

2. **Mini adapter (1 block, 640D)** instead of full-dim (3 blocks, 1280D). Reduces adapter params from ~40M to ~5M with no performance loss.

3. **Antigen context cropping (K=128)**. Only the 128 nearest antigen residues by CA distance are used as KV context, bounding memory while preserving relevant epitope information.

4. **R-Drop regularization**. Two forward passes with different dropout masks; penalizes `(seq_loss_1 - seq_loss_2)^2`. This is NOT KL divergence, but a simpler squared loss difference that encourages prediction consistency.

5. **Progressive ESM unfreezing** (ULMFiT-style). Phase 1 learns the adapter without disturbing ESM. Phase 2 fine-tunes top ESM layers at lower LR. Phase 3 polishes everything jointly.

## Citation

```bibtex
@inproceedings{ahmed2026evostruct,
    title={EvoStruct: Bridging Evolutionary and Structural Priors for Antibody CDR Design via Protein Language Model Adaptation},
    author={Ahmed, Mansoor and Lee, Sujin and Khayaz, Umar and Patterson, Murray},
    booktitle={ICML 2026 Workshop on Generative and Experimental Perspectives for Biomolecular Design},
    year={2026}
}
```

Thanks to:

- RAAD: https://github.com/LirongWu/RAAD.git
- LM-Design: https://github.com/BytedProtein/ByProt.git
- R-Drop: https://github.com/dropreg/R-Drop.git
- ESM: https://github.com/facebookresearch/esm.git