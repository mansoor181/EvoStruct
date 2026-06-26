#!/bin/bash
# EvoStruct training script 
# Usage: bash chimera_train.sh --gpu 0 --epochs 50
# bash chimera_train.sh --gpu 1 --epochs 1 --p2-epochs 1 --p3-epochs 1

# ==============================================================================
# TRAINING CONFIG - From WandB
# ==============================================================================

# Training
gpu=0
seed=42
batch_size=4
grad_clip=0.5
num_workers=4

# Phase 1: adapter + GNN + seq_head (ESM frozen)
lr=1.0e-4
max_epoch=50
anneal_base=0.9

# Phase 2: unfreeze top 4 ESM-2 layers
esm_unfreeze_top=4
phase2_epochs=40
phase2_lr=5.0e-5

# Phase 3: joint low LR polish
phase3_epochs=30
phase3_lr=1.0e-5

# Dataset
split="antigen_fold"
cdr_type="3"

# Model architecture
embed_size=32
hidden_size=256
n_layers=5
dropout=0.2
num_attn_heads=4

# ESM-2
esm_model_name="esm2_t33_650M_UR50D"

# Mini Structural Adapter
adapter_n_blocks=1
adapter_dim=640
adapter_n_heads=8
adapter_max_ag_ctx=128

# Loss weights
alpha=1.376
beta=0.525
gamma=0.5
delta=0.3
dock_cutoff=8.0

# Training strategy
rdrop_alpha=1.0
weight_decay=0.0
es_metric="loss"

# Feature modes
node_feats_mode="1111"
edge_feats_mode="1111"
interface_only=1
mode="111"

# Callbacks
early_stopping_patience=10
checkpoint_mode="min"

# WandB
wandb_enabled=true
wandb_project="chimera"

# Run name
run_name="evostruct-retrain-antigenfold"

# ==============================================================================
# CLI OVERRIDES - These override the above config
# ==============================================================================

usage() {
    echo "Usage: $0 [--gpu <id>] [--epochs <n>] [--p2-epochs <n>] [--p3-epochs <n>]"
    echo ""
    echo "All other parameters are configured in the script itself."
    echo "Edit the CONFIG section at the top of this file."
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Parse CLI args (optional overrides)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) gpu="$2"; shift 2 ;;
        --epochs) max_epoch="$2"; shift 2 ;;
        --p2-epochs) phase2_epochs="$2"; shift 2 ;;
        --p3-epochs) phase3_epochs="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

# ==============================================================================
# RUN TRAINING
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

timestamp=$(date +%Y%m%d-%H%M%S)
log_file="logs/${timestamp}.log"

echo "=========================================="
echo "EvoStruct Training (Hydra)"
echo "WandB reference: 4x88rr0w"
echo "=========================================="
echo "GPU: $gpu | Batch: $batch_size | Seed: $seed"
echo "Phase 1: lr=$lr, epochs=$max_epoch"
echo "Phase 2: lr=$phase2_lr, epochs=$phase2_epochs, unfreeze_top=$esm_unfreeze_top"
echo "Phase 3: lr=$phase3_lr, epochs=$phase3_epochs"
echo "R-Drop: $rdrop_alpha | Split: $split | CDR: H$cdr_type"
echo "Log: $log_file"
echo "=========================================="

nohup python -u trainer.py \
    training.gpu=$gpu \
    training.seed=$seed \
    training.max_epoch=$max_epoch \
    training.batch_size=$batch_size \
    training.lr=$lr \
    training.anneal_base=$anneal_base \
    training.grad_clip=$grad_clip \
    training.num_workers=$num_workers \
    training.esm_unfreeze_top=$esm_unfreeze_top \
    training.phase2_epochs=$phase2_epochs \
    training.phase2_lr=$phase2_lr \
    training.phase3_epochs=$phase3_epochs \
    training.phase3_lr=$phase3_lr \
    training.rdrop_alpha=$rdrop_alpha \
    training.weight_decay=$weight_decay \
    training.es_metric="$es_metric" \
    training.run_name="$run_name" \
    dataset.split=$split \
    dataset.cdr_type=\"$cdr_type\" \
    model.embed_size=$embed_size \
    model.hidden_size=$hidden_size \
    model.n_layers=$n_layers \
    model.dropout=$dropout \
    model.num_attn_heads=$num_attn_heads \
    model.esm_model_name="$esm_model_name" \
    model.adapter_n_blocks=$adapter_n_blocks \
    model.adapter_dim=$adapter_dim \
    model.adapter_n_heads=$adapter_n_heads \
    model.adapter_max_ag_ctx=$adapter_max_ag_ctx \
    model.alpha=$alpha \
    model.beta=$beta \
    model.gamma=$gamma \
    model.delta=$delta \
    model.dock_cutoff=$dock_cutoff \
    model.node_feats_mode=\"$node_feats_mode\" \
    model.edge_feats_mode=\"$edge_feats_mode\" \
    model.interface_only=$interface_only \
    model.mode=\"$mode\" \
    callbacks.early_stopping.patience=$early_stopping_patience \
    callbacks.checkpoint.mode=$checkpoint_mode \
    wandb.enabled=$wandb_enabled \
    wandb.project=$wandb_project \
    > "$log_file" 2>&1 &

pid=$!
echo "Started (PID: $pid)"
echo "Monitor: tail -f $log_file"
echo "Kill: kill $pid"
