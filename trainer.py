"""EvoStruct (CoRe v30h) CHIMERA trainer, Hydra-based.

Trains DockDesigner with a structural adapter on top of frozen (phase 1) then
partially-unfrozen (phase 2+) ESM-2, with shadow paratope + dock + RAAD native losses.

Supports 10 training strategies (combinable via Hydra overrides):
    training.ema=true               EMA weight averaging (decay 0.999)
    training.es_metric=aar          Early stop on val AAR instead of val loss
    training.label_smoothing=0.1    Label smoothing on seq CE loss
    training.rdrop_alpha=1.0        R-Drop regularization
    training.swa=true               Stochastic Weight Averaging in final phase
    training.lora=true              LoRA adapters on ESM-2
    training.disc_lr_decay=0.7      Discriminative LR decay per ESM layer
    training.neftune_alpha=5        NEFTune noise on ESM embeddings
    training.l2sp_lambda=0.01       L2-SP regularization toward pretrained

Usage:
    python trainer.py                                    # defaults
    python trainer.py training.gpu=1 training.max_epoch=50
    python trainer.py model.gamma=0.3 training.rdrop_alpha=2.0
    python trainer.py dataset.split=antigen_fold dataset.cdr_type=\"2\"
"""

import copy
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHIMERA_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

sys.path.insert(0, _SCRIPT_DIR)
from model.core import DockDesigner, LoRALinear
from data.dataset import EquiAACDataset
from data.pdb_utils import AAComplex, Protein, VOCAB
from utils import print_log


from chimera_utils import (
    load_shared_config, load_split_ids, split_dataset, DatasetView,
    EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, to_device, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)

sys.path.insert(0, _CHIMERA_ROOT)
from benchmark.evaluation.metrics import (
    aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score,
    count_liabilities,
)

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities"]
TOPK_KEYS = ["top3_aar", "top5_aar", "top3_caar", "top5_caar"]
LOSS_KEYS = ["loss", "seq", "coord", "pair", "dock", "shadow"]


# ---------------------------------------------------------------------------
# Top-K AAR from logits
# ---------------------------------------------------------------------------
def compute_topk_metrics(logits, true_seq, contact_positions=None, k_values=(3, 5)):
    results = {}
    L = len(true_seq)
    if logits.shape[0] != L:
        return {f'top{k}_aar': 0.0 for k in k_values}

    true_indices = [VOCAB.symbol_to_idx(aa) for aa in true_seq]

    for k in k_values:
        topk_preds = torch.topk(logits, k=k, dim=-1).indices
        in_topk = torch.tensor([
            true_indices[i] in topk_preds[i].tolist()
            for i in range(L)
        ])
        results[f'top{k}_aar'] = in_topk.float().mean().item()

        if contact_positions and len(contact_positions) > 0:
            contact_in_topk = [
                in_topk[i].item()
                for i in contact_positions if i < L
            ]
            if contact_in_topk:
                results[f'top{k}_caar'] = sum(contact_in_topk) / len(contact_in_topk)
            else:
                results[f'top{k}_caar'] = results[f'top{k}_aar']
        else:
            results[f'top{k}_caar'] = results[f'top{k}_aar']

    return results


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = OrderedDict()
        self.backup = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay)

    def apply(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def reinit(self, model):
        self.shadow.clear()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


# ---------------------------------------------------------------------------
# Robust dataset (skips PDB parse failures)
# ---------------------------------------------------------------------------
class RobustEquiAACDataset(EquiAACDataset):
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        with open(file_path, "r") as fin:
            lines = fin.read().strip().split("\n")
        for line in tqdm(lines, desc="Preprocessing"):
            item = json.loads(line)
            try:
                protein = Protein.from_pdb(item["pdb_data_path"])
            except Exception as e:
                print_log(f'parse {item["pdb"]} failed: {e}, skip', level="ERROR")
                continue
            pdb_id, peptides = item["pdb"], protein.peptides
            try:
                self.data.append(AAComplex(
                    pdb_id, peptides, item["heavy_chain"],
                    item["light_chain"], item["antigen_chains"]))
            except Exception as e:
                print_log(f'AAComplex {pdb_id} failed: {e}, skip', level="ERROR")
                continue
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------
def _aggregate(m_lists):
    return {k: float(np.mean(v)) for k, v in m_lists.items()}


def train_epoch(model, loader, optimizer, device, grad_clip, rdrop_alpha=0.0,
                ema_model=None, l2sp_lambda=0.0, pretrained_state=None):
    model.train()
    model.esm_model.eval()
    keys = list(LOSS_KEYS)
    if rdrop_alpha > 0:
        keys.append("rdrop")
    if l2sp_lambda > 0:
        keys.append("l2sp")
    ms = {k: [] for k in keys}

    for batch in loader:
        batch = to_device(batch, device)
        X, S, L, offsets = batch["X"], batch["S"], batch["L"], batch["offsets"]

        if rdrop_alpha > 0:
            loss1, seq1, coord1, pair1, dock1, shad1 = model(X, S, L, offsets)
            loss2, seq2, coord2, pair2, dock2, shad2 = model(X, S, L, offsets)
            base_loss = (loss1 + loss2) / 2
            rdrop_loss = ((seq1 - seq2) ** 2)
            loss = base_loss + rdrop_alpha * rdrop_loss
            ms["rdrop"].append(rdrop_loss.item())
            ms["loss"].append(loss.item())
            ms["seq"].append(((seq1 + seq2) / 2).item())
            ms["coord"].append(((coord1 + coord2) / 2).item())
            ms["pair"].append(((pair1 + pair2) / 2).item())
            ms["dock"].append(((dock1 + dock2) / 2).item())
            ms["shadow"].append(((shad1 + shad2) / 2).item())
        else:
            loss, seq_l, coord_l, pair_l, dock_l, shadow_l = model(X, S, L, offsets)
            ms["loss"].append(loss.item())
            ms["seq"].append(seq_l.item())
            ms["coord"].append(coord_l.item())
            ms["pair"].append(pair_l.item())
            ms["dock"].append(dock_l.item())
            ms["shadow"].append(shadow_l.item())

        if l2sp_lambda > 0 and pretrained_state is not None:
            l2sp_loss = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if param.requires_grad and name in pretrained_state:
                    l2sp_loss = l2sp_loss + (param - pretrained_state[name]).pow(2).sum()
            loss = loss + l2sp_lambda * l2sp_loss
            ms["l2sp"].append(l2sp_loss.item())

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

    return _aggregate(ms)


def valid_epoch(model, loader, device):
    model.eval()
    ms = {k: [] for k in LOSS_KEYS}
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            loss, seq_l, coord_l, pair_l, dock_l, shadow_l = model(
                batch["X"], batch["S"], batch["L"], batch["offsets"])
            ms["loss"].append(loss.item())
            ms["seq"].append(seq_l.item())
            ms["coord"].append(coord_l.item())
            ms["pair"].append(pair_l.item())
            ms["dock"].append(dock_l.item())
            ms["shadow"].append(shadow_l.item())
    return _aggregate(ms)


# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------
def run_inference_with_metrics(model, dataset, loader, device, cdr_type, idx_to_cid,
                               complex_features_dir=None):
    model.eval()
    predictions = []
    all_metrics = {k: [] for k in METRIC_KEYS + TOPK_KEYS}
    idx = 0
    cdr_label = f"H{cdr_type}"

    cdr_to_mask_val = {'H1': 0, 'H2': 1, 'H3': 2, 'L1': 0, 'L2': 1, 'L3': 2}
    contact_cache = {}
    if complex_features_dir and os.path.isdir(complex_features_dir):
        for pt_file in os.listdir(complex_features_dir):
            if pt_file.endswith('.pt'):
                cid = pt_file.replace('.pt', '')
                try:
                    feat = torch.load(os.path.join(complex_features_dir, pt_file),
                                     map_location='cpu', weights_only=False)
                    chain_type = 'heavy' if cdr_label.startswith('H') else 'light'
                    cdr_mask_val = cdr_to_mask_val.get(cdr_label, 2)
                    cdr_mask = feat.get('cdr_masks', {}).get('imgt', {}).get(chain_type, [])
                    cdr_positions = [i for i, v in enumerate(cdr_mask) if v == cdr_mask_val]

                    if not cdr_positions:
                        continue

                    numbering = feat.get('numbering', {}).get('imgt', {}).get(chain_type, [])
                    heavy_chain = cid.split('_')[1]
                    light_chain = cid.split('_')[2] if len(cid.split('_')) > 2 else ''
                    target_chain = heavy_chain if chain_type == 'heavy' else light_chain
                    paratope = feat.get('paratope_residues', [])
                    paratope_resids = {resid for chain, resid, aa in paratope if chain == target_chain}

                    contact_pos = []
                    for i, pos in enumerate(cdr_positions):
                        if pos < len(numbering):
                            num_entry = numbering[pos]
                            if isinstance(num_entry, tuple) and len(num_entry) >= 1:
                                resid = num_entry[0]
                                if resid in paratope_resids:
                                    contact_pos.append(i)

                    contact_cache[cid] = contact_pos
                except Exception:
                    pass

    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch["L"])
            ppls, seqs, xs, true_xs, logits_list = model.infer(batch, device)
            for i in range(batch_size):
                dataset_idx = dataset.idx_mapping[idx]
                cplx = dataset.data[dataset_idx]
                cid = idx_to_cid[dataset_idx]

                origin_seq = cplx.get_cdr(cdr_label).get_seq()
                pred_seq = seqs[i]
                pred_x = xs[i]
                true_x = true_xs[i]
                logits = logits_list[i]

                pred_ca = pred_x[:, 1, :] if pred_x.ndim == 3 else pred_x
                true_ca = true_x[:, 1, :] if true_x.ndim == 3 else true_x
                if isinstance(pred_ca, torch.Tensor):
                    pred_ca = pred_ca.cpu().numpy()
                if isinstance(true_ca, torch.Tensor):
                    true_ca = true_ca.cpu().numpy()

                aar_val = chimera_aar(pred_seq, origin_seq)
                rmsd_val = kabsch_rmsd(pred_ca, true_ca)
                tm_val = chimera_tm_score(pred_ca, true_ca)
                liab = count_liabilities(pred_seq)

                contact_pos = contact_cache.get(cid, None)
                topk = compute_topk_metrics(logits, origin_seq, contact_pos)

                all_metrics["ppl"].append(ppls[i])
                all_metrics["aar"].append(aar_val)
                all_metrics["rmsd"].append(rmsd_val)
                all_metrics["tm_score"].append(tm_val)
                all_metrics["n_liabilities"].append(liab)
                for tk in TOPK_KEYS:
                    all_metrics[tk].append(topk.get(tk, 0.0))

                predictions.append({
                    "complex_id": cid,
                    "cdr_type": cdr_label,
                    "pred_sequence": pred_seq,
                    "true_sequence": origin_seq,
                    "pred_coords": pred_ca,
                    "true_coords": true_ca,
                    "ppl": ppls[i],
                    "aar": aar_val,
                    "rmsd": rmsd_val,
                    "tm_score": tm_val,
                    "n_liabilities": liab,
                    "logits": logits.numpy(),
                    **topk,
                })
                idx += 1

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
    return predictions, summary


def save_test_csv(summary, cdr_label, topk_summary=None):
    csv_path = os.path.join(_SCRIPT_DIR, "test_metrics.csv")
    row = {"cdr_type": cdr_label}
    for k in FULL_METRIC_KEYS:
        v = summary.get(k, {})
        if isinstance(v, dict):
            row[k] = f"{v.get('mean', 0):.2f}\u00b1{v.get('std', 0):.2f}"
        else:
            row[k] = v
    if topk_summary:
        for tk in TOPK_KEYS:
            row[tk] = f"{topk_summary.get(tk, 0.0):.4f}"
    fieldnames = ["cdr_type"] + list(FULL_METRIC_KEYS) + list(TOPK_KEYS)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved test metrics CSV to {csv_path}")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def _build_optimizer(model, c, lr):
    if c.disc_lr_decay > 0:
        total_layers = len(model.esm_model.layers)
        param_groups = []
        esm_layer_params = {}
        non_esm_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("esm_model.layers."):
                layer_idx = int(name.split(".")[2])
                esm_layer_params.setdefault(layer_idx, []).append(p)
            else:
                non_esm_params.append(p)
        if non_esm_params:
            param_groups.append({"params": non_esm_params, "lr": lr})
        for layer_idx in sorted(esm_layer_params.keys()):
            depth_from_top = total_layers - 1 - layer_idx
            layer_lr = lr * (c.disc_lr_decay ** depth_from_top)
            param_groups.append({
                "params": esm_layer_params[layer_idx], "lr": layer_lr})
        print(f"  Discriminative LR: {len(param_groups)} groups, "
              f"decay={c.disc_lr_decay}")
    else:
        param_groups = [{"params": [p for p in model.parameters()
                                    if p.requires_grad], "lr": lr}]

    OptimCls = torch.optim.AdamW if c.weight_decay > 0 else torch.optim.Adam
    opt_kwargs = {}
    if c.weight_decay > 0:
        opt_kwargs["weight_decay"] = c.weight_decay
    return OptimCls(param_groups, **opt_kwargs)


# ---------------------------------------------------------------------------
# Single training phase
# ---------------------------------------------------------------------------
def _run_phase(phase_name, model, train_loader, valid_loader, valid_set,
               idx_to_cid, c, device, ckpt, start_epoch, max_epoch, lr,
               wandb_run, ema_model=None, swa_model=None, swa_start_epoch=None,
               pretrained_state=None):
    optimizer = _build_optimizer(model, c, lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: c.anneal_base ** epoch)

    es_mode = "max" if c.es_metric == "aar" else "min"
    early_stop = EarlyStopping(patience=c.patience, mode=es_mode)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    strat_str = []
    if c.ema: strat_str.append("EMA")
    if c.es_metric != "loss": strat_str.append(f"ES={c.es_metric}")
    if c.label_smoothing > 0: strat_str.append(f"LS={c.label_smoothing}")
    if c.rdrop_alpha > 0: strat_str.append(f"RDrop={c.rdrop_alpha}")
    if c.swa and swa_model is not None: strat_str.append("SWA")
    if c.weight_decay > 0: strat_str.append(f"WD={c.weight_decay}")
    if c.lora: strat_str.append(f"LoRA(r={c.lora_rank})")
    if c.disc_lr_decay > 0: strat_str.append(f"DiscLR={c.disc_lr_decay}")
    if c.neftune_alpha > 0: strat_str.append(f"NEFTune={c.neftune_alpha}")
    if c.l2sp_lambda > 0: strat_str.append(f"L2SP={c.l2sp_lambda}")
    strat = f" [{', '.join(strat_str)}]" if strat_str else ""
    print(f"=== {phase_name} | lr={lr:.1e} | trainable params={n_trainable:,}{strat} ===")

    epoch_idx = start_epoch
    for _ep in range(max_epoch):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, device,
                              c.grad_clip, rdrop_alpha=c.rdrop_alpha,
                              ema_model=ema_model,
                              l2sp_lambda=c.l2sp_lambda,
                              pretrained_state=pretrained_state)
        if ema_model is not None:
            ema_model.apply(model)
        val_m = valid_epoch(model, valid_loader, device)
        _, val_metrics = run_inference_with_metrics(
            model, valid_set, valid_loader, device, c.cdr_type, idx_to_cid)
        if ema_model is not None:
            ema_model.restore(model)

        scheduler.step()

        if c.es_metric == "aar":
            ckpt_metric = -val_metrics.get("aar", 0)
        else:
            ckpt_metric = val_m["loss"]

        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        is_best = ckpt.save(model, optimizer, scheduler, epoch_idx, ckpt_metric)
        if is_best and ema_model is not None:
            ema_path = os.path.join(str(ckpt.save_dir), "best.pt")
            state = torch.load(ema_path, map_location="cpu", weights_only=False)
            state["ema_state"] = {k: v.cpu() for k, v in ema_model.shadow.items()}
            torch.save(state, ema_path)

        rdrop_str = f" rdrop={train_m.get('rdrop', 0):.3f}" if c.rdrop_alpha > 0 else ""
        print(f"[{phase_name}] Epoch {epoch_idx:3d} | "
              f"loss={train_m['loss']:.4f}/{val_m['loss']:.4f} "
              f"seq={train_m['seq']:.3f}/{val_m['seq']:.3f} "
              f"coord={train_m['coord']:.3f}/{val_m['coord']:.3f} "
              f"dock={train_m['dock']:.3f}/{val_m['dock']:.3f} "
              f"shad={train_m['shadow']:.3f}/{val_m['shadow']:.3f}"
              f"{rdrop_str} "
              f"aar={val_metrics.get('aar', 0):.3f} "
              f"rmsd={val_metrics.get('rmsd', 0):.2f} "
              f"lr={cur_lr:.6f} {'*' if is_best else ''} [{elapsed:.0f}s]")

        log_dict = {
            "epoch": epoch_idx, "lr": cur_lr, "phase": phase_name,
            "train_loss": train_m["loss"], "val_loss": val_m["loss"],
            "train_seq": train_m["seq"], "val_seq": val_m["seq"],
            "train_coord": train_m["coord"], "val_coord": val_m["coord"],
            "train_pair": train_m["pair"], "val_pair": val_m["pair"],
            "train_dock": train_m["dock"], "val_dock": val_m["dock"],
            "train_shadow": train_m["shadow"], "val_shadow": val_m["shadow"],
        }
        log_dict.update({f"val_{k}": v for k, v in val_metrics.items()})
        if c.rdrop_alpha > 0:
            log_dict["train_rdrop"] = train_m.get("rdrop", 0)
        if wandb_run:
            wandb_run.log(log_dict)

        if swa_model is not None and swa_start_epoch is not None:
            if epoch_idx >= swa_start_epoch:
                swa_model.update_parameters(model)

        epoch_idx += 1
        if early_stop(ckpt_metric):
            print(f"[{phase_name}] Early stopping at epoch {epoch_idx - 1}")
            break

    return epoch_idx


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(c, wandb_run=None):
    seed_everything(c.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(c.gpu)
    device = torch.device(f"cuda:{c.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cdr_label = f"H{c.cdr_type}"
    run_name = c.run_name or f"evostruct_cdr{c.cdr_type}_{c.split}"
    save_dir = os.path.join(c.results_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(c), f, indent=2)

    data_dir = os.path.abspath(c.data_dir)
    jsonl_path = os.path.join(data_dir, "all.jsonl")
    print(f"Loading dataset from {data_dir}...")
    dataset = RobustEquiAACDataset(jsonl_path, interface_only=c.interface_only)
    dataset.mode = c.mode
    print(f"Dataset: {dataset.num_entry} complexes")

    idx_to_cid_path = os.path.join(data_dir, "idx_to_cid.json")
    with open(idx_to_cid_path) as f:
        idx_to_cid = json.load(f)
    assert len(idx_to_cid) == dataset.num_entry

    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
    split_ids = load_split_ids(c.split, c.data_root)
    views = split_dataset(dataset, split_ids, cid_to_idx)
    train_set, valid_set, test_set = views["train"], views["val"], views["test"]

    n_channel = dataset[0]["X"].shape[1]

    model = DockDesigner(
        c.embed_size, c.hidden_size, n_channel,
        n_layers=c.n_layers, dropout=c.dropout,
        cdr_type=c.cdr_type, args=c,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"DockDesigner params: {n_trainable:,} trainable / {n_total:,} total")

    strats = []
    if c.ema: strats.append("EMA")
    if c.es_metric != "loss": strats.append(f"ES={c.es_metric}")
    if c.label_smoothing > 0: strats.append(f"LS={c.label_smoothing}")
    if c.rdrop_alpha > 0: strats.append(f"RDrop={c.rdrop_alpha}")
    if c.swa: strats.append("SWA")
    if c.weight_decay > 0: strats.append(f"WD={c.weight_decay}")
    if c.joint_lora: strats.append(f"JointLoRA(r={c.lora_rank})")
    elif c.lora: strats.append(f"LoRA(r={c.lora_rank})")
    if c.disc_lr_decay > 0: strats.append(f"DiscLR={c.disc_lr_decay}")
    if c.neftune_alpha > 0: strats.append(f"NEFTune={c.neftune_alpha}")
    if c.l2sp_lambda > 0: strats.append(f"L2SP={c.l2sp_lambda}")
    if c.esm_unfreeze_top != 4: strats.append(f"ESM_top={c.esm_unfreeze_top}")
    if strats:
        print(f"Active strategies: {', '.join(strats)}")

    if c.test_only:
        ckpt_path = c.checkpoint or os.path.join(save_dir, "checkpoints", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found at {ckpt_path}")
            sys.exit(1)
        print(f"Loading checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        if "ema_state" in state and state["ema_state"]:
            print("Loading EMA weights for inference...")
            for name, param in model.named_parameters():
                if name in state["ema_state"]:
                    param.data.copy_(state["ema_state"][name])
    else:
        train_loader = DataLoader(
            train_set, batch_size=c.batch_size, shuffle=True,
            num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)
        valid_loader = DataLoader(
            valid_set, batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)

        ckpt = ModelCheckpoint(save_dir, mode="min")
        ema_model = EMAModel(model, decay=c.ema_decay) if c.ema else None

        if c.joint_lora:
            n_lora = model.inject_lora(
                c.esm_unfreeze_top, rank=c.lora_rank, alpha=c.lora_alpha)
            print(f"Joint LoRA: injected LoRA (rank={c.lora_rank}) into top "
                  f"{c.esm_unfreeze_top} ESM layers ({n_lora:,} params)")
            if c.neftune_alpha > 0:
                model.set_neftune(c.neftune_alpha)
                print(f"NEFTune enabled (alpha={c.neftune_alpha})")
            total_epochs = c.max_epoch + c.phase2_epochs + c.phase3_epochs
            epoch_cursor = _run_phase(
                "joint", model, train_loader, valid_loader, valid_set,
                idx_to_cid, c, device, ckpt,
                start_epoch=0, max_epoch=total_epochs, lr=c.lr,
                wandb_run=wandb_run, ema_model=ema_model)
        else:
            # Phase 1: adapter + GNN + seq_head; ESM fully frozen
            epoch_cursor = _run_phase(
                "phase1", model, train_loader, valid_loader, valid_set,
                idx_to_cid, c, device, ckpt,
                start_epoch=0, max_epoch=c.max_epoch, lr=c.lr,
                wandb_run=wandb_run, ema_model=ema_model)

        # Phase 2: fine-tune ESM (LoRA or full unfreezing)
        if not c.joint_lora and c.phase2_epochs > 0 and c.esm_unfreeze_top > 0:
            print(f"Phase 1 done. Loading phase 1 best before ESM fine-tuning.")
            ckpt.load_best(model, device)

            pretrained_state = None
            if c.l2sp_lambda > 0:
                pretrained_state = {
                    name: param.data.clone().to(device)
                    for name, param in model.named_parameters()
                    if param.requires_grad}

            if c.lora:
                n_lora = model.inject_lora(
                    c.esm_unfreeze_top, rank=c.lora_rank, alpha=c.lora_alpha)
                print(f"Injected LoRA (rank={c.lora_rank}) into top "
                      f"{c.esm_unfreeze_top} ESM layers ({n_lora:,} new LoRA params)")
            else:
                trainable_esm = model.set_esm_unfreeze_top(c.esm_unfreeze_top)
                print(f"Unfroze top {c.esm_unfreeze_top} ESM layers "
                      f"({trainable_esm:,} new trainable params)")

            if c.neftune_alpha > 0:
                model.set_neftune(c.neftune_alpha)
                print(f"NEFTune enabled (alpha={c.neftune_alpha})")

            if c.l2sp_lambda > 0:
                for name, param in model.named_parameters():
                    if param.requires_grad and name not in pretrained_state:
                        pretrained_state[name] = param.data.clone().to(device)
                print(f"L2-SP regularization enabled (lambda={c.l2sp_lambda})")

            if ema_model is not None:
                ema_model.reinit(model)

            swa_model = None
            swa_start_epoch = None
            if c.swa:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
                swa_start_epoch = epoch_cursor + int(c.phase2_epochs * c.swa_start_pct)
                print(f"SWA will start at epoch {swa_start_epoch}")

            epoch_cursor = _run_phase(
                "phase2", model, train_loader, valid_loader, valid_set,
                idx_to_cid, c, device, ckpt,
                start_epoch=epoch_cursor, max_epoch=c.phase2_epochs,
                lr=c.phase2_lr, wandb_run=wandb_run,
                ema_model=ema_model, swa_model=swa_model,
                swa_start_epoch=swa_start_epoch,
                pretrained_state=pretrained_state)

            if swa_model is not None and hasattr(swa_model, 'n_averaged') and swa_model.n_averaged > 0:
                print(f"SWA: averaged {swa_model.n_averaged.item()} checkpoints, updating BN...")
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
                model.load_state_dict(swa_model.module.state_dict(), strict=False)

        # Phase 3: joint low-LR polish
        if not c.joint_lora and c.phase3_epochs > 0:
            print(f"Phase 2 done. Loading best before phase 3 polish.")
            ckpt.load_best(model, device)
            if ema_model is not None:
                ema_model.reinit(model)
            epoch_cursor = _run_phase(
                "phase3", model, train_loader, valid_loader, valid_set,
                idx_to_cid, c, device, ckpt,
                start_epoch=epoch_cursor, max_epoch=c.phase3_epochs,
                lr=c.phase3_lr, wandb_run=wandb_run, ema_model=ema_model)

        print("Loading best model for test...")
        ckpt.load_best(model, device)

        if ema_model is not None:
            best_state = torch.load(
                os.path.join(save_dir, "checkpoints", "best.pt"),
                map_location=device)
            if "ema_state" in best_state and best_state["ema_state"]:
                print("Using EMA weights for test inference.")
                for name, param in model.named_parameters():
                    if name in best_state["ema_state"]:
                        param.data.copy_(best_state["ema_state"][name].to(device))

    # Test inference
    test_loader = DataLoader(
        test_set, batch_size=c.batch_size, shuffle=False,
        num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)
    complex_features_dir = os.path.join(c.data_root, "processed", "complex_features")
    predictions, test_summary = run_inference_with_metrics(
        model, test_set, test_loader, device, c.cdr_type, idx_to_cid,
        complex_features_dir=complex_features_dir)

    pred_dir = os.path.join(save_dir, "predictions", cdr_label)
    save_predictions(predictions, pred_dir)
    print(f"Saved {len(predictions)} predictions to {pred_dir}")

    _, full_summary, _ = run_full_evaluation(
        pred_dir, c.split, c.data_root, cdr_type_hint=cdr_label,
        numbering_scheme=c.numbering_scheme)

    save_test_csv(full_summary, cdr_label, topk_summary=test_summary)

    print(f"\nTest results ({len(predictions)} complexes, {cdr_label}):")
    for k in FULL_METRIC_KEYS:
        v = full_summary.get(k, {})
        if isinstance(v, dict):
            print(f"  {k}: {v.get('mean', 0):.2f}\u00b1{v.get('std', 0):.2f}")
        else:
            print(f"  {k}: {v}")

    print("\nTop-K metrics:")
    for tk in TOPK_KEYS:
        v = test_summary.get(tk, 0.0)
        print(f"  {tk}: {v:.4f}")

    if wandb_run:
        for k in FULL_METRIC_KEYS:
            v = full_summary.get(k, {})
            if isinstance(v, dict):
                v = v.get("mean", 0)
            elif isinstance(v, str) and "\u00b1" in v:
                v = float(v.split("\u00b1")[0])
            wandb_run.log({f"test_{cdr_label}_{k}": v})
        wandb_run.log({"test_n": len(predictions)})

    return full_summary


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    m = cfg.model
    t = cfg.training
    d = cfg.dataset

    c = SimpleNamespace(
        # Paths
        data_root=cfg.data_root,
        data_dir=cfg.data_dir,
        results_dir=cfg.results_dir,
        numbering_scheme=cfg.numbering_scheme,
        # Dataset
        split=d.split,
        cdr_type=str(d.cdr_type),
        # Model
        embed_size=m.embed_size,
        hidden_size=m.hidden_size,
        n_layers=m.n_layers,
        dropout=m.dropout,
        alpha=m.alpha,
        beta=m.beta,
        gamma=m.gamma,
        delta=m.delta,
        dock_cutoff=m.dock_cutoff,
        num_attn_heads=m.num_attn_heads,
        node_feats_mode=str(m.node_feats_mode),
        edge_feats_mode=str(m.edge_feats_mode),
        interface_only=m.interface_only,
        mode=str(m.mode),
        esm_model_name=m.esm_model_name,
        adapter_n_blocks=m.adapter_n_blocks,
        adapter_dim=m.adapter_dim,
        adapter_n_heads=m.adapter_n_heads,
        adapter_max_ag_ctx=m.adapter_max_ag_ctx,
        # Training
        seed=t.seed,
        gpu=t.gpu,
        lr=t.lr,
        batch_size=t.batch_size,
        max_epoch=t.max_epoch,
        anneal_base=t.anneal_base,
        grad_clip=t.grad_clip,
        num_workers=t.num_workers,
        esm_unfreeze_top=t.esm_unfreeze_top,
        phase2_epochs=t.phase2_epochs,
        phase2_lr=float(t.phase2_lr),
        phase3_epochs=t.phase3_epochs,
        phase3_lr=float(t.phase3_lr),
        test_only=t.test_only,
        checkpoint=t.checkpoint,
        run_name=t.run_name,
        # Training strategies
        rdrop_alpha=t.rdrop_alpha,
        weight_decay=t.weight_decay,
        label_smoothing=t.label_smoothing,
        es_metric=t.es_metric,
        ema=t.ema,
        ema_decay=t.ema_decay,
        swa=t.swa,
        swa_start_pct=t.swa_start_pct,
        lora=t.lora,
        lora_rank=t.lora_rank,
        lora_alpha=t.lora_alpha,
        joint_lora=t.joint_lora,
        disc_lr_decay=t.disc_lr_decay,
        neftune_alpha=t.neftune_alpha,
        l2sp_lambda=t.l2sp_lambda,
        # Callbacks
        patience=cfg.callbacks.early_stopping.patience,
        es_mode=cfg.callbacks.early_stopping.mode,
        ckpt_mode=cfg.callbacks.checkpoint.mode,
        # WandB
        use_wandb=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
    )

    # joint_lora implies lora
    if c.joint_lora:
        c.lora = True

    wandb_run = None
    if c.use_wandb and not c.test_only:
        wandb_run = setup_wandb(
            c.wandb_project,
            c.run_name or f"evostruct_cdr{c.cdr_type}_{c.split}",
            OmegaConf.to_container(cfg),
            enabled=True,
        )

    full_summary = run_pipeline(c, wandb_run)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
