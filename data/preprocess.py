"""Preprocess CHIMERA dataset into RAAD's native format.

Thanks to: https://github.com/LirongWu/RAAD.git

Generates a master JSONL of all complexes, runs EquiAACDataset
preprocessing (PDB -> AAComplex -> pkl cache), and saves mappings for
complex_id resolution during training/evaluation.

Output structure (in trans_baselines/raad/):
    all.jsonl              -- master JSONL with all complexes
    all_processed/         -- pkl cache (native format)
    idx_to_cid.json        -- [complex_id, ...] ordered by dataset index
    complex_ids.json       -- {chimera_complex_id: dataset_index}

Usage:
    cd evostruct
    python preprocess.py
"""

import json
import os
import sys

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHIMERA_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

sys.path.insert(0, _SCRIPT_DIR)
from data.dataset import EquiAACDataset
from data.pdb_utils import AAComplex, Protein
from utils import print_log

from chimera_utils import generate_master_jsonl, load_shared_config


class RobustEquiAACDataset(EquiAACDataset):
    """EquiAACDataset that skips PDBs that fail parsing.

    Tracks which JSONL entries succeeded via `self.succeeded_cids`.
    """

    def __init__(self, *args, **kwargs):
        self.succeeded_cids = []
        super().__init__(*args, **kwargs)

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
            self.succeeded_cids.append(item["complex_id"])
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)


def main():
    shared = load_shared_config()
    output_dir = os.path.join(shared["paths"]["trans_baselines"], "raad")
    data_root = shared["paths"]["data_root"]
    os.makedirs(output_dir, exist_ok=True)

    idx_to_cid_path = os.path.join(output_dir, "idx_to_cid.json")
    mapping_path = os.path.join(output_dir, "complex_ids.json")

    if os.path.exists(idx_to_cid_path) and os.path.exists(mapping_path):
        print(f"Mappings already exist: {idx_to_cid_path}")
        print("To reprocess from scratch, delete idx_to_cid.json, "
              "complex_ids.json, and all_processed/")
        jsonl_path = os.path.join(output_dir, "all.jsonl")
        dataset = RobustEquiAACDataset(jsonl_path, interface_only=1)
        with open(idx_to_cid_path) as f:
            idx_to_cid = json.load(f)
        print(f"Dataset: {dataset.num_entry} complexes, "
              f"idx_to_cid: {len(idx_to_cid)} entries")
        assert len(idx_to_cid) == dataset.num_entry, (
            f"Mismatch: {len(idx_to_cid)} idx_to_cid vs {dataset.num_entry} "
            f"dataset entries. Delete all_processed/, idx_to_cid.json, and "
            f"complex_ids.json to reprocess."
        )
        print("Preprocessing already complete.")
        return

    jsonl_path = os.path.join(output_dir, "all.jsonl")
    if not os.path.exists(jsonl_path):
        generate_master_jsonl(jsonl_path, data_root)
    else:
        print(f"Master JSONL already exists: {jsonl_path}")

    cache_dir = os.path.join(output_dir, "all_processed")
    metainfo_file = os.path.join(cache_dir, "_metainfo")
    if os.path.exists(metainfo_file):
        import shutil
        print(f"Removing stale cache {cache_dir} to force fresh preprocessing...")
        shutil.rmtree(cache_dir)

    print("Creating EquiAACDataset (this parses all PDBs)...")
    dataset = RobustEquiAACDataset(jsonl_path, interface_only=1)
    print(f"Dataset loaded: {dataset.num_entry} complexes")

    idx_to_cid = dataset.succeeded_cids
    assert len(idx_to_cid) == dataset.num_entry, (
        f"Mismatch: {len(idx_to_cid)} succeeded_cids vs {dataset.num_entry} dataset entries"
    )

    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}

    with open(idx_to_cid_path, "w") as f:
        json.dump(idx_to_cid, f, indent=2)
    print(f"Saved idx_to_cid ({len(idx_to_cid)} entries) to {idx_to_cid_path}")

    with open(mapping_path, "w") as f:
        json.dump(cid_to_idx, f, indent=2)
    print(f"Saved complex_ids ({len(cid_to_idx)} entries) to {mapping_path}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
