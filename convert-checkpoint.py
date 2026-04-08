"""
convert_checkpoint.py
=====================
Creates tmp/new-checkpoint/ from an existing Qwen3-TTS checkpoint with:
  1. config.json  → talker_config.extended_vocab_size = 4096
  2. model.safetensors → talker.model.codec_embedding.weight resized (3072,2048) → (4096,2048)
     - rows 0-3071: copied from original
     - rows 3072-4095: initialized with N(0, 0.02)
     - row 4000: copied from row 3000 (speaker "rime-gold/am")

Usage:
    python convert_checkpoint.py /path/to/original-checkpoint
"""

import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ── settings ──────────────────────────────────────────────────────────
EXTENDED_VOCAB_SIZE = 4096
SPEAKER_SRC_IDX = 3000  # rime-gold/am in original
SPEAKER_DST_IDX = 4000  # copy destination
DST_DIR = "tmp/vocab-4096-checkpoint"
EMB_KEY = "talker.model.codec_embedding.weight"


def main(src_dir: str):
    os.makedirs(DST_DIR, exist_ok=True)

    # ── 1. copy everything except the two files we'll rewrite ─────────
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(DST_DIR, item)
        if item in ("config.json", "model.safetensors"):
            continue  # we'll write these ourselves
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
    print(f"copied auxiliary files → {DST_DIR}")

    # ── 2. patch config.json ──────────────────────────────────────────
    with open(os.path.join(src_dir, "config.json")) as f:
        config = json.load(f)

    old_vocab = config["talker_config"]["vocab_size"]
    config["talker_config"]["extended_vocab_size"] = EXTENDED_VOCAB_SIZE
    print(f"config: talker_config.vocab_size={old_vocab}  "
          f"extended_vocab_size={EXTENDED_VOCAB_SIZE}")

    with open(os.path.join(DST_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"wrote {DST_DIR}/config.json")

    # ── 3. resize codec_embedding in safetensors ──────────────────────
    sf_path = os.path.join(src_dir, "model.safetensors")
    tensors = {}
    with safe_open(sf_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    old_emb = tensors[EMB_KEY]
    assert old_emb.shape == (old_vocab, config["talker_config"]["hidden_size"]), (
        f"expected ({old_vocab}, {config['talker_config']['hidden_size']}), "
        f"got {old_emb.shape}"
    )
    print(f"original {EMB_KEY}: {old_emb.shape}  dtype={old_emb.dtype}")

    # build the new embedding: old rows kept, new rows zeroed
    # (zeros mark open slots – actual embeddings are pre-computed elsewhere)
    hidden_size = old_emb.shape[1]
    new_emb = torch.zeros(EXTENDED_VOCAB_SIZE, hidden_size, dtype=old_emb.dtype)
    new_emb[:old_vocab] = old_emb

    # copy speaker "rime-gold/am" from 3000 → 4000
    new_emb[SPEAKER_DST_IDX] = old_emb[SPEAKER_SRC_IDX]
    tensors[EMB_KEY] = new_emb
    print(f"resized  {EMB_KEY}: {new_emb.shape}  "
          f"(row {SPEAKER_SRC_IDX} → {SPEAKER_DST_IDX})")

    save_file(tensors, os.path.join(DST_DIR, "model.safetensors"))
    print(f"wrote {DST_DIR}/model.safetensors")

    # ── 4. verify ─────────────────────────────────────────────────────
    with safe_open(os.path.join(DST_DIR, "model.safetensors"), framework="pt") as f:
        check = f.get_tensor(EMB_KEY)
    assert check.shape == (EXTENDED_VOCAB_SIZE, hidden_size)
    assert torch.equal(check[SPEAKER_SRC_IDX], check[SPEAKER_DST_IDX])
    print("✓ verification passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: python {sys.argv[0]} /path/to/original-checkpoint")
        sys.exit(1)
    main(sys.argv[1])
