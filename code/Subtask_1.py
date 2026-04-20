#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DimABSA2026 TrackA Subtask1 (DimASR) — Multi-lingual / Multi-domain mixing trainer

Paradigm Shift: Label Distribution Learning (LDL)
- Converts Continuous Regression to Probability Distribution Classification over 17 bins (1.0 to 9.0).
- Uses Gaussian Soft-Labeling to capture subjective rating variance.
- Computes Expected Value for final scoring.
- Retains physical two-stage isolation and Aspect Token Mean Pooling.
"""

import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from scipy.stats import pearsonr


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_jsonl(url: str, timeout: int = 60) -> List[dict]:
    data = []
    # 智能判断：如果是网址就下载，如果是本地路径就直接读
    if url.startswith("http://") or url.startswith("https://"):
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        text = r.text
    else:
        if not os.path.exists(url):
            # 兜底：如果传入的完整路径找不到，尝试只在当前目录下找同名文件
            basename = os.path.basename(url)
            if os.path.exists(basename):
                url = basename
            else:
                raise FileNotFoundError(f"Local file not found: {url}")
        with open(url, "r", encoding="utf-8") as f:
            text = f.read()

    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        data.append(json.loads(line))
    return data

def safe_bool(x: str) -> bool:
    if isinstance(x, bool): return x
    x = (x or "").strip().lower()
    return x in {"1", "true", "yes", "y", "on"}

def parse_targets(s: str) -> List[Tuple[str, str]]:
    items = []
    s = (s or "").strip()
    if not s: return items
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if "_" not in part: raise ValueError(f"Bad target '{part}'.")
        lang, domain = part.split("_", 1)
        items.append((lang.strip(), domain.strip()))
    return items


# --------------------------
# Data conversion
# --------------------------

def jsonl_to_df(data: List[dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame(columns=["ID", "Text", "Aspect", "Valence", "Arousal"])
    obj0 = data[0]
    if "Quadruplet" in obj0:
        df = pd.json_normalize(data, "Quadruplet", ["ID", "Text"])
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df = df.drop(columns=[c for c in ["VA", "Category", "Opinion"] if c in df.columns])
        return df.drop_duplicates(subset=["ID", "Aspect"], keep="first")
    if "Triplet" in obj0:
        df = pd.json_normalize(data, "Triplet", ["ID", "Text"])
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df = df.drop(columns=[c for c in ["VA", "Opinion"] if c in df.columns])
        return df.drop_duplicates(subset=["ID", "Aspect"], keep="first")
    if "Aspect_VA" in obj0:
        df = pd.json_normalize(data, "Aspect_VA", ["ID", "Text"])
        if df.columns[0] != "Aspect":
            df = df.rename(columns={df.columns[0]: "Aspect"})
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df = df.drop(columns=[c for c in ["VA"] if c in df.columns])
        return df.drop_duplicates(subset=["ID", "Aspect"], keep="first")
    if "Aspect" in obj0 and isinstance(obj0.get("Aspect"), (str, type(None))):
        df = pd.DataFrame(data)
        for c in ["ID", "Text", "Aspect"]:
            if c not in df.columns: raise KeyError(f"Missing {c}")
        if "Valence" not in df.columns: df["Valence"] = np.nan
        if "Arousal" not in df.columns: df["Arousal"] = np.nan
        return df[["ID", "Text", "Aspect", "Valence", "Arousal"]]
    raise ValueError(f"Unsupported JSONL schema")


# --------------------------
# Dataset (LDL Soft Labeling)
# --------------------------

class VADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 256,
        add_tags: bool = True,
        num_bins: int = 17,  # 1.0, 1.5, ... 9.0
        sigma: float = 0.75  # 高斯分布方差
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_tags = add_tags
        
        # LDL 专属参数设置
        self.num_bins = num_bins
        self.sigma = sigma
        self.anchors = np.linspace(1.0, 9.0, num_bins, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["Text"])
        aspect = str(row["Aspect"])
        lang = str(row.get("Lang", ""))
        domain = str(row.get("Domain", ""))

        text_in = f"[LANG={lang}][DOM={domain}] {text}" if self.add_tags else text

        enc = self.tokenizer(
            text_in,
            aspect,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        # 构建 Aspect Mask 精确定位
        aspect_mask = torch.zeros(self.max_len, dtype=torch.float32)
        has_aspect = False
        try:
            seq_ids = enc.sequence_ids(0)
            if seq_ids is not None:
                for i, sid in enumerate(seq_ids):
                    if sid == 1:
                        aspect_mask[i] = 1.0
                        has_aspect = True
        except Exception: pass
        if not has_aspect: aspect_mask[0] = 1.0
        item["aspect_mask"] = aspect_mask

        # 获取真实标签并转化为概率分布 (Label Distribution)
        v = row.get("Valence", np.nan)
        a = row.get("Arousal", np.nan)
        if not (pd.isna(v) or pd.isna(a)):
            v_f, a_f = float(v), float(a)
            
            # 生成高斯软标签
            v_dist = np.exp(-((self.anchors - v_f)**2) / (2 * self.sigma**2))
            v_dist /= v_dist.sum()
            
            a_dist = np.exp(-((self.anchors - a_f)**2) / (2 * self.sigma**2))
            a_dist /= a_dist.sum()
            
            item["labels_v"] = torch.tensor(v_dist, dtype=torch.float32)
            item["labels_a"] = torch.tensor(a_dist, dtype=torch.float32)
            
            # 保留原始绝对数值，用于算 PCC/RMSE 辅助 loss
            item["raw_v"] = torch.tensor(v_f, dtype=torch.float32)
            item["raw_a"] = torch.tensor(a_f, dtype=torch.float32)

        item["meta"] = {
            "ID": str(row["ID"]),
            "Aspect": str(row["Aspect"]),
            "Lang": lang,
            "Domain": domain,
            "Text": text,
        }
        return item


# --------------------------
# Model (Classification-based)
# --------------------------

class SingleTaskLDLRegressor(nn.Module):
    def __init__(self, backbone_name: str, num_bins: int = 17, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # 输出不再是 1，而是 17 个类别的 logits
        self.head = nn.Linear(hidden, num_bins)

    def forward(self, input_ids, attention_mask, aspect_mask=None, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        
        if aspect_mask is not None:
            mask_expanded = aspect_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(out.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_feat = sum_embeddings / sum_mask
        else:
            pooled_feat = out.last_hidden_state[:, 0]
            
        x = self.dropout(pooled_feat)
        logits = self.head(x)  # [Batch_size, 17]
        return logits


# --------------------------
# Training & Mixing logic
# --------------------------

DEFAULT_LANGS = ["zho", "eng", "jpn", "rus", "ukr", "tat"]
DEFAULT_DOMAINS = ["restaurant", "laptop", "hotel", "finance"]
SAME_DOMAIN_PRIORITY = {"restaurant", "hotel", "laptop"}

def build_train_url(base_url: str, subtask: str, lang: str, domain: str) -> str:
    if base_url.startswith("http"):
        return f"{base_url}/track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"
    return os.path.join(base_url, f"{lang}_{domain}_train_alltasks.jsonl")

def build_split_url(base_url: str, subtask: str, lang: str, domain: str, split: str, task: str) -> str:
    if base_url.startswith("http"):
        return f"{base_url}/track_a/{subtask}/{lang}/{lang}_{domain}_{split}_{task}.jsonl"
    return os.path.join(base_url, f"{lang}_{domain}_{split}_{task}.jsonl")

def load_pair_df(base_url: str, subtask: str, lang: str, domain: str, split: str, task: str,
                 is_train_alltasks: bool = False, timeout: int = 60) -> pd.DataFrame:
    if is_train_alltasks:
        # 1. 首先尝试去抓取官方标准的 alltasks 文件
        url = build_train_url(base_url, subtask, lang, domain)
        try:
            data = fetch_jsonl(url, timeout=timeout)
        except Exception as e:
            # 💡 [新增核心逻辑] 如果没找到，降级寻找对应 task 后缀的训练文件 (如 _train_task1.jsonl)
            fallback_url = build_split_url(base_url, subtask, lang, domain, "train", task)
            try:
                print(f"[INFO] 未找到 alltasks，正在尝试读取备用文件: {fallback_url}")
                data = fetch_jsonl(fallback_url, timeout=timeout)
            except Exception as e2:
                # 如果两种后缀都没找到，抛出清晰的错误提示
                raise Exception(f"训练文件缺失！已尝试以下两个路径均失败: \n1. {url} \n2. {fallback_url}")
    else:
        # 验证集/预测集走正常的读取逻辑
        url = build_split_url(base_url, subtask, lang, domain, split, task)
        data = fetch_jsonl(url, timeout=timeout)

    df = jsonl_to_df(data)
    df["Lang"] = lang
    df["Domain"] = domain
    return df

def make_mix_plan(target_lang: str, target_domain: str, langs: List[str], domains: List[str],
                  same_domain_ratio: float, related_ratio: float, reg_ratio: float) -> Dict[str, float]:
    ratios = {
        "target": max(0.0, 1.0 - same_domain_ratio - related_ratio - reg_ratio),
        "same_domain": max(0.0, same_domain_ratio),
        "related": max(0.0, related_ratio),
        "reg": max(0.0, reg_ratio),
    }
    rem = 1.0 - (ratios["same_domain"] + ratios["related"] + ratios["reg"])
    ratios["target"] = max(0.0, rem)
    s = sum(ratios.values())
    if s <= 0: return {"target": 1.0}
    for k in ratios: ratios[k] /= s
    return ratios

def build_sample_weights(train_df: pd.DataFrame, target_lang: str, target_domain: str, ratios: Dict[str, float],
                         same_domain_langs: List[str]) -> np.ndarray:
    n = len(train_df)
    w = np.zeros(n, dtype=np.float64)
    is_target = (train_df["Lang"] == target_lang) & (train_df["Domain"] == target_domain)
    is_same_domain = (train_df["Domain"] == target_domain) & (train_df["Lang"].isin(same_domain_langs)) & (~is_target)
    is_reg = ~(is_target | is_same_domain)
    
    eps = 1e-9
    buckets = [
        ("target", is_target.values, ratios.get("target", 0.0)),
        ("same_domain", is_same_domain.values, ratios.get("same_domain", 0.0)),
        ("reg", is_reg.values, ratios.get("reg", 0.0) + ratios.get("related", 0.0)),
    ]
    for _, mask, mass in buckets:
        cnt = int(mask.sum())
        if cnt > 0 and mass > 0: w[mask] = mass / (cnt + eps)
    if w.sum() <= 0: w[:] = 1.0 / n if n > 0 else 1.0
    return w


def pcc_loss(pred: torch.Tensor, gold: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.size(0) < 2: return (pred * 0.0).sum()
    pred_var = pred - pred.mean()
    gold_var = gold - gold.mean()
    covariance = (pred_var * gold_var).sum()
    denominator = torch.sqrt((pred_var ** 2).sum() + eps) * torch.sqrt((gold_var ** 2).sum() + eps)
    return 1.0 - (covariance / denominator)

def rmse_loss_fn(pred: torch.Tensor, gold: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - gold) ** 2) + eps)

def eval_single_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc = 0.0 if (np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8) else float(pearsonr(y_true, y_pred)[0])
    return {"rmse": rmse, "pcc": pcc}

@torch.no_grad()
def run_eval_single(model: nn.Module, dev_loader: DataLoader, device: str, task_name: str, num_bins: int = 17) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    anchors = torch.tensor(np.linspace(1.0, 9.0, num_bins, dtype=np.float32), device=device)
    
    for batch in tqdm(dev_loader, desc=f"eval_{task_name}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        aspect_mask = batch["aspect_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None: token_type_ids = token_type_ids.to(device)
            
        labels = batch["raw_v" if task_name == "valence" else "raw_a"].detach().cpu().numpy()
        logits = model(input_ids, attention_mask, aspect_mask, token_type_ids)
        
        # 期望值解码 (Expected Value Decoding)
        probs = torch.softmax(logits, dim=-1)
        expected_values = torch.sum(probs * anchors, dim=-1).detach().cpu().numpy()
        
        ys.append(labels)
        ps.append(expected_values)
        
    return eval_single_metrics(np.concatenate(ys), np.concatenate(ps))

def train_single_task(
    model: nn.Module, train_dataset: Dataset, dev_loader: Optional[DataLoader], device: str,
    task_name: str, args: argparse.Namespace, weights: np.ndarray, is_target_mask: np.ndarray,
    target_lang: str, target_domain: str
) -> Tuple[nn.Module, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else math.ceil(len(train_dataset) / args.train_batch_size)
    total_steps = args.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    anchors = torch.tensor(np.linspace(1.0, 9.0, args.num_bins, dtype=np.float32), device=device)
    
    best_score, best_state = -float("inf"), None

    for ep in range(1, args.epochs + 1):
        w_ep = weights.copy()
        if args.anneal_target and args.epochs > 1:
            mult = 1.0 + ((ep - 1) / (args.epochs - 1)) * (args.anneal_target_mult - 1.0)
            w_ep = w_ep * (1.0 + is_target_mask * (mult - 1.0))

        sampler = WeightedRandomSampler(weights=torch.tensor(w_ep, dtype=torch.double), num_samples=steps_per_epoch * args.train_batch_size, replacement=True)
        epoch_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler, num_workers=0)

        model.train()
        pbar = tqdm(epoch_loader, desc=f"[{target_lang}_{target_domain}|{task_name.upper()}] ep{ep}", leave=False)
        running = 0.0
        for step, batch in enumerate(pbar, start=1):
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            aspect_mask = batch["aspect_mask"].to(device)
            token_type_ids = batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None
                
            target_soft = batch["labels_v"].to(device) if task_name == "valence" else batch["labels_a"].to(device)
            raw_gold = batch["raw_v"].to(device) if task_name == "valence" else batch["raw_a"].to(device)

            logits = model(input_ids, attention_mask, aspect_mask, token_type_ids)

            # 1. 散度损失 (KL-Divergence)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss_kl = kl_loss_fn(log_probs, target_soft)

            # 2. 期望指标损失 (EV Metric Loss)
            probs = torch.exp(log_probs)
            expected_values = torch.sum(probs * anchors, dim=-1)
            loss_pcc = pcc_loss(expected_values, raw_gold)
            loss_rmse = rmse_loss_fn(expected_values, raw_gold)
            
            # 黄金组合：主治分布，辅修排序
            loss = loss_kl + 0.5 * loss_pcc + 0.1 * loss_rmse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running += float(loss.item())
            if step % args.log_every == 0:
                pbar.set_postfix(loss=running / args.log_every)
                running = 0.0

        if dev_loader is not None:
            met = run_eval_single(model, dev_loader, device, task_name, num_bins=args.num_bins)
            score = met['pcc'] - met['rmse']
            print(f"[{target_lang}_{target_domain} | {task_name.upper()}] dev: RMSE={met['rmse']:.4f} PCC={met['pcc']:.4f} Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None: model.load_state_dict(best_state)
    return model, best_score

@torch.no_grad()
def get_predictions_dual(
    model_v: nn.Module, model_a: nn.Module, pred_loader: DataLoader, device: str,
    num_bins: int, clip_min: float = 1.0, clip_max: float = 9.0,
) -> List[dict]:
    anchors = torch.tensor(np.linspace(1.0, 9.0, num_bins, dtype=np.float32), device=device)
    
    model_v.to(device); model_v.eval()
    v_preds_all, metas_all = [], []
    for batch in tqdm(pred_loader, desc="predict_valence", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        aspect_mask = batch["aspect_mask"].to(device)
        token_type_ids = batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None
            
        logits = model_v(input_ids, attention_mask, aspect_mask, token_type_ids)
        expected_v = torch.sum(torch.softmax(logits, dim=-1) * anchors, dim=-1)
        v_preds_all.extend(expected_v.detach().cpu().numpy().flatten().tolist())
        
        metas = batch["meta"]
        if isinstance(metas, dict):
            B = len(metas["ID"])
            for i in range(B): metas_all.append({k: metas[k][i] for k in metas})
        else: metas_all.extend(metas)
            
    model_v.cpu(); torch.cuda.empty_cache()

    model_a.to(device); model_a.eval()
    a_preds_all = []
    for batch in tqdm(pred_loader, desc="predict_arousal", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        aspect_mask = batch["aspect_mask"].to(device)
        token_type_ids = batch.get("token_type_ids").to(device) if batch.get("token_type_ids") is not None else None
            
        logits = model_a(input_ids, attention_mask, aspect_mask, token_type_ids)
        expected_a = torch.sum(torch.softmax(logits, dim=-1) * anchors, dim=-1)
        a_preds_all.extend(expected_a.detach().cpu().numpy().flatten().tolist())
        
    model_a.cpu(); torch.cuda.empty_cache()

    outputs = []
    for i, meta in enumerate(metas_all):
        # LDL 直接输出 1~9 的绝对分值，不需要再反归一化了！
        v = float(np.clip(v_preds_all[i], clip_min, clip_max))
        a = float(np.clip(a_preds_all[i], clip_min, clip_max))
        outputs.append({"ID": meta["ID"], "Aspect": meta["Aspect"], "VA": f"{v:.4f}#{a:.4f}"})
    return outputs

def save_jsonl_starter_grouped(rows: List[dict], out_path: str):
    def extract_num(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else -1
    recs = []
    for r in rows:
        if not r.get("ID") or not r.get("Aspect") or not r.get("VA") or "#" not in str(r.get("VA")): continue
        try: v, a = map(float, str(r["VA"]).split("#", 1))
        except Exception: continue
        recs.append({"ID": str(r["ID"]), "Aspect": str(r["Aspect"]), "Valence": v, "Arousal": a})

    df = pd.DataFrame(recs, columns=["ID", "Aspect", "Valence", "Arousal"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(df) == 0:
        with open(out_path, "w", encoding="utf-8") as f: pass
        return

    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {"ID": gid, "Aspect_VA": [{"Aspect": row["Aspect"], "VA": f"{row['Valence']:.4f}#{row['Arousal']:.4f}"} for _, row in gdf.iterrows()]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_jsonl(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_jsonl_grouped(rows: List[dict], path: str):
    from collections import defaultdict
    buf = defaultdict(list)
    for r in rows:
        _id = r.get("ID")
        asp = r.get("Aspect")
        va = r.get("VA")
        if _id is None or asp is None or va is None:
            continue
        buf[str(_id)].append({"Aspect": str(asp), "VA": str(va)})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _id, arr in buf.items():
            f.write(json.dumps({"ID": _id, "Aspect_VA": arr}, ensure_ascii=False) + "\n")


# --------------------------
# Main pipeline per target
# --------------------------

def run_one_target(args, target_lang: str, target_domain: str):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    train_parts = []
    for lg in args.langs:
        for dm in args.domains:
            try:
                df = load_pair_df(args.base_url, args.subtask, lg, dm, args.dev_split, args.task, True, args.timeout)
                if len(df) > 0: train_parts.append(df)
            except Exception as e:
                if args.strict_download: raise
                print(f"[WARN] Skip {lg}_{dm} due to error: {e}")
                continue

    if not train_parts: raise RuntimeError("No training data loaded. Check network or local paths.")
    train_df_all = pd.concat(train_parts, axis=0, ignore_index=True)

    dev_df = None
    if safe_bool(args.use_dev):
        try: dev_df = load_pair_df(args.base_url, args.subtask, target_lang, target_domain, args.dev_split, args.task, False, args.timeout)
        except Exception: pass

    pred_df = load_pair_df(args.base_url, args.subtask, target_lang, target_domain, args.pred_split, args.task, False, args.timeout)

    same_domain_langs = [lg for lg in args.langs if lg != target_lang]

    ratios = make_mix_plan(target_lang, target_domain, args.langs, args.domains, args.same_domain_ratio, args.related_ratio, args.reg_ratio)
    weights = build_sample_weights(train_df_all, target_lang, target_domain, ratios, same_domain_langs)
    is_target_mask = ((train_df_all["Lang"] == target_lang) & (train_df_all["Domain"] == target_domain)).values.astype(np.float64)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset = VADataset(train_df_all, tokenizer, args.max_len, safe_bool(args.add_tags), num_bins=args.num_bins, sigma=args.ldl_sigma)
    dev_loader = DataLoader(VADataset(dev_df, tokenizer, args.max_len, safe_bool(args.add_tags), num_bins=args.num_bins, sigma=args.ldl_sigma), batch_size=args.eval_batch_size, shuffle=False) if dev_df is not None else None
    pred_loader = DataLoader(VADataset(pred_df, tokenizer, args.max_len, safe_bool(args.add_tags), num_bins=args.num_bins, sigma=args.ldl_sigma), batch_size=args.eval_batch_size, shuffle=False)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path_v = os.path.join(ckpt_dir, f"{target_lang}_{target_domain}_valence.pt")
    ckpt_path_a = os.path.join(ckpt_dir, f"{target_lang}_{target_domain}_arousal.pt")

    model_v = SingleTaskLDLRegressor(args.model_name, num_bins=args.num_bins, dropout=args.dropout)
    model_a = SingleTaskLDLRegressor(args.model_name, num_bins=args.num_bins, dropout=args.dropout)

    best_score_v, best_score_a = None, None

    if getattr(args, "mode", "train") == "predict":
        model_v.load_state_dict(torch.load(ckpt_path_v, map_location="cpu"))
        model_a.load_state_dict(torch.load(ckpt_path_a, map_location="cpu"))
    else:
        print(f"\n========== TRAINING VALENCE (LDL) ==========")
        model_v.to(device)
        model_v, best_score_v = train_single_task(model_v, train_dataset, dev_loader, device, "valence", args, weights, is_target_mask, target_lang, target_domain)
        torch.save(model_v.state_dict(), ckpt_path_v)
        model_v.cpu(); torch.cuda.empty_cache()

        print(f"\n========== TRAINING AROUSAL (LDL) ==========")
        model_a.to(device)
        model_a, best_score_a = train_single_task(model_a, train_dataset, dev_loader, device, "arousal", args, weights, is_target_mask, target_lang, target_domain)
        torch.save(model_a.state_dict(), ckpt_path_a)
        model_a.cpu(); torch.cuda.empty_cache()

    preds = get_predictions_dual(model_v, model_a, pred_loader, device, num_bins=args.num_bins, clip_min=args.clip_min, clip_max=args.clip_max)

    out_file = os.path.join(args.output_dir, "subtask_1", f"pred_{target_lang}_{target_domain}_{args.pred_split}_{args.task}.jsonl")
    if args.output_format == "flat":
        save_jsonl(preds, out_file)
    elif args.output_format == "grouped":
        save_jsonl_grouped(preds, out_file)
    else:
        save_jsonl_starter_grouped(preds, out_file)
        
    log = {
        "target": f"{target_lang}_{target_domain}",
        "pred_split": args.pred_split,
        "model_name": args.model_name,
        "train_size_total": int(len(train_df_all)),
        "best_dev_score_v": float(best_score_v) if best_score_v else None,
        "best_dev_score_a": float(best_score_a) if best_score_a else None,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"runlog_{target_lang}_{target_domain}.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved dual-model LDL predictions: {out_file}")
    return out_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", type=str)
    ap.add_argument("--subtask", default="subtask_1", type=str)
    ap.add_argument("--base_url", default="https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset", type=str)
    ap.add_argument("--mode", default="train", choices=["train", "predict"], type=str)
    ap.add_argument("--target", default="", type=str)
    ap.add_argument("--targets", default="", type=str)
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS), type=str)
    ap.add_argument("--domains", default=",".join(DEFAULT_DOMAINS), type=str)
    ap.add_argument("--dev_split", default="dev", type=str)
    ap.add_argument("--pred_split", default="dev", type=str)
    ap.add_argument("--use_dev", default="true", type=str)

    ap.add_argument("--model_name", default="bert-base-multilingual-cased", type=str)
    ap.add_argument("--max_len", default=256, type=int)
    ap.add_argument("--dropout", default=0.1, type=float)
    ap.add_argument("--train_batch_size", default=16, type=int)
    ap.add_argument("--eval_batch_size", default=32, type=int)
    ap.add_argument("--lr", default=2e-5, type=float)
    ap.add_argument("--weight_decay", default=0.01, type=float)
    ap.add_argument("--epochs", default=3, type=int)
    ap.add_argument("--warmup_ratio", default=0.06, type=float)
    ap.add_argument("--max_grad_norm", default=1.0, type=float)

    # 补回了上一版误删的 timeout，解决了你的报错
    ap.add_argument("--timeout", default=60, type=int, help="Download timeout")

    # LDL Specific Params
    ap.add_argument("--num_bins", default=17, type=int, help="Number of classification bins (e.g. 17 for 1.0, 1.5 ... 9.0)")
    ap.add_argument("--ldl_sigma", default=0.75, type=float, help="Variance for the Gaussian Soft Labeling")

    ap.add_argument("--log_every", default=50, type=int)
    ap.add_argument("--steps_per_epoch", default=0, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--add_tags", default="true", type=str)
    ap.add_argument("--clip_min", default=1.0, type=float)
    ap.add_argument("--clip_max", default=9.0, type=float)
    ap.add_argument("--same_domain_ratio", default=0.35, type=float)
    ap.add_argument("--related_ratio", default=0.20, type=float)
    ap.add_argument("--reg_ratio", default=0.05, type=float)
    
    ap.add_argument("--use_embedding_related", default="true", type=str)
    ap.add_argument("--finance_related_topk", default=3, type=int)
    ap.add_argument("--related_topk", default=2, type=int)
    ap.add_argument("--embed_pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--embed_max_len", default=256, type=int)
    ap.add_argument("--embed_max_per_group", default=300, type=int)
    ap.add_argument("--embed_batch_size", default=16, type=int)

    ap.add_argument("--anneal_target", action="store_true")
    ap.add_argument("--anneal_target_mult", default=2.0, type=float)
    ap.add_argument("--output_format", default="starter", type=str)
    ap.add_argument("--output_dir", default="outputs", type=str)
    ap.add_argument("--strict_download", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    args.langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    args.domains = [x.strip() for x in args.domains.split(",") if x.strip()]

    targets = parse_targets(args.target.strip()) if args.target.strip() else parse_targets(args.targets)
    if not targets: targets = [(lg, dm) for lg in args.langs for dm in args.domains]

    for lg, dm in targets: run_one_target(args, lg, dm)

if __name__ == "__main__":
    main()