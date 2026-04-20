#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DimABSA 2026 Track A - Subtask2/3 LLM pipeline (parallel low-resource optimized)

Key ideas implemented:
- Joint multilingual SFT on parallel rus/ukr/tat data (shared sentiment structure)
- Optional multi-view prompts (target text + 1-2 translations) WITHOUT leaking labels
- Stricter output format instruction to reduce noisy generations
- Robust parsing + validation (must appear in text; legal labels for task3)
- VA calibration by snapping to nearest seen (valence, arousal) pair from training
- Self-consistency decoding: generate multiple candidates and pick best by heuristics

Designed to run in Colab.

Example usage:
  python dimabsa_task23_parallel_optimized.py \
    --task task2 --domain restaurant \
    --train_files rus_restaurant_train_alltasks.jsonl ukr_restaurant_train_alltasks.jsonl tat_restaurant_train_alltasks.jsonl \
    --train_langs rus ukr tat \
    --dev_file ukr_restaurant_dev_task2.jsonl \
    --model_id unsloth/Qwen3-4B-Instruct-2507-bnb-4bit \
    --output_dir ./lora_task2_restaurant \
    --do_train --do_predict
"""

import argparse
import json
import math
import os
import random
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def _all_spans(text: str, sub: str) -> list[tuple[int,int]]:
    spans = []
    if not sub:
        return spans
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(sub)))
        start = idx + 1
    return spans


def _min_char_distance(text: str, a: str, b: str) -> int:
    """Minimum character distance between any occurrence of a and b in text.
    Returns a large number if either is not found."""
    a_sp = _all_spans(text, a)
    b_sp = _all_spans(text, b)
    if not a_sp or not b_sp:
        return 10**9
    best = 10**9
    for (as_, ae) in a_sp:
        for (bs, be) in b_sp:
            if ae < bs:
                d = bs - ae
            elif be < as_:
                d = as_ - be
            else:
                d = 0
            if d < best:
                best = d
    return best


def _is_low_value_opinion(op: str) -> bool:
    """Conservative filter: drop extremely short / non-informative opinions."""
    op = (op or "").strip()
    if len(op) <= 1:
        return True
    if re.fullmatch(r"[\W\d_]+", op, flags=re.UNICODE):
        return True
    return False

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -------------------------
# Label constraints (task3)
# -------------------------
REST_ENTITY = {
    "RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"
}
REST_ATTR = {
    "GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"
}

LAPTOP_ENTITY = {
    "LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU",
    "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES",
    "BATTERY", "GRAPHICS", "HARD_DISK", "MULTIMEDIA_DEVICES", "HARDWARE",
    "SOFTWARE", "OS", "WARRANTY", "SHIPPING", "SUPPORT", "COMPANY"
}
LAPTOP_ATTR = {
    "GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES", "OPERATION_PERFORMANCE",
    "USABILITY", "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"
}

HOTEL_ENTITY = {
    "HOTEL", "ROOMS", "FACILITIES", "ROOM_AMENITIES", "SERVICE", "LOCATION", "FOOD_DRINKS"
}
HOTEL_ATTR = {
    "GENERAL", "PRICE", "COMFORT", "CLEANLINESS", "QUALITY", "DESIGN_FEATURES",
    "STYLE_OPTIONS", "MISCELLANEOUS"
}

FIN_ENTITY = {"MARKET", "COMPANY", "BUSINESS", "PRODUCT"}
FIN_ATTR = {"GENERAL", "SALES", "PROFIT", "AMOUNT", "PRICE", "COST"}

ENTITY_ATTR_MAP = {
    "restaurant": (REST_ENTITY, REST_ATTR),
    "laptop": (LAPTOP_ENTITY, LAPTOP_ATTR),
    "hotel": (HOTEL_ENTITY, HOTEL_ATTR),
    "finance": (FIN_ENTITY, FIN_ATTR),
}


def build_instruction(task: str, domain: str, lang_tag: str, use_multiview: bool) -> str:
    """A stricter instruction to reduce output noise."""

    if task == "task2":
        return (
            "You are given a customer review text.\n"
            f"Text language: {lang_tag}.\n\n"
            "Task: Extract all (Aspect, Opinion, VA) triplets from [Text].\n"
            "- Aspect: an aspect term phrase from the text\n"
            "- Opinion: the opinion expression about that aspect\n"
            "- VA: Valence#Arousal, where Valence in [1,9] (negative->positive), Arousal in [1,9] (calm->excited)\n\n"
            "Output format (STRICT):\n"
            "Return ONLY a comma-separated list of tuples:\n"
            "(Aspect, Opinion, Valence#Arousal)\n"
            "No extra words, no bullet points, no explanations.\n"
            "If there is no triplet, return an empty list: []\n\n"
            + ("You MAY see optional translation views; they are only for understanding.\n\n" if use_multiview else "")
            +"[Text]\n"
        )

    if task == "task3":
        ent, attr = ENTITY_ATTR_MAP[domain]
        ent_str = ", ".join(sorted(ent))
        attr_str = ", ".join(sorted(attr))
        return (
            "You are given a customer review text.\n"
            f"Text language: {lang_tag}.\n\n"
            "Task: Extract all (Aspect, Category, Opinion, VA) quadruplets from [Text].\n"
            "- Aspect: an aspect term phrase from the text\n"
            "- Category: ENTITY#ATTRIBUTE\n"
            "- Opinion: the opinion expression about that aspect\n"
            "- VA: Valence#Arousal, where Valence in [1,9] and Arousal in [1,9]\n\n"
            "Allowed Category labels (STRICT):\n"
            f"- ENTITY in {{{ent_str}}}\n"
            f"- ATTRIBUTE in {{{attr_str}}}\n\n"
            "Output format (STRICT):\n"
            "Return ONLY a comma-separated list of tuples:\n"
            "(Aspect, Category, Opinion, Valence#Arousal)\n"
            "No extra words, no bullet points, no explanations.\n"
            "If there is no quadruplet, return an empty list: []\n\n"
            + ("You MAY see optional translation views; they are only for understanding.\n\n" if use_multiview else "")
            +"[Text]\n"
        )

    raise ValueError("task must be task2 or task3")


@dataclass
class VASnapper:
    """Snap predicted VA to nearest seen pair in training (reduces jitter)."""

    # store as list of (v, a, original_string)
    va_list: List[Tuple[float, float, str]]
    va_set: set

    @staticmethod
    def from_train_items(train_items: Sequence[dict]) -> "VASnapper":
        seen = {}
        for item in train_items:
            for q in item.get("Quadruplet", []):
                va = q.get("VA")
                if not isinstance(va, str) or "#" not in va:
                    continue
                try:
                    v_str, a_str = va.split("#", 1)
                    v = float(v_str)
                    a = float(a_str)
                except Exception:
                    continue
                # keep one canonical string per numeric pair (prefer shorter)
                key = (v, a)
                if key not in seen or len(va) < len(seen[key]):
                    seen[key] = va
        va_list = [(v, a, s) for (v, a), s in seen.items()]
        va_set = set(seen.values())
        return VASnapper(va_list=va_list, va_set=va_set)

    def snap(self, va_str: str) -> str:
        va_str = va_str.strip()
        if va_str in self.va_set:
            return va_str
        # parse
        try:
            v_str, a_str = va_str.split("#", 1)
            v = float(v_str)
            a = float(a_str)
        except Exception:
            # fallback: default neutral-ish
            return "5.0#5.0"

        # clamp
        v = min(9.0, max(1.0, v))
        a = min(9.0, max(1.0, a))

        # nearest
        best = None
        best_d = 1e9
        for tv, ta, ts in self.va_list:
            d = (tv - v) * (tv - v) + (ta - a) * (ta - a)
            if d < best_d:
                best_d = d
                best = ts
        return best if best is not None else f"{v:.1f}#{a:.1f}"




def apply_chat_template_safe(tokenizer, messages, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool | None = None):
    """Compatibility wrapper: some tokenizer implementations don't accept enable_thinking."""
    try:
        if enable_thinking is None:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Fallback for older versions
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
def _assistant_text(decoded: str) -> str:
    """Try to isolate assistant completion from a chat-formatted decode."""
    # common patterns
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>", 1)[1]
    # remove any special tokens at end
    decoded = decoded.strip()
    # some templates include an end token; keep only first chunk
    decoded = decoded.split("<|endoftext|>", 1)[0].strip()
    return decoded


def parse_task2(text: str) -> List[Dict[str, str]]:
    text = text.strip()
    if text == "[]":
        return []
    # be tolerant: find tuples anywhere
    # (Aspect, Opinion, VA)
    pattern = r"\(([^,()]+),\s*([^,()]+),\s*([0-9]+(?:\.[0-9]+)?#[0-9]+(?:\.[0-9]+)?)\)"
    matches = re.findall(pattern, text)
    out = []
    for aspect, opinion, va in matches:
        out.append({"Aspect": aspect.strip(), "Opinion": opinion.strip(), "VA": va.strip()})
    return out


def parse_task3(text: str) -> List[Dict[str, str]]:
    text = text.strip()
    if text == "[]":
        return []
    # (Aspect, Category, Opinion, VA)
    pattern = r"\(([^,()]+),\s*([^,()]+),\s*([^,()]+),\s*([0-9]+(?:\.[0-9]+)?#[0-9]+(?:\.[0-9]+)?)\)"
    matches = re.findall(pattern, text)
    out = []
    for aspect, category, opinion, va in matches:
        out.append({
            "Aspect": aspect.strip(),
            "Category": category.strip(),
            "Opinion": opinion.strip(),
            "VA": va.strip(),
        })
    return out


def is_valid_category(domain: str, cat: str) -> bool:
    if "#" not in cat:
        return False
    ent, attr = cat.split("#", 1)
    ent = ent.strip()
    attr = attr.strip()
    ent_set, attr_set = ENTITY_ATTR_MAP[domain]
    return ent in ent_set and attr in attr_set


def normalize_items(
    task: str,
    domain: str,
    text: str,
    items: List[Dict[str, str]],
    va_snapper: Optional[VASnapper] = None,
    pair_window_chars: int = 0,
    max_aspects_per_opinion: int = 0,
) -> List[Dict[str, str]]:

    """Heuristic cleanup:
    - keep only entries whose aspect/opinion appear in text (exact substring)
    - proximity constraint (optional): aspect/opinion must be within a char window
    - deduplicate
    - category constraint (task3)
    - VA clamp + snap
    - limit fan-out: one opinion cannot attach to too many aspects (optional)
    """

    norm = []
    seen = set()

    tmp = []  # (dist, item)
    for it in items:
        aspect = it.get("Aspect", "").strip()
        opinion = it.get("Opinion", "").strip()
        if not aspect or not opinion:
            continue
        if _is_low_value_opinion(opinion):
            continue
        if aspect not in text or opinion not in text:
            continue

        dist = _min_char_distance(text, aspect, opinion)
        if pair_window_chars and dist > pair_window_chars:
            continue

        va = it.get("VA", "").strip()
        if va_snapper is not None:
            va = va_snapper.snap(va)

        if task == "task2":
            key = (aspect, opinion, va)
            if key in seen:
                continue
            seen.add(key)
            tmp.append((dist, {"Aspect": aspect, "Opinion": opinion, "VA": va}))
        else:
            cat = it.get("Category", "").strip()
            if not is_valid_category(domain, cat):
                continue
            key = (aspect, cat, opinion, va)
            if key in seen:
                continue
            seen.add(key)
            tmp.append((dist, {"Aspect": aspect, "Category": cat, "Opinion": opinion, "VA": va}))

    if max_aspects_per_opinion and max_aspects_per_opinion > 0:
        buckets = {}
        for dist, it in tmp:
            op = it.get("Opinion", "")
            buckets.setdefault(op, []).append((dist, it))
        tmp2 = []
        for op, lst in buckets.items():
            lst.sort(key=lambda x: x[0])
            tmp2.extend(lst[:max_aspects_per_opinion])
        tmp = tmp2

    tmp.sort(key=lambda x: x[0])
    norm = [it for _, it in tmp]
    return norm


def score_candidate(task: str, domain: str, text: str, items: List[Dict[str, str]]) -> float:
    """Simple heuristic score for self-consistency selection."""
    if not items:
        return 0.0
    score = 0.0
    for it in items:
        asp = it.get("Aspect", "")
        opn = it.get("Opinion", "")
        va = it.get("VA", "")
        if asp in text:
            score += 1.0
        if opn in text:
            score += 1.0
        # VA well-formed
        if re.match(r"^\d+(?:\.\d+)?#\d+(?:\.\d+)?$", va or ""):
            try:
                v, a = va.split("#")
                v = float(v)
                a = float(a)
                if 1.0 <= v <= 9.0:
                    score += 0.5
                if 1.0 <= a <= 9.0:
                    score += 0.5
            except Exception:
                pass
        if task == "task3":
            if is_valid_category(domain, it.get("Category", "")):
                score += 0.5
    # Prefer precision: allow a few items, penalize large fan-out (common FP source)
    n = len(items)
    score += 0.2 * min(n, 3)
    if n > 3:
        score -= 0.15 * (n - 3)
    return score


def build_multiview_text(sample_by_lang: Dict[str, dict], target_lang: str) -> str:
    """Construct [Text] with optional translation views.

    We always include the target language text first.
    Additional views are included as plain text only (no labels).
    """
    parts = []
    parts.append(f"(TARGET {target_lang}) {sample_by_lang[target_lang]['Text']}")
    other = [l for l in sample_by_lang.keys() if l != target_lang]
    for l in other:
        parts.append(f"(VIEW {l}) {sample_by_lang[l]['Text']}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["task2", "task3"], required=True)
    ap.add_argument("--domain", choices=["restaurant", "laptop", "hotel", "finance"], required=True)

    ap.add_argument("--train_files", nargs="+", default=[])
    ap.add_argument("--train_langs", nargs="+", default=[])

    ap.add_argument("--dev_file", default=None, help="Path to dev jsonl for prediction")
    ap.add_argument("--dev_lang", default=None, help="dev language tag (e.g., rus/ukr/tat)")

    ap.add_argument("--model_id", default="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit")
    ap.add_argument("--max_seq_length", type=int, default=1024)

    ap.add_argument("--output_dir", default="./lora")
    ap.add_argument("--seed", type=int, default=42)

    # training
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # multiview
    ap.add_argument("--use_multiview", action="store_true")
    ap.add_argument("--multiview_ratio", type=float, default=0.25,
                    help="Fraction of training examples that include translation views")

    # prediction
    ap.add_argument("--do_predict", action="store_true")
    ap.add_argument("--num_candidates", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=1024)

    # prediction quality / robustness (no retraining)
    ap.add_argument("--predict_stream", action="store_true", help="Stream-write jsonl during prediction to avoid losing progress")
    ap.add_argument("--predict_log_every", type=int, default=10, help="Log progress every N dev samples")
    ap.add_argument("--max_triplets_per_sample", type=int, default=30, help="Hard cap on extracted items to curb FP explosions")
    ap.add_argument("--pair_window_chars", type=int, default=120, help="Max char gap allowed between aspect and opinion (0 disables). Helps cut FP from wrong attachments")
    ap.add_argument("--max_aspects_per_opinion", type=int, default=3, help="Limit how many aspects can share the same opinion (0 disables). Helps cut list-style FP")

    args = ap.parse_args()

    seed_everything(args.seed)

    if (not args.do_train) and (not args.do_predict):
        print("[INFO] Nothing to do: set --do_train and/or --do_predict.")
        return

    seed_everything(args.seed)

    # Lazy imports for Colab
    from datasets import Dataset
    from unsloth import FastLanguageModel

    ensure_dir(args.output_dir)

    # ------------------
    # Load training data
    # ------------------
    if args.do_train:
        if not args.train_files or not args.train_langs or len(args.train_files) != len(args.train_langs):
            raise ValueError("Provide --train_files and --train_langs with the same length")

        train_items_all = []
        by_lang = {}
        for path, lang in zip(args.train_files, args.train_langs):
            items = load_jsonl(path)
            for it in items:
                it["__lang"] = lang
            by_lang[lang] = items
            train_items_all.extend(items)

        # VA snapper from all training
        va_snapper = VASnapper.from_train_items(train_items_all)

        # ------------------
        # Load model/tokenizer
        # ------------------
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )

        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )

        # Build training examples
        def make_messages(user_prompt: str, answer: str) -> List[Dict[str, str]]:
            return [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
            ]

        # For multiview, we need aligned samples by ID
        aligned_by_id = {}
        if args.use_multiview:
            # assume parallel: IDs match across all langs
            ids = [it["ID"] for it in next(iter(by_lang.values()))]
            for i, _id in enumerate(ids):
                aligned_by_id[_id] = {lang: by_lang[lang][i] for lang in by_lang.keys()}

        def build_answer(item: dict) -> str:
            quads = item.get("Quadruplet", []) or []
            if args.task == "task2":
                tuples = []
                for q in quads:
                    tuples.append(f"({q['Aspect']}, {q['Opinion']}, {q['VA']})")
                return ", ".join(tuples) if tuples else "[]"
            else:
                tuples = []
                for q in quads:
                    tuples.append(f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})")
                return ", ".join(tuples) if tuples else "[]"

        def convert(item: dict) -> Dict[str, str]:
            lang = item["__lang"]
            instruction = build_instruction(args.task, args.domain, lang, args.use_multiview)

            if args.use_multiview and random.random() < args.multiview_ratio:
                # include translation views alongside target text
                mv_text = build_multiview_text(aligned_by_id[item["ID"]], lang)
                user_prompt = instruction + mv_text
            else:
                user_prompt = instruction + item["Text"]

            answer = build_answer(item)
            messages = make_messages(user_prompt, answer)
            text = apply_chat_template_safe(tokenizer, messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        ds = Dataset.from_list(train_items_all)
        train_dataset = ds.map(convert, remove_columns=ds.column_names)

        # ------------------
        # Train
        # ------------------
        from trl import SFTTrainer
        from transformers import TrainingArguments

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=True,
            args=TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                warmup_steps=args.warmup_steps,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                fp16=False,
                bf16=True,
                report_to="none",
                output_dir=args.output_dir,
                save_total_limit=2,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                weight_decay=0.01,
            ),
        )

        trainer.train()
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Save VA snapper set for later prediction
        with open(os.path.join(args.output_dir, "va_pairs.json"), "w", encoding="utf-8") as f:
            json.dump({"va_list": va_snapper.va_list, "va_set": list(va_snapper.va_set)}, f, ensure_ascii=False)

    # ------------------
    # Predict
    # ------------------
    if args.do_predict:
        if not args.dev_file:
            raise ValueError("Provide --dev_file for prediction")
        if not args.dev_lang:
            raise ValueError("Provide --dev_lang (e.g., rus/ukr/tat)")

        # Load VA snapper if available
        va_snapper = None
        va_path = os.path.join(args.output_dir, "va_pairs.json")
        if os.path.exists(va_path):
            with open(va_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            va_snapper = VASnapper(va_list=[tuple(x) for x in obj["va_list"]], va_set=set(obj["va_set"]))

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.output_dir,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )

        dev_items = load_jsonl(args.dev_file)

        def make_user_prompt(text: str) -> str:
            instruction = build_instruction(args.task, args.domain, args.dev_lang, args.use_multiview)
            return instruction + text


        # Prepare output path early so we can stream-write (and you can watch the file grow).
        subtask_folder = "subtask_2" if args.task == "task2" else "subtask_3"
        ensure_dir(subtask_folder)
        out_name = f"pred_{args.dev_lang}_{args.domain}.jsonl"
        jsonl_path = os.path.join(subtask_folder, out_name)

        import torch
        model.eval()

        stream = bool(getattr(args, "predict_stream", False))
        log_every = max(1, int(getattr(args, "predict_log_every", 10)))

        # If streaming, write incrementally; otherwise accumulate in memory then write once.
        fout = open(jsonl_path, "w", encoding="utf-8") if stream else None
        results = [] if not stream else None

        print(f"[INFO] Predicting {len(dev_items)} samples -> {jsonl_path} (stream={stream})")

        for i, sample in enumerate(dev_items):
            user_prompt = make_user_prompt(sample["Text"])
            messages = [{"role": "user", "content": user_prompt}]
            prompt = apply_chat_template_safe(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # Tokenize once per sample (reused across self-consistency candidates).
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            best_items = []
            best_score = -1e9

            with torch.inference_mode():
                for c in range(max(1, args.num_candidates)):
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature if args.num_candidates > 1 else 0.0,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=(args.num_candidates > 1),
                        pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    )

                    decoded = tokenizer.decode(gen[0], skip_special_tokens=False)
                    completion = _assistant_text(decoded)

                    parsed = parse_task2(completion) if args.task == "task2" else parse_task3(completion)
                    cleaned = normalize_items(args.task, args.domain, sample["Text"], parsed, va_snapper=va_snapper,
                                                   pair_window_chars=args.pair_window_chars,
                                                   max_aspects_per_opinion=args.max_aspects_per_opinion)
                    sc = score_candidate(args.task, args.domain, sample["Text"], cleaned)

                    if sc > best_score:
                        best_score = sc
                        best_items = cleaned

            # Hard cap to avoid FP explosions on noisy generations
            max_items = int(getattr(args, "max_triplets_per_sample", 30))
            if max_items > 0 and len(best_items) > max_items:
                best_items = best_items[:max_items]

            key = "Triplet" if args.task == "task2" else "Quadruplet"
            out_obj = {
                "ID": sample.get("ID", f"sample_{i}"),
                "Text": sample["Text"],
                key: best_items,
            }

            if stream:
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                results.append(out_obj)

            if (i + 1) % log_every == 0 or (i + 1) == len(dev_items):
                print(f"[predict] {i+1}/{len(dev_items)} done")

        if fout is not None:
            fout.close()

        if not stream:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Zip as required
        zip_name = f"{subtask_folder}.zip"
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files_in_dir in os.walk(subtask_folder):
                for fn in files_in_dir:
                    full_path = os.path.join(root, fn)
                    zf_path = os.path.relpath(full_path, ".")
                    zf.write(full_path, zf_path)

        print("Saved:", jsonl_path)
        print("Zipped:", zip_name)

if __name__ == "__main__":
    main()
