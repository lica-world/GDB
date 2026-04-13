"""Category benchmarks: category-1, category-2.

Data contract: ``samples.csv`` in the task directory under
``benchmarks/category/<TaskDirName>/`` with columns
``sample_id``, ``prompt``, ``image_path``, ``expected_output``.
``image_path`` is resolved against ``dataset_root``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.utils.data_helpers import build_vision_input, load_csv_samples

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers — category normalisation & matching
# ---------------------------------------------------------------------------


def _normalize_category(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower().replace("-", " "))


def _category_match(pred: str, gt: str) -> bool:
    if pred == gt:
        return True
    if pred.rstrip("s") == gt.rstrip("s"):
        return True
    return False


def _parse_predictions(raw_text: str) -> List[str]:
    """Extract up to 5 category predictions (one per line) from model output."""
    lines = [
        _normalize_category(
            line.strip().lower().lstrip("0123456789.-) ").strip()
        )
        for line in raw_text.strip().splitlines()
        if line.strip()
    ]
    return [ln for ln in lines if ln][:5]


# ---------------------------------------------------------------------------
# Shared helpers — classification metrics
# ---------------------------------------------------------------------------


def _accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions) if predictions else 0.0


def _top_k_accuracy(
    predictions: List[List[str]],
    ground_truth: List[str],
    k: int = 5,
) -> float:
    correct = sum(
        1 for preds, gt in zip(predictions, ground_truth) if gt in preds[:k]
    )
    return correct / len(predictions) if predictions else 0.0


def _macro_f1(predictions: List[str], ground_truths: List[str]) -> float:
    gt_classes = sorted(set(ground_truths))
    if not gt_classes:
        return 0.0
    f1_sum = 0.0
    for c in gt_classes:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g == c)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == c and g != c)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != c and g == c)
        denom = 2 * tp + fp + fn
        f1_sum += (2 * tp / denom) if denom > 0 else 0.0
    return f1_sum / len(gt_classes)


# ===================================================================
# category-1  Fine-Grained Category Classification
# ===================================================================


@benchmark
class FineGrainedCategoryClassification(BaseBenchmark):
    """category-1 — Classify rendered layouts into one of 500 fine-grained categories."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="category-1",
        name="Fine-Grained Category Classification",
        task_type=TaskType.UNDERSTANDING,
        domain="category",
        data_subpath="category/CategoryClassification",
        description="Classify a rendered layout into one of 500 fine-grained categories",
        input_spec="Rendered layout image",
        output_spec="Up to five predicted category labels (one per line)",
        metrics=["top1_accuracy", "top5_accuracy", "macro_f1"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def parse_model_output(self, output):
        return _parse_predictions(output.text)

    def evaluate(self, predictions, ground_truth):
        gts = [_normalize_category(str(g)) for g in ground_truth]
        gt_label_set = set(gts)

        def _resolve(p: str) -> str:
            if p in gt_label_set:
                return p
            for gt_label in gt_label_set:
                if _category_match(p, gt_label):
                    return gt_label
            return p

        top5_lists = [
            [_resolve(p) for p in (preds if isinstance(preds, list) else [preds])]
            for preds in predictions
        ]
        preds = [t5[0] if t5 else "" for t5 in top5_lists]

        return {
            "top1_accuracy": _accuracy(preds, gts),
            "top5_accuracy": _top_k_accuracy(top5_lists, gts, k=5),
            "macro_f1": _macro_f1(preds, gts),
        }


# ===================================================================
# category-2  User Intent Prediction
# ===================================================================


@benchmark
class UserIntentPrediction(BaseBenchmark):
    """category-2 — Predict the designer intent that motivated the layout (free text)."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="category-2",
        name="User Intent Prediction",
        task_type=TaskType.UNDERSTANDING,
        domain="category",
        data_subpath="category/UserIntentPrediction",
        description="Predict the user intent that motivated the design",
        input_spec="Rendered layout image",
        output_spec="Free-text user intent",
        metrics=["bertscore_f1", "semantic_cosine_similarity"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        return load_csv_samples(data_dir, n=n, dataset_root=dataset_root)

    def build_model_input(self, sample, *, modality=None):
        return build_vision_input(sample, modality=modality)

    def evaluate(self, predictions, ground_truth):
        preds = [str(p).strip() for p in predictions]
        refs = [str(g).strip() for g in ground_truth]
        scores: Dict[str, float] = {}

        try:
            from bert_score import score as bert_score_fn

            _P, _R, F1 = bert_score_fn(preds, refs, lang="en", verbose=False)
            scores["bertscore_f1"] = float(F1.mean())
        except ImportError:
            pass

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(model_id)
            embed_model = AutoModel.from_pretrained(model_id).to(device).eval()

            all_texts = preds + refs
            all_embs: list = []
            for start in range(0, len(all_texts), 8):
                batch = all_texts[start : start + 8]
                enc = tok(
                    batch, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    hidden = embed_model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embs.append(pooled)
            embs = torch.nn.functional.normalize(torch.cat(all_embs, dim=0), dim=-1)
            pred_embs = embs[: len(preds)]
            ref_embs = embs[len(preds) :]
            cosines = (pred_embs * ref_embs).sum(dim=-1)
            scores["semantic_cosine_similarity"] = float(cosines.mean())

            del embed_model
            torch.cuda.empty_cache()
        except ImportError:
            logger.warning(
                "semantic_cosine_similarity: torch/transformers not installed — "
                "install lica-bench[svg-metrics] for this metric."
            )

        return scores


