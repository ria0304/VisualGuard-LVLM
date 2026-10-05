import json
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from PIL import Image


class MMEvaluator:
    """MME (Multi-modal Benchmark) evaluation.

    MME contains 14 sub-tasks across 4 main categories:
    - Perception: 12 tasks (color, texture, quantity, etc.)
    - Reasoning: 7 tasks (common sense, inference, etc.)
    - Knowledge: 8 tasks (object attributes, etc.)
    - Other: additional tasks

    This evaluator provides a minimal but functional implementation
    for conference paper reporting.
    """

    SUBTASKS = [
        # Perception
        "color", "texture", "size", "quantity", "position",
        # Reasoning
        "common_sense", "inference", "deduction",
        # Knowledge
        "attribute", "action", "exist",
    ]

    def __init__(
        self,
        question_file: str,
        image_dir: str,
        model_generator,
        model_name: str = "lvlm",
    ):
        self.question_file = question_file
        self.image_dir = image_dir
        self.model_generator = model_generator
        self.model_name = model_name

        with open(question_file, "r") as f:
            self.data = json.load(f)

        # Group by subtask
        self.by_subtask: Dict[str, List[Dict]] = {s: [] for s in self.SUBTASKS}
        for q in self.data:
            subtask = q.get("subtask", "other")
            if subtask in self.by_subtask:
                self.by_subtask[subtask].append(q)

        print(f"MME loaded: {sum(len(v) for v in self.by_subtask.values())} questions across {len(self.by_subtask)} subtasks")

    def _load_image(self, image_path: str) -> Image.Image:
        full_path = (
            path.join(self.image_dir, image_path)
            if not path.isabs(image_path)
            else image_path
        )
        return Image.open(full_path).convert("RGB")

    def evaluate_subtask(self, subtask_name: str) -> Dict[str, float]:
        """Evaluate a single MME subtask."""

        questions = self.by_subtask.get(subtask_name, [])
        if not questions:
            return {
                f"{self.model_name}/{subtask_name}/accuracy": 0.0,
                f"{self.model_name}/{subtask_name}/total": 0.0,
            }

        correct = 0
        total = len(questions)

        for q in questions:
            image_path = q["image"]
            question = q["question"]
            answer = q.get("answer", "").strip().lower()

            try:
                predicted = self.model_generator(
                    image_path=image_path,
                    question=question,
                )
            except Exception:
                predicted = ""

            predicted_lower = predicted.strip().lower()

            # Simple exact match for demo
            # In practice, use more sophisticated answer normalization
            if answer and predicted_lower == answer:
                correct += 1
            # Also try contains
            elif answer and answer in predicted_lower:
                correct += 1

        acc = correct / total * 100 if total > 0 else 0.0
        return {
            f"{self.model_name}/{subtask_name}/accuracy": acc,
            f"{self.model_name}/{subtask_name}/total": float(total),
        }

    def full_eval(self) -> Dict[str, float]:
        """Run full MME evaluation across all subtasks."""

        results = {}

        # Report per-subtask accuracy
        for subtask in self.SUBTASKS:
            subtask_results = self.evaluate_subtask(subtask)
            results.update(subtask_results)

        # Compute overall average (excluding empty subtasks)
        valid_accs = [
            results.get(f"{self.model_name}/{s}/accuracy", 0)
            for s in self.SUBTASKS
            if f"{self.model_name}/{s}/total" in results
            and results.get(f"{self.model_name}/{s}/total", 0) > 0
        ]
        overall_avg = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0

        results["summary/mme_overall_accuracy"] = overall_avg
        results["summary/mme_total_questions"] = sum(
            results.get(f"{self.model_name}/{s}/total", 0) for s in self.SUBTASKS
        )

        # Print summary
        print(f"MME Overall Accuracy: {overall_avg:.2f}%")
        for subtask in self.SUBTASKS:
            acc = results.get(f"{self.model_name}/{subtask}/accuracy", 0)
            total = results.get(f"{self.model_name}/{subtask}/total", 0)
            if total > 0:
                print(f"  {subtask:20s}: {acc:.1f}% ({int(total)} Qs)")

        return results

    def save_results(self, results: Dict[str, float], output_file: str):
        """Save evaluation results to JSON."""
        os.makedirs(path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"MME results saved to {output_file}")
        return results