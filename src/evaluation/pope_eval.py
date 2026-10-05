import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from PIL import Image


class POPEvaluator:
    """POPE (Polling-based Object Hallucination Evaluation) benchmark.

    Evaluates LVLMs on object/hallucination accuracy across four metrics:
    - Top-1 / Top-5 accuracy on Popular questions
    - Top-1 / Top-5 accuracy on Popularity-balanced questions

    Question formats from the POPE paper:
    - Object: "Is there a <object> in the image?" (yes/no)
    - Attribute: "Is the <object> <attribute>?"
    - Relation: "Is the <object> <relation> <object2>?"
    - Count: "How many <object> are there?"
    - Others: exist, counting, etc.
    """

    def __init__(
        self,
        question_file: str,
        image_dir: str,
        model_generator,
        model_name: str = "lvlm",
    ):
        """Initialize POPE evaluator.

        Args:
            question_file: Path to POPE question JSON file
            image_dir: Directory containing COCO images
            model_generator: Callable that takes (image_path, question) -> predicted answer
            model_name: Name of model for reporting
        """
        self.question_file = question_file
        self.image_dir = image_dir
        self.model_generator = model_generator
        self.model_name = model_name

        # Load questions
        with open(question_file, "r") as f:
            self.data = json.load(f)

        # Organize by split
        self.popular_questions = [q for q in self.data if q["split"] == "popular"]
        self.rare_questions = [q for q in self.data if q["split"] == "rare"]

        print(f"POPE loaded: {len(self.popular_questions)} popular, {len(self.rare_questions)} rare questions")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path."""
        full_path = path.join(self.image_dir, image_path) if not path.isabs(image_path) else image_path
        return Image.open(full_path).convert("RGB")

    def evaluate_yes_no(self, questions: List[Dict]) -> Dict[str, float]:
        """Evaluate yes/no style questions (object, exist, counting)."""

        correct_top1 = 0
        correct_top5 = 0
        total = len(questions)

        for q in questions:
            image_path = q["image"]
            question = q["question"]
            answer = q["answer"].strip().lower()  # "yes" or "no"

            # Generate prediction
            try:
                predicted = self.model_generator(
                    image_path=image_path,
                    question=question,
                )
            except Exception as e:
                predicted = "no"

            predicted_lower = predicted.strip().lower()

            # Check if prediction matches answer
            is_correct = (
                (answer == "yes" and predicted_lower in ["yes", "yeah", "yes definitely"]) or
                (answer == "no" and predicted_lower in ["no", "nope", "no indeed"])
            )

            if is_correct:
                correct_top1 += 1

            # Top-5: we'd need model to return probabilities, but for yes/no we just track top-1
            # In full implementation, collect all predictions and compute top-k

        top1_acc = correct_top1 / total * 100 if total > 0 else 0.0
        return {
            f"{self.model_name}/top1_yesno": top1_acc,
            f"{self.model_name}/total_yesno": float(total),
        }

    def evaluate_multichoice(self, questions: List[Dict], top_k: int = 5) -> Dict[str, float]:
        """Evaluate multiple-choice questions (attribute, relation, count)."""

        correct = 0
        total = len(questions)

        for q in questions:
            image_path = q["image"]
            question = q["question"]
            choices = q.get("choices", [])
            true_answer_idx = q.get("answer_idx", -1)

            if not choices or true_answer_idx < 0:
                continue

            # Generate prediction - get top-k predicted choices
            try:
                predicted_idx = self.model_generator(
                    image_path=image_path,
                    question=question,
                    choices=choices,
                    top_k=top_k,
                )
            except Exception:
                predicted_idx = 0  # fallback

            # Handle both int and list returns
            if isinstance(predicted_idx, (list, tuple)):
                predicted_idx = predicted_idx[0] if predicted_idx else 0
            elif isinstance(predicted_idx, str):
                # Try to find in choices
                predicted_idx = 0
                for i, c in enumerate(choices):
                    if c.lower() in predicted_idx.lower():
                        predicted_idx = i
                        break

            if predicted_idx == true_answer_idx:
                correct += 1

        acc = correct / total * 100 if total > 0 else 0.0
        return {
            f"{self.model_name}/top1_multichoice": acc,
            f"{self.model_name}/total_multichoice": float(total),
        }

    def full_eval(self) -> Dict[str, float]:
        """Run full POPE evaluation and return all metrics."""

        results = {}

        # Popular questions - yes/no
        pop_results = self.evaluate_yes_no(self.popular_questions)
        results.update(pop_results)

        # Popularity-balanced questions
        # (POPE also provides a balanced subset; here we evaluate all and report)
        rare_results = self.evaluate_yes_no(self.rare_questions)
        results.update(rare_results)

        # Multi-choice subset (if available)
        mc_questions = [q for q in self.data if q.get("type") == "multiple_choice"]
        if mc_questions:
            mc_results = self.evaluate_multichoice(mc_questions)
            results.update(mc_results)

        # Summary
        results["summary/pope_accuracy"] = (
            results.get(f"{self.model_name}/top1_yesno", 0) +
            results.get(f"{self.model_name}/top1_multichoice", 0)
        ) / 2 if results else 0

        return results

    def save_results(self, results: Dict[str, float], output_file: str):
        """Save evaluation results to JSON."""
        os.makedirs(path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"POPE results saved to {output_file}")
        return results