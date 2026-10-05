# Factuality & Hallucination Reduction in Large Vision-Language Models

## Core Idea
Current LVLMs (e.g., LLaVA) frequently hallucinate objects or spatial relationships that are not present in the image.

## Proposed Approach
Attention-guided decoding mechanism with dynamic visual feedback loop that grounds text tokens back to specific pixel regions during autoregressive generation.

## Key Metrics/Benchmarks
- **POPE** (Polling-based Object Hallucination Evaluation)
- **MME** (Multi-modal Benchmark)
- **MS COCO** dataset

## Novelty
- First attention-guided decoding with dynamic pixel-region grounding for hallucination reduction in LVLMs
- Real-time visual attention feedback during autoregressive generation
- Object-level consistency enforcement via attention mask projection

## Project Structure
```
factuality-hallucination-reduction/
├── data/              # COCO dataset pipeline
├── coco/              # COCO annotations/images (download separately)
├── src/
│   ├── model/         # LLaVA-inspired LVLM with attention-guided decoding
│   ├── decoding/      # Attention-guided token generation
│   ├── evaluation/    # POPE, MME benchmarks
│   └── train/         # Training scripts
├── results/           # Generated outputs, metrics
├── paper/             # LaTeX conference paper
├── figures/           # Plots and visualizations
└── run.py             # Main entry point
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
python run.py --mode train --epochs 3
# or for evaluation:
python run.py --mode eval --benchmark pope
```