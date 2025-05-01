# Financial NLP - Sentiment, Summarization & QA

This repository contains experiments on using Small Language Models (SLMs) like SmolLM2 to perform:
- Financial Sentiment Analysis
- Financial Document Summarization
- Financial QA using tabular/textual reasoning

We evaluate fine-tuning strategies including LoRA, sequential transfer learning, and multi-task learning via X-LoRA.

## Project Structure

```
224UProject/
├── data/                      # Preprocessed datasets
│   ├── finsentiment/
│   ├── earningscallsum/
│   └── tatqa_finqa/
├── models/                    # Base and fine-tuned checkpoints
│   ├── base/
│   ├── lora/
│   └── xlora/
├── experiments/               # Hypothesis-driven experiments
│   ├── h1_sentiment_transfer/
│   ├── h2_hierarchical_summarization/
│   └── h3_xlora_multitask/
├── evaluation/                # Metrics, plotting, hallucination analysis
├── utils/                     # Config parser, logging, seeds
├── configs/                   # YAML configs per experiment
├── scripts/                   # Shell/Python launchers
├── results/                   # Evaluation outputs and logs
├── requirements.txt           # Project dependencies
└── README.md
```

