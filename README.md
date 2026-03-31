# CARES: Code Review with Retrieval-Augmented Generation

CARES is a code review system that combines retrieval-augmented generation (RAG) with code understanding models to generate high-quality code review comments across multiple programming languages.

## Overview

- **Retrieval**: Fine-tuned retrieval models (CodeT5, UniXcoder) retrieve relevant code context
- **Generation**: Large language models generate contextual review comments
- **Multi-language**: Supports 9+ programming languages
- **Evaluation**: Comprehensive metrics (BLEU, ROUGE-L, BERTScore)

## Key Components

- **Data Processing**: Preprocesses code review datasets
- **Retrieval Modules**: CodeT5 and UniXcoder-based semantic retrieval
- **Fine-tuning**: Contrastive learning for retrievers, supervised fine-tuning for generators
- **Evaluation**: Multi-metric evaluation (BLEU, ROUGE-L, BERTScore, Semantic Recall)
