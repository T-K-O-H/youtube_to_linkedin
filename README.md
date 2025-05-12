---
title: YouTube to LinkedIn Post Converter
emoji: 🎥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# YouTube to LinkedIn Post Converter

This application converts YouTube video content into professional LinkedIn posts using AI-powered content enhancement and RAGAS evaluation metrics.

## Features

- 🎥 YouTube transcript extraction
- ✨ AI-powered content enhancement
- 📊 RAGAS evaluation metrics
- 🔄 Automatic content verification
- 📱 Professional LinkedIn post formatting

## Models

The application uses three embedding models for comparison:
- OpenAI's text-embedding-3-small
- Base MPNet (sentence-transformers/all-mpnet-base-v2)
- Fine-tuned MPNet (Shipmaster1/finetuned_mpnet_matryoshka_mnr)

## Evaluation Metrics

- Faithfulness: How well answers align with context
- Answer Relevancy: How relevant answers are to questions
- Context Recall: How well context covers required information
- Context Precision: How focused and precise the context is

## Usage

1. Enter a YouTube URL
2. Wait for transcript extraction
3. Review enhanced content
4. Get your professional LinkedIn post

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## License

MIT License 