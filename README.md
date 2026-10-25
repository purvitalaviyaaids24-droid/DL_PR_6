# 🎬 Transformer-Based Sentiment Analysis and Attention Visualization

## 📌 Project Overview

This project explores **Transformer-based Natural Language Processing (NLP)** using pretrained models and the IMDB Movie Reviews dataset.

The main objective is to understand how transformer models process movie reviews, predict sentiment, and use attention mechanisms to identify relationships between tokens in a sequence.

The project works with three publicly available pretrained transformer models and a sample of 200 real movie reviews. It investigates sentiment classification, attention weights, token-level relationships, and the effect of the 512-token sequence limit on long reviews.

The project also demonstrates how attention mechanisms help visualize the internal behavior of transformer models and provide insights into how textual information is processed.

## 🎯 Project Objectives

- Understand the architecture and working principles of Transformer models.
- Load and use three publicly available pretrained models from Hugging Face.
- Perform sentiment analysis on IMDB movie reviews.
- Extract attention weights from transformer models.
- Visualize attention patterns using heatmaps.
- Analyze relationships between tokens using attention scores.
- Investigate how models handle long reviews exceeding the 512-token limit.
- Compare attention behavior across different pretrained models.
- Understand the limitations of attention-based interpretations.

## 🛠️ Technologies and Libraries

The project uses the following Python libraries:

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch (`torch`) | Tensor operations and model inference |
| Hugging Face Transformers | Loading pretrained transformer models |
| Hugging Face Datasets | Loading the IMDB dataset |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Attention heatmaps and visualization |
| Jupyter Notebook | Interactive development environment |

## 📂 Dataset Description

**Dataset:** IMDB Movie Reviews Dataset

**Source:** [Hugging Face IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)

The IMDB dataset contains movie reviews labeled according to their sentiment.

For this project, a reproducible sample of 200 reviews is selected from the test split.

### Dataset Features

| Feature | Description |
|---|---|
| `text` | The movie review text |
| `label` | Sentiment classification label |

### Sentiment Labels

- `0` — Negative sentiment
- `1` — Positive sentiment

### Dataset Loading

The dataset is loaded using the Hugging Face Datasets library.

```python
from datasets import load_dataset

ds = load_dataset("imdb", split="test")
ds = ds.shuffle(seed=42).select(range(200))
```

This code shuffles the test dataset using a fixed random seed and selects 200 reviews.

**Note:** Several selected reviews may exceed the 512-token input limit used in the long-review analysis.

## 🤖 Pretrained Transformer Models

The project uses three publicly available pretrained transformer models.

These models are loaded from the Hugging Face Model Hub and used for inference and attention analysis.

The experiments focus on:

- Loading pretrained model architectures and tokenizers.
- Processing movie reviews into model-compatible inputs.
- Generating sentiment predictions where supported by the selected model.
- Extracting attention weights from transformer layers.
- Visualizing attention patterns between tokens.
- Comparing attention behavior across the three models.

The exact model names should be documented in the notebook or updated in this section to match the models actually used in the experiments.

### Hugging Face Model Loading

The Transformers library provides the `AutoModel` interface for loading pretrained transformer architectures.

For attention extraction, models are loaded using eager attention:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager"
)
```

The eager attention implementation allows attention weights to be returned when supported by the selected model.

## 🧠 Understanding Transformer Attention

Attention is a fundamental mechanism in transformer architectures that allows each token to assign different weights to other tokens in a sequence.

These weights help the model combine contextual information when processing text.

### Key Concepts

**1. Query, Key, and Value**

Self-attention uses query, key, and value representations to calculate relationships between tokens and combine information from the sequence.

**2. Self-Attention**

Each token can attend to other tokens in the input sequence according to the attention mechanism used by the model.

**3. Multi-Head Attention**

Multiple attention heads can learn different patterns of relationships between tokens.

**4. Attention Weights**

Attention weights represent the relative contributions assigned to tokens within an attention operation.

**5. Attention Visualization**

Attention matrices can be visualized as heatmaps to examine patterns between words and tokens.

## 📊 Attention Weight Extraction

To obtain attention weights, the model must be configured to support attention outputs.

The following example demonstrates the general approach:

```python
inputs = tokenizer(
    review,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

model.eval()

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True
    )

attentions = outputs.attentions
```

The returned attention tensors contain attention weights for the model's layers and heads, subject to the selected architecture.

The attention matrices can then be analyzed and visualized to explore token-to-token relationships.

**Important:** Use `attn_implementation="eager"` when loading compatible models if you need attention weights. Recent versions of Transformers may use faster attention implementations that do not return attention weights.

Also, `AutoModel` alone does not necessarily provide sentiment predictions. Use an appropriate sequence-classification model or task-specific classification head when sentiment prediction is required.

## 📈 Attention Visualization

The project uses Matplotlib and Seaborn to visualize attention matrices.

Heatmaps help examine how tokens relate to other tokens within a particular attention head and layer.

Typical visualization steps include:

1. Tokenize a movie review.
2. Run the input through a pretrained transformer model.
3. Extract the attention weights.
4. Select a specific layer and attention head.
5. Create a heatmap of the attention matrix.
6. Compare attention patterns across different reviews or models.

Example visualization code:

```python
import matplotlib.pyplot as plt
import seaborn as sns

attention = attentions[-1][0, 0].cpu().numpy()

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

sns.heatmap(
    attention,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis"
)

plt.title("Transformer Attention Heatmap")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

This example visualizes one attention head from the final transformer layer.

For long reviews, displaying every token may make the heatmap difficult to read. A shorter review or a selected token range can provide a clearer visualization.

## 📏 Long Review Analysis: The 512-Token Limit

Transformer models commonly impose a maximum input sequence length. In this project, the long-review experiment examines a 512-token limit.

Some movie reviews contain more than 512 tokens, so their inputs must be truncated or handled using another long-document strategy.

### Key Observations

- Short reviews can generally fit within the selected input limit.
- Long reviews may lose information when truncated.
- Truncation can remove important sentiment-bearing statements.
- Attention visualizations cover the tokens actually supplied to the model, not text removed during preprocessing.
- The chosen truncation strategy affects which parts of a review are available for analysis.

This experiment demonstrates the practical challenges of applying transformer models to long documents.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.9 or a compatible Python version supported by your installed libraries.
- Jupyter Notebook or JupyterLab.
- An internet connection for the initial download of datasets and pretrained models.

A Hugging Face account is not required for the public models and dataset used in this project.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Replace the placeholders with your actual repository URL and directory name.

### Step 2: Install Dependencies

Run the following command in a terminal or Jupyter environment:

```bash
pip install transformers torch datasets matplotlib seaborn numpy pandas
```

### Step 3: Open Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook in the Jupyter interface.

### Step 4: Run the Notebook

Execute the notebook cells in the appropriate order.

The notebook will load the dataset, initialize the pretrained models, perform inference, extract attention weights, and generate visualizations according to the implemented experiments.

## 🌐 Internet and Model Caching

An internet connection is required the first time the notebook downloads the public dataset and pretrained model files.

After downloading, Hugging Face caches the dataset and model files locally.

Subsequent runs can reuse cached files, provided that the required files remain available and the loading configuration does not require additional downloads.

Offline execution is therefore possible after the necessary resources have been cached successfully.

## 🔁 Reproducibility

The project uses fixed random seeds to improve reproducibility.

Set the seeds near the beginning of the notebook:

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

For consistent inference behavior, place the model in evaluation mode before running it:

```python
model.eval()
```

Use `torch.no_grad()` during inference to avoid unnecessary gradient calculations.

The dataset sample is also shuffled with `seed=42` to make the selection reproducible.

Although these settings improve consistency, exact outputs may still vary across hardware, library versions, and computational environments.

## 📁 Project Structure

A recommended repository structure is:

```text
Transformer-Attention-Analysis/
│
├── README.md
├── requirements.txt
├── Code-5(1).ipynb
└── figures/
    └── attention_heatmaps.png
```

The `figures/` directory is optional and can contain saved attention visualizations generated by the notebook.

The notebook filename can be changed to a more descriptive name before publishing the repository.

### requirements.txt

The following dependencies can be placed in `requirements.txt`:

```text
transformers
torch
datasets
matplotlib
seaborn

Author
PURVI TALAVIYA
