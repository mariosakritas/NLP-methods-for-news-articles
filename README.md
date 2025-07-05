# 🗞️ NLP Methods for News Articles

This repository explores natural language processing (NLP) pipelines for analyzing news articles. The project includes text preprocessing, topic modeling, and language model–based classification using modern Python NLP tools. One 

## 📁 Project Structure

Spring23_DW/
├── notebooks/ # Jupyter notebooks for main pipelines and experiments
│ ├── pipeline1_main.ipynb
│ ├── pipeline2_main.ipynb
│ └── suplementary/, outdated/
├── data/ # Input and intermediate data
├── models/ # Saved models (if any)
├── src/ # Source code (functions, scripts)
├── reports/ # Outputs (charts, metrics, tables)
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md


## 🧪 What the Pipelines Do

### 1️⃣ Classical NLP Pipeline (`pipeline1_main.ipynb`)
This notebook uses traditional NLP techniques for unsupervised topic modeling:
- Loads and cleans raw text data
- TF-IDF vectorization for sparse document representation
- Latent Dirichlet Allocation (LDA) for topic extraction
- Clustering and dimensionality reduction for interpretability
- Emphasis on explainability and topic visualization

### 2️⃣ Modern Transformer-Based Pipeline (`pipeline2_main.ipynb`)
This notebook leverages pre-trained language models for zero-shot or few-shot learning:
- Uses models like `BERT` or `GPT` for semantic understanding
- Fine-tunes on custom-labeled data or uses prompt-based querying
- Supports classification of news articles into predefined categories
- More powerful and accurate on short, complex texts

## 🌐 Web Scraping with Google Search
To populate the dataset, a custom Google Search scraping script was used:
- Targets high-quality news domains (e.g., The Guardian, BBC, etc.)
- Extracts article titles and snippets from search results
- Filters for specific query terms related to news topics
- Outputs are saved to structured CSVs for downstream processing

This allowed for creation of a semi-automated dataset with minimal labeling effort — ideal for bootstrapping topic models and zero-shot classification tasks.



🚀 Running the Project
Open notebooks/pipeline1_main.ipynb to explore classical NLP methods

Open notebooks/pipeline2_main.ipynb to explore transformer-based models

Outputs will be saved in reports/ or models/

🧰 Dependencies
Main Python libraries used:

transformers

scikit-learn

pandas, numpy

matplotlib, seaborn

jupyter

python-dotenv

See requirements.txt for the full list.

📄 License
This project is for educational and research purposes. Please cite appropriately if reused.



