# 🗞️ NLP Methods for News Articles

This repository explores natural language processing (NLP) pipelines for analyzing news articles. The project includes text preprocessing, topic modeling, and language model–based classification using modern Python NLP tools.

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

### 🔹 `pipeline1_main.ipynb`
- Loads and cleans raw text data
- Tokenizes and vectorizes text
- Applies classical NLP methods like TF-IDF and LDA
- Visualizes topic distributions and clusters

### 🔹 `pipeline2_main.ipynb`
- Fine-tunes and evaluates transformer-based models (e.g., BERT)
- Performs zero-shot or few-shot classification
- Includes plotting of prediction confidence and topic distribution



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



