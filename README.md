# Improving Patient Outcomes through Predictive Analytics: Predicting ICU Readmission Using NLP and AI Models

**Author:** Andrea Wroblewski  
**Course:** Artificial Intelligence in Healthcare
**Institution:** Michigan Technological University  
**Date:** April 2024

## YouTube Presentation
https://youtu.be/Eku01n0-044

##  Project Objective

This project aims to build a predictive analytics model to identify ICU patients at risk of readmission within 30 days. The model combines both structured EHR data and unstructured clinical notes using machine learning (Random Forest, XGBoost) and NLP techniques (ClinicalBERT).

## Datasets

**MIMIC-IV v2.2**: Structured electronic health record data  
  [https://physionet.org/content/mimiciv/2.2/](https://physionet.org/content/mimiciv/2.2/)

**MIMIC-IV-Note v2.2**: Deidentified free-text clinical discharge notes  
  [https://physionet.org/content/mimic-iv-note/2.2/](https://physionet.org/content/mimic-iv-note/2.2/)

> Note: Dataset access requires credentialed approval via PhysioNet.
## Data Files Included

These preprocessed files are included in the repository:
- `structured_data.pkl` – Cleaned and engineered structured EHR features
- `bert_embeddings.pkl` – ClinicalBERT-generated embeddings from discharge summaries

## Models Used

- **Random Forest Classifier**
- **XGBoost Classifier**
- **ClinicalBERT Embeddings** (from discharge summaries)
- **SMOTE** (Synthetic Minority Over-sampling Technique for class balancing)

## Results

- Combining structured data + ClinicalBERT text embeddings significantly improved performance.
- Best Model: Narrowed XGBoost on combined data.
- AUROC: 0.98 after SMOTE and hyperparameter tuning.

##  Key Insights

- Discharge summaries contain clinical cues that structured data may miss.
- ClinicalBERT effectively captures domain-specific language patterns.
- SMOTE improves recall for the minority (readmitted) class.

---
##  Dependencies

To run this project, the following Python libraries are required:

- `pandas` – Data manipulation
- `numpy` – Numerical computations
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – Machine learning models and metrics
- `xgboost` – Gradient boosting model
- `imblearn` – SMOTE for class imbalance handling
- `torch` – PyTorch framework used for ClinicalBERT
- `transformers` – Hugging Face library for loading ClinicalBERT
- `tqdm` – Progress bar for embedding loop

### Installation

You can install all dependencies with:

```bash
pip install 
