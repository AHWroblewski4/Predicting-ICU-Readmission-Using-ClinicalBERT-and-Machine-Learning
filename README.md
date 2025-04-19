# Improving Patient Outcomes through Predictive Analytics: Predicting ICU Readmission Using NLP and AI Models

**Author:** Andrea Wroblewski  
**Course:** Artificial Intelligence in Healthcare  
**Institution:** Michigan Technological University  
**Date:** April 2024

---

## YouTube Presentation

[https://youtu.be/Eku01n0-044](https://youtu.be/Eku01n0-044)

---

## Project Objective

This project aims to build a predictive analytics model to identify ICU patients at risk of readmission within 30 days. The model combines both structured EHR data and unstructured clinical notes using machine learning (Random Forest, XGBoost) and NLP techniques (ClinicalBERT).

---

## Datasets

- **MIMIC-IV v2.2** – Structured electronic health record data  
  🔗 https://physionet.org/content/mimiciv/2.2/

- **MIMIC-IV-Note v2.2** – Deidentified free-text clinical discharge notes  
  🔗 https://physionet.org/content/mimic-iv-note/2.2/

> Note: Dataset access requires credentialed approval via PhysioNet.

---

## Data Files Included

These preprocessed files are included in the repository:

- `structured_data.pkl` – Cleaned and engineered structured EHR features
- `bert_embeddings.pkl` – ClinicalBERT-generated embeddings from discharge summaries

---

## Models Used

- **Random Forest Classifier**
- **XGBoost Classifier**
- **ClinicalBERT Embeddings** (from discharge summaries)
- **SMOTE** (Synthetic Minority Over-sampling Technique for class balancing)

---

## Results

- Combining structured data + ClinicalBERT text embeddings significantly improved performance.
- **Best Model:** Narrowed XGBoost on combined data.
- **AUROC:** 0.98 after SMOTE and hyperparameter tuning.

---

## Key Insights

- Discharge summaries contain clinical cues that structured data may miss.
- ClinicalBERT effectively captures domain-specific language patterns.
- SMOTE improves recall for the minority (readmitted) class.

---

## Related Work

This project is grounded in recent research that highlights the value of combining structured clinical data with natural language processing (NLP) techniques to improve ICU readmission prediction.

### Literature Summary

- **Alsentzer et al. (2019)** introduced ClinicalBERT, a domain-specific BERT model trained on clinical text from MIMIC-III. It demonstrated improved performance on readmission prediction tasks using discharge summaries.  
  🔗 https://arxiv.org/abs/1904.03323

- **Chiu et al. (2024)** combined BERTopic and LSTM to semantically analyze discharge summaries and significantly improve readmission prediction accuracy.  
  🔗 https://doi.org/10.3390/jcm13185503

- **González-Nóvoa et al. (2023)** used optimized XGBoost with Bayesian tuning and explainable AI techniques, achieving an AUROC of 0.92 for ICU readmission prediction.  
  🔗 https://doi.org/10.3390/ijerph20043455

- **Lu, Nguyen, & Dou (2021)** developed MedText, an enhanced model that integrates medical knowledge graphs from UMLS and outperformed ClinicalBERT in predictive tasks.  
  🔗 https://doi.org/10.1145/3404835.3463062

- **Orangi-Fard et al. (2022)** showed that machine learning combined with NLP applied to discharge summaries yielded superior results compared to models using structured data alone.  
  🔗 https://doi.org/10.3390/informatics9010010

- **Rojas et al. (2018)** conducted a large-scale study using gradient-boosted machines (GBM) on structured EHR data and found them to outperform traditional logistic regression.  
  🔗 https://doi.org/10.1513/AnnalsATS.201710-787OC

- **Sheetrit et al. (2023)** conducted a multimodal evaluation of ICU readmission predictors and found that discharge notes outperformed all other modalities for unplanned readmissions. Their work emphasized the importance of NLP in clinical decision-making and validated the predictive value of ClinicalBERT-like models.  
  🔗 https://doi.org/10.1038/s41598-023-42372-y

These studies support this project's approach to building a hybrid model that integrates ClinicalBERT embeddings with structured data to enhance predictive accuracy and clinical relevance.

---

## Dependencies

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

---

## Installation

You can install all dependencies with:

```bash
pip install -r requirements.txt
