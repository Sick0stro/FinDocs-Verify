# FinDocs-Verify Benchmark Report

_Generated 2026-08-26T15:37:51+00:00 — standard Markdown + JSON output (no rendered PNG tables)._

## Model
- Architecture: DistilBERT (fine-tuned on invoice mismatch detection)
- Model path: `D:\FinDocs-Verify\models\invoice_classifier_epoch10`
- Training corpus: 632 documents (162 mismatches, 470 clean)
- Test set: 63 held-out documents (16 mismatch, 47 clean)
- Device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU)

## Metrics (hf-evaluate)

| Metric | Value |
|--------|-------|
| Precision | 0.6154 |
| Recall | 1.0000 |
| F1-Score | 0.7619 |
| Accuracy | 0.8413 |

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| `clean` | 1.0000 | 0.7872 | 0.8810 | 47 |
| `mismatch` | 0.6154 | 1.0000 | 0.7619 | 16 |
| `macro avg` | 0.8077 | 0.8936 | 0.8214 | 63 |
| `weighted avg` | 0.9023 | 0.8413 | 0.8507 | 63 |

## Confusion Matrix

| Actual \ Predicted | CLEAN | MISMATCH |
|---|---|---|
| **CLEAN** | 37 | 10 |
| **MISMATCH** | 0 | 16 |

(True=positive=mismatch, Predicted=positive=mismatch)

- True Negatives (clean, correctly classified): 37
- True Positives (mismatch, correctly detected): 16
- False Negatives (mismatch, missed): 0
- False Positives (clean, flagged): 10

## Error Analysis
- False positives: 10
- False negatives: 0
- Confidence range: 0.5320 – 0.9910
- Avg confidence: 0.8241

## Sample Predictions (10 examples)

| # | True | Pred | Conf | Text preview |
|---|------|------|------|--------------|
| 1 | CLEAN | CLEAN | 0.9880 | ﺮﺗ ﻱﻭﺎﻀﻴﺑ ﻊﻄﻗ 3 ﻡﺎﻤﺣ ﺵﺮﻔﻣ ﻢﻗﺎﻃ 70*130 ﻱﺪﻨﻫ ﻒﺸﻨﻣ 90*150 ﻱﺪﻨﻫ ﻒﺸﻨﻣ |
| 2 | CLEAN | CLEAN | 0.9807 | ﻢﺟ 500 ﻥﺎﻬﺒﺤﻟﺎﺑ ﺪﻣﺎﺣ ﺓﻮﻬﻗ |
| 3 | CLEAN | CLEAN | 0.9873 | ﺔﻟﻭﺍ ﺔﺟﺭﺩ ﻲﺣﺮﻣ ﻞﻴﺒﺠﻧﺯ |
| 4 | CLEAN | MISMATCH | 0.5951 |  |
| 5 | CLEAN | CLEAN | 0.9902 | ﻲﺣﺮﻣ ﻞﻴﺒﺠﻧﺯ ﻝﺎﺘﺴﻳﺮﻛ ﺢﻠﻣ |
| 6 | MISMATCH | MISMATCH | 0.5951 |  |
| 7 | MISMATCH | MISMATCH | 0.5951 |  |
| 8 | CLEAN | CLEAN | 0.9903 | ﻲﺣﺮﻣ ﺕﻮﺣ ﻥﻮﻤﻛ ﺭﺎﺣ ﺮﻤﺣﺍ ﻞﻔﻠﻓ ﻲﺴﻧﻮﺗ ﻞﺑﺎﺗ |
| 9 | CLEAN | MISMATCH | 0.5951 |  |
| 10 | CLEAN | CLEAN | 0.9906 | SY ﺏﺍﻮﻛﻻﺎﺑ ﺔﻌﺑﺮﻣ ﺓﺭﻭﺭﺎﻗ ﻢﻗﺎﻃ 36-3 ﺐﺸﺧ ﺪﻳ ﻞﻴﺘﺳﺍ ﺝﺮﻜﺑ SH100-050/24 ﺐﺸﺧ ﺪﻳ 304 ﺝﺮﻜﺑ ﺕﺍﻭ 2200 ﺮﺘﻟ 1.7 ﻞﻴﺘﺳ ﻲﺋﺎﺑﺮﻬﻛ ﺀﺎﻣ ﻥﺎﺨﺳ |
