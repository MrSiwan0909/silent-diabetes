# Silent Diabetes

**Track:** Health

**Team:** Siwan Ahn, Shrey Rupareliya

> **This project is a prototype and is not medical advice.**

## The problem

About 64% of adults with diabetes in the UAE don't know they have it (IDF Diabetes Atlas). Undiagnosed type 2 diabetes can silently damage the eyes, kidneys and heart for years, but many people never get the HbA1c blood test that would find it.

## What we built

A one-minute screening tool for a **pharmacist in Dubai**. The customer answers 11 no-blood-test questions (age, sex, height and weight for BMI, blood pressure, cholesterol, heart disease, stroke, general health, exercise, walking difficulty, smoking). The app shows **Low risk** (green) or **Higher risk: please see a doctor for an HbA1c blood test** (orange), with the top reasons in English or Arabic, and rejects unrealistic heights and weights.

## What we added

- **A real baseline:** we rebuilt the American Diabetes Association (ADA) risk checklist from the data (5 of its 7 questions) and compared at the same workload.
- **A deliberate threshold:** missing a diabetic is worse than a false alarm, so we chose the threshold on the validation set to catch 85% of diabetics.
- **A fairness check:** the model missed about half of diabetics under 40. We fixed it with age-group thresholds and measured the cost.
- **Honesty checks:** test set used once, overfitting check, prediabetes label check, failure analysis.
- **Extra analyses** (chosen on validation, test reported once): "AI or extra questions?" and a UAE stress test.

## How it works

We used the CDC Diabetes Health Indicators dataset (253,680 US adults, BRFSS 2015), split 60/20/20 into train, validation and test. We trained **logistic regression** with balanced class weights: one weight per question, so every result is explainable. Gradient boosting was only slightly better (ROC-AUC 0.820 vs 0.815). The threshold (0.402) catches 85% of diabetics on validation. The Gradio app uses the same weights (`model.json`) and matched the notebook for all 50,736 test people.

## Results

Held-out test set: 50,736 people, 7,069 with diabetes.

| | ADA checklist | Our model | App version (age-group thresholds) |
|---|---|---|---|
| Diabetics caught (recall) | 81.2% | 84.9% | 88.3% |
| Missed | 1,331 | 1,065 | 829 |
| False alarms | 18,450 | 16,145 | 20,066 |
| % flagged | 47.7% | 43.7% | 51.8% |

- **Our model beats the ADA checklist on both counts:** 266 fewer diabetics missed and 2,305 fewer false alarms (ROC-AUC 0.815 vs 0.757).
- **Fairness:** with one threshold, recall for under-40s was 48.4%. Age-group thresholds raised it to 88.0%, catching 236 more diabetics overall for 3,921 more false alarms (about 17 extra blood tests per extra diabetic). The app uses this fairer version.
- **AI vs extra questions:** with only the ADA's 5 questions, our model still missed 1,209 vs 1,331. About 28% of the gain comes from better weights and 72% from the extra questions.
- **UAE stress test (simulation):** if diabetics are 2.5 BMI units leaner, as is common in South Asian populations, recall drops from 88.3% to 84.0% (300 more missed). A BMI adjustment recovers them for 2,608 more false alarms; real data is needed before changing the app.
- **Where it fails:** missed diabetics look healthy (e.g. a man aged 25-29, BMI 20, "excellent" health, possibly type 1). False alarms often have real risk factors and may be prediabetic or undiagnosed.

## What we'd do next

Main limitation: the data is self-reported, from **US adults in 2015, not UAE residents**. Next: UAE data where people answer our questions *and* take an HbA1c test, adding family history and ethnic background, re-checking fairness by nationality, and a pharmacy pilot.

## Sources

- CDC Diabetes Health Indicators dataset, UCI id 891: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- IDF Diabetes Atlas, UAE: https://diabetesatlas.org/data-by-location/country/united-arab-emirates/
- ADA Type 2 Diabetes Risk Test: https://diabetes.org/diabetes-risk-test
- WHO Expert Consultation, Asian BMI cut-offs, *The Lancet* 2004: https://doi.org/10.1016/S0140-6736(03)15268-3
- Libraries: scikit-learn, pandas, NumPy, matplotlib, Gradio, ucimlrepo
- AI tool used: Claude Code (Anthropic), for planning, code, explanations and drafting text. All results come from our own notebook runs.
- Full list: [Sources in README.md](README.md#sources)
