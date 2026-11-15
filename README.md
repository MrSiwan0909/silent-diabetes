# Silent Diabetes

**Track:** Health

**Team:** _(add team member names here)_

> **This project is a prototype and is not medical advice.**

**Links**
- Notebook (opens in Google Colab): https://colab.research.google.com/github/MrSiwan0909/silent-diabetes/blob/main/silent_diabetes.ipynb
- Notebook on GitHub (with all outputs): [silent_diabetes.ipynb](silent_diabetes.ipynb)
- App code: [app.py](app.py) · Model numbers: [model.json](model.json)
- Live app: run the last cell of the notebook to get a public `.gradio.live` link (see [How to run](#how-to-run)).

---

## The problem

The International Diabetes Federation estimates that about 64% of adults with diabetes in the UAE don't know they have it. Undiagnosed type 2 diabetes can quietly damage the eyes, kidneys, nerves and heart for years, yet many people never get the HbA1c blood test that would find it.

## What we built

A one-minute screening tool for a **pharmacist in Dubai** (or a volunteer at a mall health booth).

- **What goes in:** 11 questions anyone can answer without a blood test: age, sex, height and weight (we calculate BMI), high blood pressure, high cholesterol, heart disease, stroke, general health, physical activity, difficulty walking, and smoking.
- **What comes out:** **Low risk** (green) or **Higher risk: please see a doctor for an HbA1c blood test** (orange), with the top reasons in plain words (e.g. "High blood pressure", "BMI of 34"), the next steps, and the disclaimer.
- **How a pharmacist uses it:** open the app on a phone or laptop, ask the questions in English or Arabic, press **Check risk**, and explain the result to the customer. The app rejects unrealistic heights and weights, so a typing mistake can't produce a false result.

## What we added

Beyond training a model, we added:

1. **Our own user:** a UAE pharmacist screening customers in under a minute, with an English/Arabic app built around that.
2. **The American Diabetes Association (ADA) checklist as a baseline:** we rebuilt 5 of the 7 questions of the ADA Type 2 Diabetes Risk Test from the data, and compared fairly at the *same workload* (same share of people flagged). We state clearly that 2 ADA questions (family history and gestational diabetes) aren't in the data.
3. **A deliberate threshold choice:** missing a diabetic is worse than a false alarm, so we chose the threshold on the validation set to catch 85% of diabetics while still flagging fewer people than the ADA checklist.
4. **A fairness check by sex and age:** we found the model missed about half of diabetics under 40, tried one fix (age-group thresholds), measured its cost, and used it in the app.
5. **Honesty checks:** a prediabetes label check (the dataset's documentation is wrong about it), an overfitting and leakage check, a check on test people whose answers never appeared in training, and a failure analysis of real mistakes.
6. **An explainable app:** every result shows the exact reasons behind it.
7. **Extra analyses added after the main results** (all choices made on the validation set, test set reported once):
   - **"Is it the AI, or just more questions?"** We trained the same model on only the ADA's 5 questions to separate the gain from better weights and the gain from extra questions.
   - **A UAE stress test (simulation):** we simulated a population where diabetics have a lower BMI (as is common in South Asian populations), measured how much the app's recall drops, and tested a simple BMI adjustment.
   - **A shorter 9-question version:** we retrained without physical activity and smoking to see if the screening can be faster.

## How it works

**Data.** The CDC Diabetes Health Indicators dataset: 253,680 US adults from the 2015 BRFSS phone survey (UCI dataset id 891). It's downloaded inside the notebook and not stored in this repo.

**Cleaning and checks.**
- No missing values. 13.9% of people have diabetes, so the data is imbalanced.
- **Prediabetes:** the documentation says `Diabetes_binary = 1` means "prediabetes or diabetes", but the count (35,346) exactly matches the diabetes-only count in the 3-class version. So **prediabetes is labelled as "no diabetes"**, and our target means diagnosed diabetes.
- We kept only the 11 questions a pharmacist can ask without a lab test.
- 67% of rows are exact duplicates of another row (only 11 simple answers), so we also report results on test people whose answer pattern never appeared in training.

**Split.** 60% train / 20% validation / 20% test, stratified. All choices were made on train and validation. **The test set was used once, at the end.**

**Models.**
- **Logistic regression** (used in the app): one weight per question, so every result can be explained exactly. `class_weight='balanced'` makes a missed diabetic count about 6 times more than a false alarm. Because of this weighting, the output is a risk score, not a probability, so the app never shows a percentage.
- **Gradient boosting** (`HistGradientBoostingClassifier`) for comparison. It was only slightly better (ROC-AUC 0.820 vs 0.815 on test), so we chose logistic regression for explainability.

**Threshold.** On the validation set, threshold 0.402 catches 85% of diabetics. For the app, younger age groups get lower thresholds (18-39: 0.110, 40-49: 0.284, 50-59: 0.353, 60+: 0.402), chosen on the validation set.

**App.** The whole model is saved as 11 weights plus scaling numbers and thresholds in `model.json`. `app.py` (Gradio) does the same maths. We checked that the app gives the same decision as the notebook for all 50,736 test people (0 differences).

## Results

All numbers below are from the **held-out test set** (50,736 people, 7,069 diabetics), unless marked otherwise.

### 1. Model vs. baselines (single threshold 0.402)

| Method | Diabetics caught (recall) | Missed (false negatives) | False alarms (false positives) | % of people flagged | Precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline 1: always say "no" | 0% | 7,069 | 0 | 0% | n/a | 0.500 |
| Baseline 2: ADA checklist (score ≥ 5) | 81.2% | 1,331 | 18,450 | 47.7% | 23.7% | 0.757 |
| **Logistic regression** | **84.9%** | **1,065** | **16,145** | **43.7%** | **27.1%** | **0.815** |
| Gradient boosting | 85.1% | 1,052 | 15,950 | 43.3% | 27.4% | 0.820 |

- **Our model beats the ADA checklist on both counts at once:** 266 fewer diabetics missed *and* 2,305 fewer healthy people sent to a doctor.
- "Always no" has the highest accuracy (86.1%) but catches nobody. That's why we judge by recall and count each mistake separately.
- **Honest caveat:** our model uses 11 questions, while our ADA version uses 5, so part of the gain comes from the extra questions, not only from the AI.
- **Stability:** validation and test results differed by at most 0.002, and the train-validation ROC-AUC gap was 0.003 (no sign of overfitting or leakage).

### 2. The app version (age-group thresholds)

| | Single threshold | **App version (age-group thresholds)** | ADA checklist |
|---|---|---|---|
| Diabetics caught (recall) | 84.9% | **88.3%** | 81.2% |
| Missed | 1,065 | **829** | 1,331 |
| False alarms | 16,145 | **20,066** | 18,450 |
| % of people flagged | 43.7% | **51.8%** | 47.7% |

The app version **trades some of the referral advantage for fairness across ages:** it catches 236 more diabetics than the single threshold, but flags 51.8% of people, slightly more than the ADA checklist (47.7%).

### 3. Fairness

By sex the gap was small (recall: women 83.7%, men 86.3%). **By age there was a big gap.** With the single threshold, the model **missed about half of diabetics under 40:**

| Age group | Recall: single threshold → app version | False alarm rate: single → app |
|---|---|---|
| 18-39 | **48.4% → 88.0%** | 7.0% → 42.0% |
| 40-49 | 67.9% → 81.6% | 19.0% → 30.1% |
| 50-59 | 80.4% → 84.7% | 32.3% → 38.2% |
| 60-69 | 88.8% (unchanged) | 47.0% (unchanged) |
| 70+ | 91.4% (unchanged) | 68.0% (unchanged) |

**The fix and its cost:** lower thresholds for the three younger groups (chosen on validation) catch **236 more diabetics** but cause **3,921 more false alarms**, about **17 extra blood tests per extra diabetic caught**. We chose the fix for the app because missing young diabetics would fail many of the people the tool is for, and an HbA1c test is cheap. The young-group threshold is based on only about 260 young diabetics, so it's less precise.

### 4. What matters most, and where it fails

- **Most important questions** (both models agree): general health, BMI, age, high blood pressure, high cholesterol. Physical activity and smoking added almost nothing once the others were known.
- **New answer patterns:** for the 20.3% of test people whose exact answers never appeared in training, recall was 93.3%, but the false alarm rate was 65.5% (vs 30.8%) and ROC-AUC 0.743 (vs 0.821). These people are unusual, often with several risk factors.
- **Where it fails (real test examples):**
  - Missed: a man aged 25-29 with BMI 20 and "excellent" health. Nothing in his answers suggests diabetes (possibly type 1, which questions can't detect).
  - Missed: a lean woman aged 65-69 (BMI 24, no high blood pressure). Diagnosed people may have lost weight after diagnosis, and lean diabetes may be more common in South Asian populations in the UAE.
  - False alarm: a woman aged 75-79 with a recorded BMI of 90 (likely a data error), which is why the app checks inputs.
  - False alarm: a woman aged 55-59 with high blood pressure and high cholesterol. She could be prediabetic or undiagnosed, which the survey can't tell.
- **Overall:** missed diabetics look healthy, and false alarms look like diabetics. Questions can narrow down who needs a blood test, but they can't replace it.

### 5. Extra analyses (added after the main results)

All thresholds were chosen on the validation set, then the test set was reported once.

**Is it the AI, or just more questions?** Test set, at the same workload as the ADA checklist (about 47.5% flagged):

| Method | ROC-AUC | Diabetics missed | False alarms |
|---|---|---|---|
| ADA score (5 questions, expert points) | 0.757 | 1,331 | 18,450 |
| Logistic regression, same 5 questions | 0.774 | 1,209 | 18,308 |
| Logistic regression, all 11 questions | 0.815 | 895 | 17,880 |

- **About 28% of the improvement comes from better weights** (learned from data instead of expert points), and **about 72% from the 6 extra questions.**
- At the same recall as the ADA (about 81%), the 5-question model flags 45.0% of people and the 11-question model 38.9%, vs 47.7% for ADA.
- The learned weights suggest high blood pressure deserves about as much weight as age and BMI, while the ADA checklist gives it less.

**UAE stress test (a simulation, not real UAE data).** People of Asian origin tend to develop type 2 diabetes at a lower BMI (a WHO expert consultation suggested BMI action points of about 23 and 27.5 for Asian populations, instead of 25 and 30). We lowered every test diabetic's BMI by 2.5 and ran the app version:

| Test set | Recall | Diabetics missed | False alarms | % flagged |
|---|---|---|---|---|
| Normal data (app as it is) | 88.3% | 829 | 20,066 | 51.8% |
| Simulated leaner diabetics, no fix | 84.0% | 1,129 | 20,066 | 51.3% |
| Simulated, with fix (+2.5 BMI for South/East Asian customers) | 88.3% | 829 | 22,674 | 57.0% |

- If UAE diabetics are leaner, the app could **miss 300 more diabetics**. A BMI adjustment (chosen on validation) recovers them, but costs **2,608 extra false alarms**.
- Because we built the 2.5 gap ourselves, the fix is almost guaranteed to work in the simulation. This shows the size of the risk, not the right adjustment. **Only real UAE data with blood tests can measure it.** The app was not changed.

**Do we need all 11 questions?** Physical activity and smoking added almost nothing (Section 12), so we retrained without them (thresholds chosen on validation, as for the main model):

| Test set | Recall | Diabetics missed | False alarms | ROC-AUC |
|---|---|---|---|---|
| 11 questions, single threshold | 84.9% | 1,065 | 16,145 | 0.815 |
| 9 questions, single threshold | 84.8% | 1,071 | 16,169 | 0.815 |
| 11 questions, app version | 88.3% | 829 | 20,066 | 0.815 |
| 9 questions, app version | 88.2% | 836 | 20,071 | 0.815 |

- The 9-question version is practically identical (6-7 more missed out of 7,069, ROC-AUC down 0.0001), so the screening could be **2 questions shorter**. We kept 11 questions in the app for now, because they're quick to ask and useful for health advice, and UAE data should confirm the result first.

## What we'd do next

1. **Collect UAE data with blood tests:** UAE residents of different nationalities answer our questions *and* get an HbA1c test, so the label is the true answer, including undiagnosed people.
2. **Add missing questions:** family history, gestational diabetes, and ethnic background (people of South Asian origin tend to develop diabetes at a lower BMI).
3. **Retrain, re-choose the threshold, and re-check fairness**, also by nationality group.
4. **Pilot in real pharmacies:** measure how many flagged customers get tested and diagnosed.
5. **Reduce the referral rate**, e.g. a cheap finger-prick test at the pharmacy as a second step for flagged customers.
6. **Host the app permanently** (e.g. Hugging Face Spaces) instead of a temporary Colab link.

## Sources

- **Dataset:** CDC Diabetes Health Indicators (BRFSS 2015), UCI Machine Learning Repository, id 891: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- **3-class version counts** (used for the prediabetes check): Diabetes Health Indicators Dataset on Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- **UAE statistic:** IDF Diabetes Atlas (International Diabetes Federation): https://diabetesatlas.org/
- **ADA Type 2 Diabetes Risk Test:** American Diabetes Association: https://diabetes.org/diabetes-risk-test
- **Libraries:** scikit-learn, pandas, NumPy, matplotlib, Gradio, ucimlrepo (and joblib listed in requirements)
- **Tools:** Google Colab (running the notebook), GitHub and GitHub Desktop
- **AI tools used:** Claude Code (Anthropic) helped plan the project, write and explain the notebook code and app, check the outputs we pasted back from Colab, and draft this README. All results come from our own runs of the notebook.

> **This project is a prototype and is not medical advice.**

## How to run

**The notebook (no installation needed):**
1. Open https://colab.research.google.com/github/MrSiwan0909/silent-diabetes/blob/main/silent_diabetes.ipynb
2. Click **Runtime → Run all**. The first cell installs the packages, and the data downloads automatically. It takes a few minutes.

**The app:**
- **In Colab:** after Run all, the last cell prints a public link ending in `.gradio.live`. Open it on any phone or laptop. The link works while the Colab session is running.
- **On a computer with Python:**
  ```bash
  pip install -r requirements.txt
  python app.py
  ```
  Then open the local address it prints (usually http://127.0.0.1:7860).
