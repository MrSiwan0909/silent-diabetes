"""
Silent Diabetes: a screening app for pharmacists and health-booth volunteers.

It asks about 10 simple questions (no blood test), calculates BMI from height
and weight, and says whether the person is at low or higher risk of type 2
diabetes. If the risk is higher, it suggests seeing a doctor for an HbA1c test.

The model is a logistic regression trained in silent_diabetes.ipynb on the
CDC BRFSS 2015 survey (US adults). All its numbers are stored in model.json.

This project is a prototype and is not medical advice.

Run in Google Colab: see the last cell of silent_diabetes.ipynb.
Run on a computer:   python app.py
"""

import json
import math
import os
import urllib.request

import gradio as gr

# ---------------------------------------------------------------------------
# 1. Load the model numbers from model.json
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.json")
MODEL_URL = "https://raw.githubusercontent.com/MrSiwan0909/silent-diabetes/main/model.json"

if not os.path.exists(MODEL_PATH):
    # If model.json isn't next to this file, download it from our GitHub repo
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

with open(MODEL_PATH) as f:
    MODEL = json.load(f)

FEATURES = MODEL["features"]  # the 11 questions, in the order the model expects

# Realistic ranges. Anything outside these is probably a typing mistake.
MIN_AGE, MAX_AGE = 18, 100
MIN_HEIGHT_CM, MAX_HEIGHT_CM = 120, 220
MIN_WEIGHT_KG, MAX_WEIGHT_KG = 30, 250
MIN_BMI, MAX_BMI = 12, 70

AGE_LABELS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
              7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}

DISCLAIMER_EN = "This project is a prototype and is not medical advice."
DISCLAIMER_AR = "هذا المشروع نموذج أولي وليس نصيحة طبية."


# ---------------------------------------------------------------------------
# 2. The model maths (the same formula as in the notebook, Section 15)
# ---------------------------------------------------------------------------
def age_to_code(age_years):
    """Turn age in years into the survey's 13 age groups (1 = 18-24, ..., 13 = 80+)."""
    code = 1 + (int(age_years) - 20) // 5
    return max(1, min(13, code))


def age_band(age_code):
    """Which of our 5 age groups (18-39, 40-49, ...) an age code belongs to."""
    for band, codes in MODEL["age_bands"].items():
        if age_code in codes:
            return band
    raise ValueError(f"Unknown age code {age_code}")


def predict(answers):
    """answers: dict with the 11 coded answers, e.g. {'Age': 9, 'Sex': 1, 'BMI': 31.2, ...}.
    Returns (risk_score, is_higher_risk, pushes), where pushes says how much each
    answer moved the risk up (+) or down (-) compared with an average person."""
    pushes = {}
    for i, feature in enumerate(FEATURES):
        scaled = (answers[feature] - MODEL["scaler_mean"][i]) / MODEL["scaler_scale"][i]
        pushes[feature] = MODEL["weights"][i] * scaled
    log_odds = MODEL["intercept"] + sum(pushes.values())
    risk_score = 1 / (1 + math.exp(-log_odds))
    threshold = MODEL["age_group_thresholds"][age_band(answers["Age"])]
    return risk_score, risk_score >= threshold, pushes


# ---------------------------------------------------------------------------
# 3. All the words shown on screen, in English and Arabic
# ---------------------------------------------------------------------------
TEXT = {
    "en": {
        "title": "# Silent Diabetes",
        "intro": ("A one-minute check of type 2 diabetes risk, **without a blood test**. "
                  "Ask the customer each question below, then press **Check risk**."),
        "age": "How old are you? (years)",
        "sex": "Sex",
        "sex_choices": [("Female", 0), ("Male", 1)],
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "HighBP": "Has a doctor ever told you that you have high blood pressure?",
        "HighChol": "Has a doctor ever told you that you have high cholesterol?",
        "PhysActivity": "In the past 30 days, did you do any physical activity or exercise (not counting your job)?",
        "GenHlth": "In general, how would you rate your health?",
        "genhlth_choices": [("Excellent", 1), ("Very good", 2), ("Good", 3), ("Fair", 4), ("Poor", 5)],
        "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",
        "Smoker": "Have you smoked at least 100 cigarettes (about 5 packs) in your whole life?",
        "HeartDiseaseorAttack": "Have you ever had heart disease or a heart attack?",
        "Stroke": "Have you ever had a stroke?",
        "yes_no": [("Yes", 1), ("No", 0)],
        "button": "Check risk",
        "low": "Low risk",
        "high": "Higher risk: please see a doctor for an HbA1c blood test",
        "reasons_high": "Main reasons:",
        "reasons_low": "Main reasons the risk is low:",
        "watch": "Worth keeping an eye on:",
        "low_note": ("If you have a parent, brother or sister with diabetes, or symptoms such as strong thirst "
                     "or passing urine often, see a doctor anyway. This check can't ask about everything."),
        "high_note": ("This is not a diagnosis. Many people flagged by this check do not have diabetes. "
                      "The HbA1c blood test will give the real answer."),
        "bmi_line": "Calculated BMI: {bmi:.1f}",
        "missing": "Please answer all the questions.",
        "bad_age": f"Age must be between {MIN_AGE} and {MAX_AGE} years.",
        "bad_body": (f"Please check height ({MIN_HEIGHT_CM}-{MAX_HEIGHT_CM} cm) and weight "
                     f"({MIN_WEIGHT_KG}-{MAX_WEIGHT_KG} kg). The BMI they give looks unrealistic."),
        "disclaimer": f"**{DISCLAIMER_EN}**",
        # plain-words reasons: (text if answer is yes, text if answer is no)
        "reason": {
            "HighBP": ("High blood pressure", "No high blood pressure"),
            "HighChol": ("High cholesterol", "No high cholesterol"),
            "PhysActivity": ("Physically active", "No physical activity in the past 30 days"),
            "DiffWalk": ("Difficulty walking or climbing stairs", "No difficulty walking"),
            "Smoker": ("History of smoking", "Not a smoker"),
            "HeartDiseaseorAttack": ("History of heart disease", "No heart disease"),
            "Stroke": ("History of stroke", "No stroke"),
        },
        "reason_age": "Age group {label}",
        "reason_sex": ("Male", "Female"),
        "reason_bmi": "BMI of {bmi:.0f}",
        "reason_health": "Rates general health as '{word}'",
        "health_words": {1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"},
    },
    "ar": {
        "title": "# السكري الصامت",
        "intro": ("فحص سريع لمدة دقيقة لخطر الإصابة بالسكري من النوع الثاني، **بدون فحص دم**. "
                  "اطرح كل سؤال على العميل ثم اضغط **تحقّق من الخطر**."),
        "age": "كم عمرك؟ (بالسنوات)",
        "sex": "الجنس",
        "sex_choices": [("أنثى", 0), ("ذكر", 1)],
        "height": "الطول (سم)",
        "weight": "الوزن (كغ)",
        "HighBP": "هل أخبرك طبيب من قبل بأن لديك ارتفاعاً في ضغط الدم؟",
        "HighChol": "هل أخبرك طبيب من قبل بأن لديك ارتفاعاً في الكوليسترول؟",
        "PhysActivity": "خلال آخر 30 يوماً، هل مارست أي نشاط بدني أو رياضة (غير عملك)؟",
        "GenHlth": "بشكل عام، كيف تصف صحتك؟",
        "genhlth_choices": [("ممتازة", 1), ("جيدة جداً", 2), ("جيدة", 3), ("مقبولة", 4), ("ضعيفة", 5)],
        "DiffWalk": "هل تجد صعوبة كبيرة في المشي أو صعود الدرج؟",
        "Smoker": "هل دخّنت 100 سيجارة على الأقل (حوالي 5 علب) طوال حياتك؟",
        "HeartDiseaseorAttack": "هل أُصبت من قبل بمرض في القلب أو بنوبة قلبية؟",
        "Stroke": "هل أُصبت من قبل بسكتة دماغية؟",
        "yes_no": [("نعم", 1), ("لا", 0)],
        "button": "تحقّق من الخطر",
        "low": "خطر منخفض",
        "high": "خطر أعلى: يُنصح بمراجعة الطبيب لإجراء فحص السكر التراكمي (HbA1c)",
        "reasons_high": "أهم الأسباب:",
        "reasons_low": "أسباب انخفاض الخطر:",
        "watch": "أمور تستحق الانتباه:",
        "low_note": ("إذا كان أحد والديك أو إخوتك مصاباً بالسكري، أو كانت لديك أعراض مثل العطش الشديد "
                     "أو كثرة التبول، فراجع الطبيب على أي حال. هذا الفحص لا يسأل عن كل شيء."),
        "high_note": ("هذا ليس تشخيصاً. كثير ممن يظهر لديهم خطر أعلى في هذا الفحص ليسوا مصابين بالسكري. "
                      "فحص السكر التراكمي (HbA1c) يعطي الإجابة الحقيقية."),
        "bmi_line": "مؤشر كتلة الجسم المحسوب: {bmi:.1f}",
        "missing": "يرجى الإجابة عن جميع الأسئلة.",
        "bad_age": f"يجب أن يكون العمر بين {MIN_AGE} و{MAX_AGE} سنة.",
        "bad_body": (f"يرجى التحقق من الطول ({MIN_HEIGHT_CM}-{MAX_HEIGHT_CM} سم) والوزن "
                     f"({MIN_WEIGHT_KG}-{MAX_WEIGHT_KG} كغ). يبدو مؤشر كتلة الجسم غير واقعي."),
        "disclaimer": f"**{DISCLAIMER_AR}**<br>**{DISCLAIMER_EN}**",
        "reason": {
            "HighBP": ("ارتفاع ضغط الدم", "ضغط الدم ليس مرتفعاً"),
            "HighChol": ("ارتفاع الكوليسترول", "الكوليسترول ليس مرتفعاً"),
            "PhysActivity": ("ممارسة النشاط البدني", "عدم ممارسة نشاط بدني خلال آخر 30 يوماً"),
            "DiffWalk": ("صعوبة في المشي أو صعود الدرج", "لا توجد صعوبة في المشي"),
            "Smoker": ("تاريخ من التدخين", "غير مدخّن"),
            "HeartDiseaseorAttack": ("تاريخ من أمراض القلب", "لا توجد أمراض في القلب"),
            "Stroke": ("تاريخ من السكتة الدماغية", "لا توجد سكتة دماغية"),
        },
        "reason_age": "الفئة العمرية {label}",
        "reason_sex": ("ذكر", "أنثى"),
        "reason_bmi": "مؤشر كتلة الجسم {bmi:.0f}",
        "reason_health": "تقييم الصحة العامة: {word}",
        "health_words": {1: "ممتازة", 2: "جيدة جداً", 3: "جيدة", 4: "مقبولة", 5: "ضعيفة"},
    },
}

YES_NO_QUESTIONS = ["HighBP", "HighChol", "PhysActivity", "DiffWalk", "Smoker", "HeartDiseaseorAttack", "Stroke"]
LANGUAGES = {"English": "en", "العربية": "ar"}


def reason_words(feature, value, lang):
    """Turn one answer into plain words, in the chosen language."""
    t = TEXT[lang]
    if feature == "Age":
        return t["reason_age"].format(label=AGE_LABELS[value])
    if feature == "Sex":
        return t["reason_sex"][0] if value == 1 else t["reason_sex"][1]
    if feature == "BMI":
        return t["reason_bmi"].format(bmi=value)
    if feature == "GenHlth":
        return t["reason_health"].format(word=t["health_words"][value])
    yes_text, no_text = t["reason"][feature]
    return yes_text if value == 1 else no_text


# ---------------------------------------------------------------------------
# 4. What happens when the pharmacist presses "Check risk"
# ---------------------------------------------------------------------------
def message_box(text, lang):
    direction = "rtl" if lang == "ar" else "ltr"
    return (f'<div dir="{direction}" style="padding:14px;border-radius:10px;background:#f1f3f4;'
            f'color:#1a1a1a;font-size:16px">{text}</div>')


def assess(language, age, sex, height_cm, weight_kg, high_bp, high_chol, phys_activity,
           gen_hlth, diff_walk, smoker, heart, stroke):
    lang = LANGUAGES[language]
    t = TEXT[lang]

    # Check every question has an answer
    inputs = [age, sex, height_cm, weight_kg, high_bp, high_chol, phys_activity,
              gen_hlth, diff_walk, smoker, heart, stroke]
    if any(value is None for value in inputs):
        return message_box(t["missing"], lang)

    # Check the numbers are realistic (catches typing mistakes like the BMI of 90 in our failure analysis)
    if not (MIN_AGE <= age <= MAX_AGE):
        return message_box(t["bad_age"], lang)
    bmi = weight_kg / (height_cm / 100) ** 2
    if not (MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM and MIN_WEIGHT_KG <= weight_kg <= MAX_WEIGHT_KG
            and MIN_BMI <= bmi <= MAX_BMI):
        return message_box(t["bad_body"], lang)

    answers = {
        "Age": age_to_code(age), "Sex": int(sex), "BMI": bmi,
        "HighBP": int(high_bp), "HighChol": int(high_chol), "PhysActivity": int(phys_activity),
        "GenHlth": int(gen_hlth), "DiffWalk": int(diff_walk), "Smoker": int(smoker),
        "HeartDiseaseorAttack": int(heart), "Stroke": int(stroke),
    }
    _, higher_risk, pushes = predict(answers)

    # Sort answers by how much they push risk up or down; ignore tiny pushes
    min_push = MODEL["min_push_to_show"]
    raising = [f for f in sorted(pushes, key=pushes.get, reverse=True) if pushes[f] > min_push]
    lowering = [f for f in sorted(pushes, key=pushes.get) if pushes[f] < -min_push]

    def bullet_list(features):
        return "".join(f"<li>{reason_words(f, answers[f], lang)}</li>" for f in features)

    if higher_risk:
        colour, border, headline = "#fff4e5", "#e8710a", t["high"]
        details = f"<b>{t['reasons_high']}</b><ul>{bullet_list(raising[:3])}</ul><p>{t['high_note']}</p>"
    else:
        colour, border, headline = "#e6f4ea", "#1e8e3e", t["low"]
        details = f"<b>{t['reasons_low']}</b><ul>{bullet_list(lowering[:3])}</ul>"
        if raising:
            details += f"<b>{t['watch']}</b><ul>{bullet_list(raising[:2])}</ul>"
        details += f"<p>{t['low_note']}</p>"

    direction = "rtl" if lang == "ar" else "ltr"
    disclaimer = DISCLAIMER_EN if lang == "en" else f"{DISCLAIMER_AR}<br>{DISCLAIMER_EN}"
    return (
        f'<div dir="{direction}" style="padding:18px;border-radius:12px;background:{colour};'
        f'border:3px solid {border};color:#1a1a1a;font-size:16px">'
        f'<div style="font-size:24px;font-weight:bold;color:{border};margin-bottom:8px">{headline}</div>'
        f'<p>{t["bmi_line"].format(bmi=bmi)}</p>{details}'
        f'<p style="font-size:13px;color:#444"><i>{disclaimer}</i></p></div>'
    )


# ---------------------------------------------------------------------------
# 5. Switching language: relabel every question and choice
# ---------------------------------------------------------------------------
def switch_language(language):
    t = TEXT[LANGUAGES[language]]
    updates = [
        gr.update(value=t["title"]),
        gr.update(value=t["intro"]),
        gr.update(label=t["age"]),
        gr.update(label=t["sex"], choices=t["sex_choices"]),
        gr.update(label=t["height"]),
        gr.update(label=t["weight"]),
    ]
    for question in ["HighBP", "HighChol", "PhysActivity"]:
        updates.append(gr.update(label=t[question], choices=t["yes_no"]))
    updates.append(gr.update(label=t["GenHlth"], choices=t["genhlth_choices"]))
    for question in ["DiffWalk", "Smoker", "HeartDiseaseorAttack", "Stroke"]:
        updates.append(gr.update(label=t[question], choices=t["yes_no"]))
    updates.append(gr.update(value=t["button"]))
    updates.append(gr.update(value=t["disclaimer"]))
    updates.append(gr.update(value=""))  # clear the old result
    return updates


# ---------------------------------------------------------------------------
# 6. Build the page
# ---------------------------------------------------------------------------
def build_app():
    t = TEXT["en"]
    with gr.Blocks(title="Silent Diabetes") as demo:
        language = gr.Radio(list(LANGUAGES), value="English", label="Language / اللغة")
        title = gr.Markdown(t["title"])
        intro = gr.Markdown(t["intro"])

        with gr.Row():
            age = gr.Number(label=t["age"], precision=0)
            sex = gr.Radio(t["sex_choices"], label=t["sex"])
        with gr.Row():
            height = gr.Number(label=t["height"])
            weight = gr.Number(label=t["weight"])

        high_bp = gr.Radio(t["yes_no"], label=t["HighBP"])
        high_chol = gr.Radio(t["yes_no"], label=t["HighChol"])
        phys_activity = gr.Radio(t["yes_no"], label=t["PhysActivity"])
        gen_hlth = gr.Radio(t["genhlth_choices"], label=t["GenHlth"])
        diff_walk = gr.Radio(t["yes_no"], label=t["DiffWalk"])
        smoker = gr.Radio(t["yes_no"], label=t["Smoker"])
        heart = gr.Radio(t["yes_no"], label=t["HeartDiseaseorAttack"])
        stroke = gr.Radio(t["yes_no"], label=t["Stroke"])

        check = gr.Button(t["button"], variant="primary")
        result = gr.HTML()
        disclaimer = gr.Markdown(t["disclaimer"])  # always visible at the bottom

        question_inputs = [age, sex, height, weight, high_bp, high_chol, phys_activity,
                           gen_hlth, diff_walk, smoker, heart, stroke]
        check.click(assess, inputs=[language] + question_inputs, outputs=result)
        language.change(switch_language, inputs=language,
                        outputs=[title, intro] + question_inputs + [check, disclaimer, result])
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
