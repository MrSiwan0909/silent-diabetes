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
        "header_pill": "Diabetes screening · Prototype",
        "header_title": "Silent Diabetes",
        "header_sub": "A one-minute type 2 diabetes check. <i>No blood test.</i>",
        "header_tag": "For pharmacists and health-booth volunteers",
        "header_stats": ["11 simple questions", "Under one minute", "Caught 88% of diabetics in testing"],
        "sec_about": "### 1. About the customer",
        "sec_medical": "### 2. Medical history",
        "sec_lifestyle": "### 3. Health and lifestyle",
        "placeholder": ("📝 Ask the customer each question, then press <b>Check risk</b>. "
                        "The result will appear here."),
        "clear": "New customer",
        "age": "How old are you? (years, 18 or older)",
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
        "headline_low": "Low risk of type 2 diabetes",
        "pill_high": "Higher risk",
        "headline_high": "Please see a doctor for an HbA1c blood test",
        "high": "Higher risk: please see a doctor for an HbA1c blood test",
        "reasons_high": "Main reasons:",
        "reasons_low": "Main reasons the risk is low:",
        "watch": "Worth keeping an eye on:",
        "low_note": ("If you have a parent, brother or sister with diabetes, or symptoms such as strong thirst "
                     "or passing urine often, see a doctor anyway. This check can't ask about everything."),
        "high_note": ("This is not a diagnosis. Many people flagged by this check do not have diabetes. "
                      "The HbA1c blood test will give the real answer."),
        "bmi_line": "BMI {bmi:.1f} · {category}",
        "bmi_categories": ["underweight", "healthy weight", "overweight", "obese"],
        "what_high": ("These answers look similar to people who have type 2 diabetes. "
                      "Only a blood test can say for sure."),
        "what_low": "These answers look similar to people who usually don't have type 2 diabetes.",
        "stat_high": "In our test data, about {n} in 100 people with this result had diabetes.",
        "stat_low": "In our test data, about {n} in 100 people with this result had diabetes.",
        "in_favour": "Things in your favour:",
        "dots_legend": "Dots show how strongly each answer affects the result.",
        "next_title": "Next steps",
        "next_high": ["Book an HbA1c blood test with a doctor or clinic, ideally in the next few weeks.",
                      "It's a simple blood test, and you usually don't need to fast for it.",
                      "Show this result to the doctor."],
        "next_low": ["Based on these answers, no blood test is needed right now.",
                     "Check again in about a year, or sooner if your health changes."],
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
        "header_pill": "فحص السكري · نموذج أولي",
        "header_title": "السكري الصامت",
        "header_sub": "فحص سريع للسكري من النوع الثاني في دقيقة واحدة. <i>بدون فحص دم.</i>",
        "header_tag": "للصيادلة ومتطوعي الأكشاك الصحية",
        "header_stats": ["11 سؤالاً بسيطاً", "أقل من دقيقة", "اكتشف 88% من مرضى السكري في الاختبار"],
        "sec_about": "### 1. معلومات عن العميل",
        "sec_medical": "### 2. التاريخ الطبي",
        "sec_lifestyle": "### 3. الصحة ونمط الحياة",
        "placeholder": "📝 اطرح كل سؤال على العميل ثم اضغط <b>تحقّق من الخطر</b>. ستظهر النتيجة هنا.",
        "clear": "عميل جديد",
        "age": "كم عمرك؟ (بالسنوات، 18 فأكثر)",
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
        "headline_low": "خطر منخفض للإصابة بالسكري من النوع الثاني",
        "pill_high": "خطر أعلى",
        "headline_high": "يُنصح بمراجعة الطبيب لإجراء فحص السكر التراكمي (HbA1c)",
        "high": "خطر أعلى: يُنصح بمراجعة الطبيب لإجراء فحص السكر التراكمي (HbA1c)",
        "reasons_high": "أهم الأسباب:",
        "reasons_low": "أسباب انخفاض الخطر:",
        "watch": "أمور تستحق الانتباه:",
        "low_note": ("إذا كان أحد والديك أو إخوتك مصاباً بالسكري، أو كانت لديك أعراض مثل العطش الشديد "
                     "أو كثرة التبول، فراجع الطبيب على أي حال. هذا الفحص لا يسأل عن كل شيء."),
        "high_note": ("هذا ليس تشخيصاً. كثير ممن يظهر لديهم خطر أعلى في هذا الفحص ليسوا مصابين بالسكري. "
                      "فحص السكر التراكمي (HbA1c) يعطي الإجابة الحقيقية."),
        "bmi_line": "مؤشر كتلة الجسم {bmi:.1f} · {category}",
        "bmi_categories": ["نقص في الوزن", "وزن صحي", "زيادة في الوزن", "سمنة"],
        "what_high": ("هذه الإجابات تشبه إجابات أشخاص مصابين بالسكري من النوع الثاني. "
                      "فحص الدم وحده يمكنه التأكد."),
        "what_low": "هذه الإجابات تشبه إجابات أشخاص لا يعانون عادةً من السكري من النوع الثاني.",
        "stat_high": "في بيانات الاختبار لدينا، كان حوالي {n} من كل 100 شخص حصلوا على هذه النتيجة مصابين بالسكري.",
        "stat_low": "في بيانات الاختبار لدينا، كان حوالي {n} من كل 100 شخص حصلوا على هذه النتيجة مصابين بالسكري.",
        "in_favour": "عوامل في صالحك:",
        "dots_legend": "تُظهر النقاط مدى تأثير كل إجابة على النتيجة.",
        "next_title": "الخطوات التالية",
        "next_high": ["احجز فحص السكر التراكمي (HbA1c) لدى طبيب أو عيادة، ويُفضَّل خلال الأسابيع القليلة القادمة.",
                      "إنه فحص دم بسيط، وعادةً لا يحتاج إلى صيام.",
                      "اعرض هذه النتيجة على الطبيب."],
        "next_low": ["بناءً على هذه الإجابات، لا حاجة لفحص دم حالياً.",
                     "أعد الفحص بعد سنة تقريباً، أو قبل ذلك إذا تغيّرت حالتك الصحية."],
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
# Text colours are set on EVERY element (not just the box), because Gradio's dark mode
# otherwise turns paragraph and list text white, which is unreadable on our light boxes.
DARK_TEXT = "color:#1c1917 !important"   # warm near-black
GREY_TEXT = "color:#57534e !important"   # warm grey

# Fonts: an elegant serif for big headings, a clean sans-serif for everything else
# (loaded from Google Fonts in the CSS below; the Arabic fonts are used when the text is Arabic)
TITLE_FONT = "'Instrument Serif', 'Noto Naskh Arabic', Georgia, serif"
BODY_FONT = "'Inter', 'IBM Plex Sans Arabic', system-ui, sans-serif"

# From the notebook's failure analysis (Section 13), app version on the 50,736 test people:
TEST_HIGHER_DIABETIC, TEST_HIGHER_TOTAL = 6240, 6240 + 20066   # flagged: diabetic / all flagged
TEST_LOW_DIABETIC, TEST_LOW_TOTAL = 829, 829 + 23601           # low risk: diabetic / all low risk


def card(inner_html, lang, background="#ffffff", accent="#d6d3d1"):
    """A rounded box with a coloured stripe on one side."""
    direction = "rtl" if lang == "ar" else "ltr"
    side = "right" if lang == "ar" else "left"
    return (f'<div dir="{direction}" style="padding:24px 26px;border-radius:18px;background:{background};'
            f'border:1px solid #e7e5e4;border-{side}:4px solid {accent};'
            f'box-shadow:0 1px 2px rgba(0,0,0,0.04),0 8px 24px rgba(0,0,0,0.05);'
            f'{DARK_TEXT};font-family:{BODY_FONT};font-size:16px;line-height:1.6">{inner_html}</div>')


def para(text, style=DARK_TEXT, extra=""):
    return f'<p style="{style};margin:6px 0;{extra}">{text}</p>'


def message_box(text, lang):
    return card(para(text), lang)


def placeholder(lang):
    return card(para(TEXT[lang]["placeholder"]), lang)


def bmi_category(bmi, lang):
    """WHO adult categories: under 18.5, 18.5-24.9, 25-29.9, 30 and over."""
    names = TEXT[lang]["bmi_categories"]
    if bmi < 18.5:
        return names[0]
    if bmi < 25:
        return names[1]
    if bmi < 30:
        return names[2]
    return names[3]


def strength_dots(push, colour="#1c1917"):
    """3 dots = strong effect, 2 = medium, 1 = small. No numbers, just a visual hint."""
    size = abs(push)
    filled = 3 if size >= 0.5 else (2 if size >= 0.2 else 1)
    return (f'<span style="color:#1c1917 !important;letter-spacing:2px">{"●" * filled}</span>'
            f'<span style="color:#d6d3d1 !important;letter-spacing:2px">{"●" * (3 - filled)}</span>')


def assess(language, age, sex, height_cm, weight_kg, high_bp, high_chol, phys_activity,
           gen_hlth, diff_walk, smoker, heart, stroke):
    lang = LANGUAGES[language]
    t = TEXT[lang]

    # Check every question has an answer
    inputs = [age, sex, height_cm, weight_kg, high_bp, high_chol, phys_activity,
              gen_hlth, diff_walk, smoker, heart, stroke]
    if any(value is None for value in inputs):
        return message_box("⚠️ " + t["missing"], lang)

    # Check the numbers are realistic (catches typing mistakes like the BMI of 90 in our failure analysis)
    if not (MIN_AGE <= age <= MAX_AGE):
        return message_box("⚠️ " + t["bad_age"], lang)
    bmi = weight_kg / (height_cm / 100) ** 2
    if not (MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM and MIN_WEIGHT_KG <= weight_kg <= MAX_WEIGHT_KG
            and MIN_BMI <= bmi <= MAX_BMI):
        return message_box("⚠️ " + t["bad_body"], lang)

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

    def reason_list(features, colour):
        items = "".join(
            f'<li style="{DARK_TEXT};margin:5px 0;list-style:none">'
            f'{strength_dots(pushes[f], colour)}&nbsp;&nbsp;{reason_words(f, answers[f], lang)}</li>'
            for f in features)
        return f'<ul style="margin:4px 0 10px 0;padding:0">{items}</ul>'

    def heading(text):
        return f'<div style="{DARK_TEXT};font-weight:700;margin-top:12px">{text}</div>'

    def steps_list(steps):
        items = "".join(f'<li style="{DARK_TEXT};margin:3px 0">{s}</li>' for s in steps)
        return f'<ul style="margin:4px 0 10px 0">{items}</ul>'

    if higher_risk:
        background, accent, pill_bg, pill_text, headline = "#ffffff", "#ea580c", "#fff1e6", t["pill_high"], t["headline_high"]
        what = t["what_high"]
        stat = t["stat_high"].format(n=round(100 * TEST_HIGHER_DIABETIC / TEST_HIGHER_TOTAL))
        details = heading(t["reasons_high"]) + reason_list(raising[:3], accent)
        if lowering:
            details += heading(t["in_favour"]) + reason_list(lowering[:2], "#16a34a")
        details += heading(t["next_title"]) + steps_list(t["next_high"])
        details += para(t["high_note"], GREY_TEXT, "font-size:14px")
    else:
        background, accent, pill_bg, pill_text, headline = "#ffffff", "#16a34a", "#e9f7ee", t["low"], t["headline_low"]
        what = t["what_low"]
        stat = t["stat_low"].format(n=round(100 * TEST_LOW_DIABETIC / TEST_LOW_TOTAL))
        details = heading(t["reasons_low"]) + reason_list(lowering[:3], accent)
        if raising:
            details += heading(t["watch"]) + reason_list(raising[:2], "#ea580c")
        details += heading(t["next_title"]) + steps_list(t["next_low"])
        details += para(t["low_note"], GREY_TEXT, "font-size:14px")

    bmi_text = t["bmi_line"].format(bmi=bmi, category=bmi_category(bmi, lang))
    disclaimer = DISCLAIMER_EN if lang == "en" else f"{DISCLAIMER_AR}<br>{DISCLAIMER_EN}"
    inner = (
        # coloured status pill (orange / green), then the headline in the serif font
        f'<span style="display:inline-block;background:{pill_bg};color:{accent} !important;border-radius:999px;'
        f'padding:4px 14px;font-size:13px;font-weight:600;letter-spacing:0.3px">● {pill_text}</span>'
        f'<div style="font-family:{TITLE_FONT};font-size:30px;line-height:1.2;{DARK_TEXT};margin:12px 0 10px 0">'
        f'{headline}</div>'
        f'<span style="display:inline-block;background:#f5f5f4;border-radius:999px;'
        f'padding:3px 12px;font-size:13px;{DARK_TEXT}">{bmi_text}</span>'
        + para(what)
        + para(stat, GREY_TEXT, "font-size:14px")
        + details
        + para(t["dots_legend"], GREY_TEXT, "font-size:12px")
        + f'<hr style="border:none;border-top:1px solid #e7e5e4;margin:12px 0">'
        + para(f"<i>{disclaimer}</i>", GREY_TEXT, "font-size:13px")
    )
    return card(inner, lang, background, accent)


# ---------------------------------------------------------------------------
# 5. Page header and look (black-and-white style)
# ---------------------------------------------------------------------------
def header_html(lang):
    t = TEXT[lang]
    direction = "rtl" if lang == "ar" else "ltr"
    stats = "".join(f'<span class="sd-stat">{s}</span>' for s in t["header_stats"])
    return (f'<div class="sd-header" dir="{direction}">'
            f'<span class="sd-pill">{t["header_pill"]}</span>'
            f'<div class="sd-title">{t["header_title"]}</div>'
            f'<div class="sd-sub">{t["header_sub"]}</div>'
            f'<div class="sd-tag">{t["header_tag"]}</div>'
            f'<div class="sd-stats">{stats}</div></div>')


# Black-and-white, formal look: warm off-white page, near-black text, serif headings,
# white cards with thin borders, black pill-shaped buttons.
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;600&family=Noto+Naskh+Arabic:wght@400;600&display=swap');

body, .gradio-container {{background: #faf9f7 !important;}}
.gradio-container {{max-width: 880px !important; margin: auto !important; font-family: {BODY_FONT} !important;}}

.sd-header {{text-align: center; padding: 34px 16px 26px 16px;}}
.sd-pill {{display: inline-block; background: #1c1917; color: #ffffff !important; border-radius: 999px;
           padding: 5px 16px; font-size: 13px; letter-spacing: 0.3px;}}
.sd-title {{font-family: {TITLE_FONT}; font-size: 64px; line-height: 1.05; margin-top: 16px; color: #1c1917 !important;}}
.sd-sub {{font-family: {TITLE_FONT}; font-size: 28px; margin-top: 6px; color: #1c1917 !important;}}
.sd-tag {{font-size: 15px; margin-top: 10px; color: #57534e !important;}}
.sd-stats {{display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; margin-top: 18px;}}
.sd-stat {{background: #ffffff; border: 1px solid #e7e5e4; border-radius: 999px; padding: 6px 14px;
           font-size: 13px; color: #1c1917 !important;}}

/* The form card: white, rounded, with plenty of space inside so nothing touches the edges */
.sd-card {{background: #ffffff !important; border: 1px solid #e7e5e4 !important; border-radius: 22px !important;
           padding: 28px 36px 32px 36px !important; gap: 0 !important;
           box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 10px 30px rgba(0,0,0,0.05) !important;}}
/* Remove the grey boxes Gradio puts around things inside the card */
.sd-card .block, .sd-card .form, .sd-card .prose, .sd-card .wrap {{background: transparent !important;
           border: none !important; box-shadow: none !important;}}

/* Part headings (1. About the customer ...) */
.sd-section {{padding: 22px 0 6px 0 !important;}}
.sd-section h3 {{font-family: {TITLE_FONT} !important; font-weight: 400 !important; font-size: 28px !important;
                 color: #1c1917 !important; margin: 0 !important;}}

/* One question per row: question on the left, answer on the right, thin line between rows */
.sd-qrow {{padding: 14px 0 !important; border-top: 1px solid #f0eeeb !important; align-items: center !important;
           gap: 20px !important;}}
.sd-q p {{font-size: 16px !important; color: #1c1917 !important; margin: 0 !important; line-height: 1.45 !important;}}
/* Stop the first letter of each line being cut off: give the text a little room and let it show fully */
.sd-q, .sd-q *, .sd-section, .sd-section * {{overflow: visible !important;}}
.sd-q .prose, .sd-q p, .sd-section .prose, .sd-section h3 {{padding-left: 6px !important; padding-right: 4px !important;
           font-style: normal !important;}}
.sd-qrow input[type=number] {{border: 1px solid #d6d3d1 !important; border-radius: 10px !important;
           padding: 10px 12px !important; background: #ffffff !important;}}

/* Last row of the form: the buttons */
.sd-buttons {{padding-top: 26px !important; border-top: 1px solid #f0eeeb !important; gap: 12px !important;}}
#check-btn, #clear-btn {{border-radius: 999px !important; min-height: 52px; font-size: 16px !important;}}

.sd-disclaimer p {{text-align: center; color: #57534e !important; font-size: 14px !important;}}
#lang-switch {{justify-content: flex-end;}}

@media (max-width: 640px) {{
  .sd-title {{font-size: 44px;}}
  .sd-sub {{font-size: 22px;}}
  .sd-card {{padding: 20px 20px 24px 20px !important;}}
  .sd-qrow {{gap: 8px !important;}}
}}
"""

# Always show the light (black-on-white) version, even if the computer is in dark mode
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}
"""


def make_theme():
    """A plain black-and-white Gradio theme."""
    try:
        return gr.themes.Base(
            primary_hue="neutral", neutral_hue="stone",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ).set(
            body_background_fill="#faf9f7",
            body_text_color="#1c1917",
            block_background_fill="#ffffff",
            block_border_color="#e7e5e4",
            block_radius="16px",
            block_shadow="0 1px 2px rgba(0,0,0,0.04)",
            input_background_fill="#ffffff",
            input_border_color="#d6d3d1",
            button_primary_background_fill="#1c1917",
            button_primary_background_fill_hover="#44403c",
            button_primary_text_color="#ffffff",
            button_secondary_background_fill="#ffffff",
            button_secondary_text_color="#1c1917",
            button_secondary_border_color="#1c1917",
            checkbox_label_background_fill_selected="#1c1917",
            checkbox_label_text_color_selected="#ffffff",
            checkbox_background_color_selected="#1c1917",
            checkbox_border_color_selected="#1c1917",
        )
    except Exception:
        # If this Gradio version doesn't know one of the settings, fall back to the plain theme
        return gr.themes.Base(primary_hue="neutral", neutral_hue="stone")


# ---------------------------------------------------------------------------
# 6. Build the page
# ---------------------------------------------------------------------------
# Gradio 6 moved the theme, css and js settings from gr.Blocks() to launch(), so we support both
GRADIO_MAJOR_VERSION = int(gr.__version__.split(".")[0])
STYLE = {"theme": make_theme(), "css": CSS, "js": FORCE_LIGHT_JS}
BLOCKS_STYLE = STYLE if GRADIO_MAJOR_VERSION < 6 else {}
LAUNCH_STYLE = STYLE if GRADIO_MAJOR_VERSION >= 6 else {}


def build_app():
    t = TEXT["en"]

    # The questions in each part of the form (keys into TEXT)
    sections = [
        ("sec_about", ["age", "sex", "height", "weight"]),
        ("sec_medical", ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"]),
        ("sec_lifestyle", ["GenHlth", "PhysActivity", "DiffWalk", "Smoker"]),
    ]
    # Which answer choices each multiple-choice question uses (the rest are number boxes)
    choice_keys = {"sex": "sex_choices", "GenHlth": "genhlth_choices"}
    for key in ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "PhysActivity", "DiffWalk", "Smoker"]:
        choice_keys[key] = "yes_no"
    number_keys = ["age", "height", "weight"]

    headings, question_text, answer_boxes = {}, {}, {}

    with gr.Blocks(title="Silent Diabetes", **BLOCKS_STYLE) as demo:
        # Language switch in the top corner, then the header
        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML("")
            with gr.Column(scale=1, min_width=200):
                language = gr.Radio(list(LANGUAGES), value="English", show_label=False,
                                    container=False, elem_id="lang-switch")
        header = gr.HTML(header_html("en"))

        # The form: one white card, one question per row, 'Check risk' on the last row
        with gr.Column(elem_classes="sd-card"):
            for section_key, keys in sections:
                headings[section_key] = gr.Markdown(t[section_key], elem_classes="sd-section")
                for key in keys:
                    with gr.Row(elem_classes="sd-qrow", equal_height=True):
                        with gr.Column(scale=3, min_width=260):
                            question_text[key] = gr.Markdown(t[key], elem_classes="sd-q")
                        with gr.Column(scale=2, min_width=240):
                            if key in number_keys:
                                answer_boxes[key] = gr.Number(show_label=False, container=False,
                                                              precision=0 if key == "age" else None)
                            else:
                                answer_boxes[key] = gr.Radio(t[choice_keys[key]], show_label=False,
                                                             container=False)
            with gr.Row(elem_classes="sd-buttons"):
                check = gr.Button(t["button"], variant="primary", elem_id="check-btn", scale=3)
                clear = gr.Button(t["clear"], variant="secondary", elem_id="clear-btn", scale=1)

        # The result appears under the form
        result = gr.HTML(placeholder("en"))
        disclaimer = gr.Markdown(t["disclaimer"], elem_classes="sd-disclaimer")  # always visible

        # The order here must match the order of assess()'s arguments
        assess_order = ["age", "sex", "height", "weight", "HighBP", "HighChol", "PhysActivity",
                        "GenHlth", "DiffWalk", "Smoker", "HeartDiseaseorAttack", "Stroke"]
        question_inputs = [answer_boxes[key] for key in assess_order]

        check.click(assess, inputs=[language] + question_inputs, outputs=result)

        def new_customer(language_choice):
            """Empty every answer and show the placeholder again."""
            return [gr.update(value=None) for _ in question_inputs] + [placeholder(LANGUAGES[language_choice])]

        clear.click(new_customer, inputs=language, outputs=question_inputs + [result])

        # Everything that changes when the language changes, in a fixed order
        all_question_keys = [key for _, keys in sections for key in keys]
        choice_question_keys = [key for key in all_question_keys if key in choice_keys]
        language_outputs = ([header] + [headings[s] for s, _ in sections]
                            + [question_text[k] for k in all_question_keys]
                            + [answer_boxes[k] for k in choice_question_keys]
                            + [check, clear, disclaimer, result])

        def switch_language(language_choice):
            """Rewrite every heading, question and choice in the chosen language
            (same order as language_outputs above)."""
            lang = LANGUAGES[language_choice]
            tt = TEXT[lang]
            return ([gr.update(value=header_html(lang))]
                    + [gr.update(value=tt[s]) for s, _ in sections]
                    + [gr.update(value=tt[k]) for k in all_question_keys]
                    + [gr.update(choices=tt[choice_keys[k]]) for k in choice_question_keys]
                    + [gr.update(value=tt["button"]), gr.update(value=tt["clear"]),
                       gr.update(value=tt["disclaimer"]), gr.update(value=placeholder(lang))])

        language.change(switch_language, inputs=language, outputs=language_outputs)
    return demo


demo = build_app()


def launch(**kwargs):
    """Start the app, e.g. launch(share=True) in Colab for a public link."""
    return demo.launch(**LAUNCH_STYLE, **kwargs)


if __name__ == "__main__":
    launch()
