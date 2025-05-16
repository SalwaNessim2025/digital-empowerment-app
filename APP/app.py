import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ✅ يجب أن يكون هذا أول أمر بعد الاستيراد
st.set_page_config(page_title="نموذج التمكين النفسي الرقمي", layout="centered")

# ✅ تنسيق الواجهة (من اليمين لليسار + خطوط)
st.markdown("""
    <style>
    body, .main, .block-container {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .stSlider label, .stSelectbox label, .stTextInput label, .stNumberInput label {
        direction: rtl;
        text-align: right;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ✅ عنوان رئيسي
st.title("🧠 التنبؤ بالتمكين النفسي الرقمي")
st.markdown("أدخل درجاتك لتقدير مستوى التمكين النفسي الرقمي باستخدام نموذج تعلم الآلة القابل للتفسير.")

# ✅ تحميل النموذج
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_model.pkl')
    return joblib.load(model_path)

model = load_model()

# ✅ عرض معلومات سريعة عن النموذج
st.markdown("### ⚙️ تفاصيل النموذج:")
st.write(f"✅ النموذج الفعلي داخل Pipeline: {type(model.named_steps['gbr'])}")

# ✅ نموذج الإدخال
with st.form("prediction_form"):
    st.subheader("📝 أدخل بياناتك:")

    st.markdown("""
        <style>
            .stSlider {
                direction: ltr !important;
            }
            .slider-label {
                text-align: center;
                font-weight: bold;
                display: block;
                margin-bottom: 6px;
                font-size: 18px;
            }
        </style>
    """, unsafe_allow_html=True)

    # المتغيرات منفصلة مع تسمية جميلة
    st.markdown('<div class="slider-label">💪 الصمود الرقمي</div>', unsafe_allow_html=True)
    dr = st.slider(" ", min_value=1.0, max_value=5.0, step=0.1, key="dr", label_visibility="collapsed")

    st.markdown('<div class="slider-label">🎯 الذكاء الانفعالي الرقمي</div>', unsafe_allow_html=True)
    dei = st.slider(" ", min_value=1.0, max_value=5.0, step=0.1, key="dei", label_visibility="collapsed")

    st.markdown('<div class="slider-label">🤝 الدعم الاجتماعي</div>', unsafe_allow_html=True)
    ss = st.slider(" ", min_value=1.0, max_value=5.0, step=0.1, key="ss", label_visibility="collapsed")

    submitted = st.form_submit_button("🔍 تنفيذ التنبؤ")

# ✅ التنبؤ بالنتيجة
if submitted:
    input_df = pd.DataFrame({
        'Digital_Resilience': [dr],
        'Digital_Emotional_Intelligence': [dei],
        'Social_Support': [ss]
    })

    prediction = model.predict(input_df)[0]
    prediction_rounded = round(prediction, 2)

    if prediction < 2.5:
        level = "🔴 منخفض"
        color = "red"
    elif prediction < 3.5:
        level = "🟠 متوسط"
        color = "orange"
    else:
        level = "🟢 مرتفع"
        color = "green"

    st.markdown("---")
    st.subheader("📈 النتيجة المتوقعة:")
    st.markdown(f"<h2 style='color:{color};'>التمكين النفسي الرقمي المتوقع: {prediction_rounded} ({level})</h2>", unsafe_allow_html=True)
    st.progress(min(prediction / 5.0, 1.0))

    st.markdown("### 🔍 تفاصيل الإدخال:")
    st.dataframe(input_df.style.format(precision=2))

# ✅ معلومات إضافية عن النموذج
st.markdown("---")
st.markdown("<h2 style='text-align: right;'>ℹ️ <b>عن النموذج</b></h2>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='font-size:17px; text-align: right; font-weight: bold; line-height:1.8'>
    طورت هذا النموذج/ سلوى سامي نسيم الباحثة بقسم علم النفس بكلية التربية - جامعة عين شمس، باستخدام خوارزميات تعلم الآلة القابلة للتفسير، وتم تدريبه على بيانات طلاب المرحلة الثانوية.<br><br>
    يعتمد النموذج على المدخلات التالية:<br>
    💪 الصمود الرقمي | 🎯 الذكاء الانفعالي الرقمي | 🤝 الدعم الاجتماعي
    </div>
    """,
    unsafe_allow_html=True
)
