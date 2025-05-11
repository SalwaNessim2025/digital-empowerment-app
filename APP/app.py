import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
# إعداد واجهة الصفحة
st.set_page_config(page_title="نموذج التمكين النفسي الرقمي", layout="centered")

# تغيير اتجاه الصفحة إلى من اليمين لليسار باستخدام CSS
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
    </style>
""", unsafe_allow_html=True)
# عنوان رئيسي
st.title("🧠 التنبؤ بالتمكين النفسي الرقمي")
st.markdown("أدخل درجاتك لتقدير مستوى التمكين النفسي الرقمي باستخدام نموذج تعلم الآلة القابل للتفسير.")

# تحميل النموذج المدرب
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_model.pkl')
    return joblib.load(model_path)
model = load_model()

st.markdown("### ⚙️ تفاصيل النموذج:")
st.write(f"✅ النموذج الفعلي داخل Pipeline: {type(model.named_steps['gbr'])}")

# إدخال المستخدم
with st.form("prediction_form"):
    st.subheader("📝 أدخل بياناتك:")

    # إصلاح اتجاه السلايدر باستخدام CSS
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

    # المتغيرات في سطور منفصلة
    st.markdown('<div class="slider-label">💪 الصمود الرقمي</div>', unsafe_allow_html=True)
    dr = st.slider(" ", min_value=1.0, max_value=5.0, step=0.1, key="dr", label_visibility="collapsed")

    st.markdown('<div class="slider-label">🎯 الذكاء الانفعالي الرقمي</div>', unsafe_allow_html=True)
    dei = st.slider(" ", min_value=1.0, max_value=5.0, step=0.1, key="dei", label_visibility="collapsed")

    st.markdown('<div class="slider-label">🤝 الدعم الاجتماعي</div>', unsafe_allow_html=True)
    ss = st.slider(" ", min_value=1.0, max_value=5.0, step=0.1, key="ss", label_visibility="collapsed")

    # زر تنفيذ التنبؤ
    submitted = st.form_submit_button("🔍 تنفيذ التنبؤ")

# التنبؤ بالنتيجة وعرضها
if submitted:
    input_df = pd.DataFrame({
        'Digital_Resilience': [dr],
        'Digital_Emotional_Intelligence': [dei],
        'Social_Support': [ss]
    })

    prediction = model.predict(input_df)[0]
    prediction_rounded = round(prediction, 2)

    # تفسير النتيجة
    if prediction < 2.5:
        level = "🔴 منخفض"
        color = "red"
    elif prediction < 3.5:
        level = "🟠 متوسط"
        color = "orange"
    else:
        level = "🟢 مرتفع"
        color = "green"

    # عرض النتائج
    st.markdown("---")
    st.subheader("📈 النتيجة المتوقعة:")
    st.markdown(f"<h2 style='color:{color};'>التمكين النفسي الرقمي المتوقع: {prediction_rounded} ({level})</h2>", unsafe_allow_html=True)

    # عرض شريط تقدمي
    st.progress(min(prediction / 5.0, 1.0))

    # عرض جدول المدخلات
    st.markdown("### 🔍 تفاصيل الإدخال:")
    st.dataframe(input_df.style.format(precision=2))

# معلومات إضافية
st.markdown("---")
st.markdown("## ℹ️ عن النموذج")
st.write("""
طورت هذا النموذج/ سلوى سامي نسيم الباحثة بقسم علم النفس بكلية التربية - جامعة عين شمس، باستخدام خوارزميات تعلم الآلة القابلة للتفسير، وتم تدريبه على بيانات طلاب المرحلة الثانوية.

يعتمد النموذج على مدخلات:
- 💪 الصمود الرقمي  
- 🎯 الذكاء الانفعالي الرقمي  
- 🤝 الدعم الاجتماعي  
""")
