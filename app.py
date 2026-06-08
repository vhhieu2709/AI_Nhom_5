import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
import mediapipe as mp
from PIL import Image
import io

# ── Cấu hình trang ──────────────────────────────────────────
st.set_page_config(
    page_title="ASL Hand Sign Recognition",
    page_icon="🤟",
    layout="wide",
)

# ── CSS tuỳ chỉnh ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background: #0d0d0d; color: #f0f0f0; }

.result-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #00f5a0;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 16px;
}
.result-letter {
    font-family: 'Space Mono', monospace;
    font-size: 80px;
    color: #00f5a0;
    line-height: 1;
}
.result-conf {
    font-size: 18px;
    color: #aaa;
    margin-top: 8px;
}
.model-badge {
    display: inline-block;
    background: #00f5a020;
    border: 1px solid #00f5a0;
    color: #00f5a0;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 12px;
}
.compare-col {
    background: #111;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #222;
}
</style>
""", unsafe_allow_html=True)

# ── Class names (A-Z trừ J, Z) ──────────────────────────────
CLASS_NAMES = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
IMG_SIZE = (224, 224)

# ── Load models ─────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("model_A_final.keras")

@st.cache_resource
def load_xgb_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

# ── MediaPipe setup ─────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def extract_landmarks(img_rgb):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            coords = []
            for l in lm:
                coords.extend([l.x, l.y, l.z])
            return np.array(coords).reshape(1, -1), result.multi_hand_landmarks[0]
    return None, None

def draw_landmarks_on_image(img_rgb, landmarks):
    img_copy = img_rgb.copy()
    h, w, _ = img_copy.shape
    img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    mp_draw.draw_landmarks(img_bgr,
                           landmarks,
                           mp_hands.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(0, 245, 160), thickness=2, circle_radius=4),
                           mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── Predict functions ────────────────────────────────────────
def predict_cnn(img_pil):
    model = load_cnn_model()
    img = img_pil.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argmax(probs)
    return CLASS_NAMES[idx], probs[idx], probs

def predict_xgb(img_pil):
    model = load_xgb_model()
    img_rgb = np.array(img_pil.convert("RGB"))
    features, landmarks = extract_landmarks(img_rgb)
    if features is None:
        return None, None, None, None
    probs = model.predict_proba(features)[0]
    idx   = np.argmax(probs)
    img_annotated = draw_landmarks_on_image(img_rgb, landmarks)
    return CLASS_NAMES[idx], probs[idx], probs, img_annotated

# ── Hàm xử lý ảnh đầu vào ───────────────────────────────────
def load_image_from_source(source):
    """Nhận bytes hoặc PIL, trả về PIL RGB"""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(io.BytesIO(source.read())).convert("RGB")

# ── UI Header ────────────────────────────────────────────────
st.markdown("# 🤟 ASL Hand Sign Recognition")
st.markdown("Nhận diện ngôn ngữ ký hiệu tay (A–Z) bằng CNN & MediaPipe+XGBoost")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    mode = st.radio("**Chế độ nhận diện**",
                    ["🧠 CNN (MobileNetV2)", "📐 MediaPipe + XGBoost", "⚡ So sánh cả hai"],
                    index=0)
    st.divider()
    input_src = st.radio("**Nguồn ảnh**", ["📁 Upload ảnh", "📷 Chụp webcam"])
    st.divider()
    st.markdown("**Về các model:**")
    st.markdown("""
- **CNN**: MobileNetV2 fine-tuned, nhận diện đặc trưng thị giác
- **XGBoost**: MediaPipe trích xuất 63 landmarks → XGBoost phân loại
    """)

# ── Input ────────────────────────────────────────────────────
img_pil = None

if input_src == "📁 Upload ảnh":
    uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_pil = load_image_from_source(uploaded)
        st.image(img_pil, caption="Ảnh đã tải lên", width=320)

else:  # Webcam
    img_bytes = st.camera_input("📷 Chụp ảnh bàn tay")
    if img_bytes:
        img_pil = load_image_from_source(img_bytes)

# ── Predict ──────────────────────────────────────────────────
if img_pil is not None:
    st.divider()

    # ── Chế độ CNN ──────────────────────────────────────────
    if mode == "🧠 CNN (MobileNetV2)":
        with st.spinner("Đang nhận diện..."):
            letter, conf, probs = predict_cnn(img_pil)
        st.markdown(f'<div class="model-badge">CNN · MobileNetV2</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box">
            <div class="result-letter">{letter}</div>
            <div class="result-conf">Độ tin cậy: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📊 Top 5 dự đoán"):
            top5 = np.argsort(probs)[::-1][:5]
            for i in top5:
                st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}  —  {probs[i]*100:.1f}%")

    # ── Chế độ XGBoost ──────────────────────────────────────
    elif mode == "📐 MediaPipe + XGBoost":
        with st.spinner("Đang trích xuất landmarks..."):
            letter, conf, probs, img_ann = predict_xgb(img_pil)

        st.markdown(f'<div class="model-badge">MediaPipe · XGBoost</div>', unsafe_allow_html=True)
        if letter is None:
            st.warning("⚠️ Không phát hiện được bàn tay trong ảnh. Hãy thử ảnh khác.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_pil, caption="Ảnh gốc", use_column_width=True)
            with col2:
                st.image(img_ann, caption="Landmarks MediaPipe", use_column_width=True)

            st.markdown(f"""
            <div class="result-box">
                <div class="result-letter">{letter}</div>
                <div class="result-conf">Độ tin cậy: {conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📊 Top 5 dự đoán"):
                top5 = np.argsort(probs)[::-1][:5]
                for i in top5:
                    st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}  —  {probs[i]*100:.1f}%")

    # ── Chế độ So sánh ──────────────────────────────────────
    else:
        with st.spinner("Đang chạy cả hai model..."):
            letter_cnn, conf_cnn, probs_cnn           = predict_cnn(img_pil)
            letter_xgb, conf_xgb, probs_xgb, img_ann = predict_xgb(img_pil)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="compare-col">', unsafe_allow_html=True)
            st.markdown(f'<div class="model-badge">CNN · MobileNetV2</div>', unsafe_allow_html=True)
            if letter_cnn:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-letter">{letter_cnn}</div>
                    <div class="result-conf">{conf_cnn*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                with st.expander("Top 5"):
                    for i in np.argsort(probs_cnn)[::-1][:5]:
                        st.progress(float(probs_cnn[i]), text=f"{CLASS_NAMES[i]} {probs_cnn[i]*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="compare-col">', unsafe_allow_html=True)
            st.markdown(f'<div class="model-badge">MediaPipe · XGBoost</div>', unsafe_allow_html=True)
            if letter_xgb is None:
                st.warning("⚠️ Không phát hiện bàn tay")
            else:
                if img_ann is not None:
                    st.image(img_ann, caption="Landmarks", use_column_width=True)
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-letter">{letter_xgb}</div>
                    <div class="result-conf">{conf_xgb*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                with st.expander("Top 5"):
                    for i in np.argsort(probs_xgb)[::-1][:5]:
                        st.progress(float(probs_xgb[i]), text=f"{CLASS_NAMES[i]} {probs_xgb[i]*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Nhận xét tổng hợp
        if letter_cnn and letter_xgb:
            st.divider()
            if letter_cnn == letter_xgb:
                st.success(f"✅ Cả hai model đều dự đoán: **{letter_cnn}**")
            else:
                st.info(f"ℹ️ CNN dự đoán **{letter_cnn}** ({conf_cnn*100:.1f}%) · XGBoost dự đoán **{letter_xgb}** ({conf_xgb*100:.1f}%)")
