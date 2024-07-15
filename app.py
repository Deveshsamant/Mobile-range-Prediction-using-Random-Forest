import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Mobile Price Range Prediction",
                   page_icon='📱',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            color: #000000; 
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border: none;
            border-radius: 5px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #000000; 
        }
        .stSlider label, .stSelectbox label {
            color: #000000 !important;
        }
        .stAlert.success {
            background-color: #006400 !important; 
            color: #000000 !important; 
        }
    </style>
    """, unsafe_allow_html=True)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


def predict_price(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

st.title("📱 Mobile Price Range Prediction")

st.subheader("Enter Mobile Specifications")

col1, col2 = st.columns(2)

with col1:
    battery_power = st.slider("Battery Power (mAh)", min_value=500, max_value=2000, step=1, help="Battery capacity in milliamp hours (mAh)")
    fc = st.slider("Front Camera (MP)", min_value=0, max_value=20, step=1, help="Front camera resolution in megapixels (MP)")
    int_memory = st.slider("Internal Memory (GB)", min_value=2, max_value=64, step=2, help="Internal memory in gigabytes (GB)")
    mobile_wt = st.slider("Mobile Weight (grams)", min_value=80, max_value=200, step=1, help="Weight of the mobile in grams")
    px_height = st.slider("Pixel Height", min_value=0, max_value=2000, step=1, help="Height of the screen in pixels")

with col2:
    dual_sim = st.selectbox("Dual Sim", [0, 1], help="Does the phone support dual SIM?")
    pc = st.slider("Primary Camera (MP)", min_value=0, max_value=20, step=1, help="Primary camera resolution in megapixels (MP)")
    px_width = st.slider("Pixel Width", min_value=0, max_value=2000, step=1, help="Width of the screen in pixels")
    ram = st.slider("RAM (MB)", min_value=256, max_value=8192, step=256, help="RAM in megabytes (MB)")
    talk_time = st.slider("Talk Time (hours)", min_value=2, max_value=20, step=1, help="Maximum talk time on a single charge")
    touch_screen = st.selectbox("Touch Screen", [0, 1], help="Does the phone have a touch screen?")
    wifi = st.selectbox("WiFi", [0, 1], help="Does the phone support WiFi?")

input_data = {
    'battery_power': battery_power,
    'dual_sim': dual_sim,
    'fc': fc,
    'int_memory': int_memory,
    'mobile_wt': mobile_wt,
    'pc': pc,
    'px_height': px_height,
    'px_width': px_width,
    'ram': ram,
    'talk_time': talk_time,
    'touch_screen': touch_screen,
    'wifi': wifi
}

if st.button("Predict"):
    with st.spinner('Predicting...'):
        prediction = predict_price(input_data)
    st.markdown(f'<div class="stAlert success"><b>The predicted price range is: {prediction}</b></div>', unsafe_allow_html=True)
