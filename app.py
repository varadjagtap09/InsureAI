
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Premium Lens | InsureAI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Change this whenever you want to update the conversion rate.
# This is an illustrative rate, not a live exchange rate.
USD_TO_INR = 85.0


# ==========================================
# CURRENCY FORMATTER
# ==========================================

def format_inr(amount):
    amount = float(amount)
    integer_part, decimal_part = f"{amount:.2f}".split(".")

    if len(integer_part) > 3:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]

        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]

        integer_part = ",".join(groups + [last_three])

    return f"₹{integer_part}.{decimal_part}"


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #081719;
    color: #e7f4f3;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4, p, label {
    color: #e7f4f3;
}

/* Main heading */

.main-title {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    color: #e8f5f3;
    margin-bottom: 0;
}

.subtitle {
    color: #8caeae;
    font-size: 15px;
    line-height: 1.7;
}

/* Top information banner */

.info-banner {
    background: #183537;
    border: 1px solid #24484a;
    padding: 15px 20px;
    border-radius: 14px;
    color: #c4dddd;
    font-size: 13px;
    margin-top: 25px;
    margin-bottom: 20px;
}

/* Cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #102426;
    border: 1px solid #244144;
    border-radius: 20px;
}

/* Labels */

div[data-testid="stWidgetLabel"] p {
    color: #e3f1ef !important;
    font-weight: 600;
    font-size: 14px;
}

/* Sliders */

div[data-testid="stSlider"] {
    padding-top: 5px;
    padding-bottom: 12px;
}

div[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 5px;
}

/* Inputs */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #193436;
    border: 1px solid #29494b;
    border-radius: 10px;
    color: white;
}

/* Radio buttons */

div[role="radiogroup"] {
    gap: 10px;
}

div[role="radiogroup"] label {
    background: #193436;
    border: 1px solid #29494b;
    border-radius: 10px;
    padding: 9px 15px;
}

/* Predict button */

.stFormSubmitButton button {
    background: linear-gradient(90deg, #32c6b1, #718cff);
    color: #061718;
    font-weight: 800;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 15px;
    transition: 0.2s;
}

.stFormSubmitButton button:hover {
    color: white;
    border: 1px solid #718cff;
}

/* Result amount */

.price {
    font-size: clamp(36px, 5vw, 65px);
    font-weight: 800;
    letter-spacing: -3px;
    background: linear-gradient(90deg, #e6f5f3, #8c9eff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
    overflow-wrap: anywhere;
}

.price-label {
    color: #9bbaba;
    font-size: 14px;
}

/* Result badges */

.pill {
    display: inline-block;
    background: #1b393b;
    color: #54d8c4;
    padding: 8px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
    margin-right: 5px;
}

/* ECG */

.ecg {
    width: 100%;
    height: 90px;
    margin-top: 25px;
}

.ecg path {
    fill: none;
    stroke: #8297ff;
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    filter: drop-shadow(0 0 5px #8297ff);
    stroke-dasharray: 1000;
    animation: heartbeat 4s linear infinite;
}

@keyframes heartbeat {
    from { stroke-dashoffset: 1000; }
    to { stroke-dashoffset: 0; }
}

/* Section headings */

.section-heading {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 15px;
    color: #e7f4f3;
}

.small-text {
    font-size: 12px;
    color: #88aaaa;
}

.footer {
    text-align: center;
    color: #668181;
    font-size: 12px;
    padding-top: 20px;
}

hr {
    border-color: #244144;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_files():

    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    return model, scaler, columns


try:
    model, scaler, columns = load_files()

except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()


# ==========================================
# HEADER
# ==========================================

header1, header2 = st.columns([1.2, 1])

with header1:
    st.markdown(
        '<div class="main-title">Premium Lens</div>',
        unsafe_allow_html=True
    )

with header2:
    st.markdown("""
    <div style="padding-top:15px; color:#8caeae; font-size:14px;">
    Explore how age, BMI, lifestyle and family details
    influence your predicted medical insurance charges.
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="info-banner">
    <b>INSUREAI MODEL</b> &nbsp; Your prediction is generated
    using your trained K-Nearest Neighbors regression model.
    Adjust the inputs and explore your estimated premium.
</div>
""", unsafe_allow_html=True)


# ==========================================
# MAIN LAYOUT
# ==========================================

left, right = st.columns([0.9, 1.25], gap="large")


# ==========================================
# INPUT PANEL
# ==========================================

with left:

    with st.container(border=True):

        st.markdown(
            '<div class="section-heading">Your Information</div>',
            unsafe_allow_html=True
        )

        st.caption("Adjust the details to calculate your estimate.")

        with st.form("insurance_form"):

            age = st.slider(
                "Age",
                min_value=18,
                max_value=100,
                value=25
            )

            bmi = st.slider(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1
            )

            bmi_category = (
                "Obese" if bmi > 29.9
                else "Non-obese"
            )

            st.markdown(
                f'<div class="small-text">BMI category: {bmi_category}</div>',
                unsafe_allow_html=True
            )

            st.write("")

            children = st.select_slider(
                "Number of Children",
                options=[0, 1, 2, 3, 4, 5],
                value=0
            )

            smoker = st.radio(
                "Smoker",
                ["No", "Yes"],
                horizontal=True
            )

            sex = st.radio(
                "Gender",
                ["Male", "Female"],
                horizontal=True
            )

            region = st.selectbox(
                "Region",
                [
                    "Northeast",
                    "Northwest",
                    "Southeast",
                    "Southwest"
                ]
            )

            st.write("")

            predict = st.form_submit_button(
                "✦  Calculate My Premium",
                use_container_width=True
            )


# ==========================================
# PREDICTION
# ==========================================

if predict:

    try:

        input_data = pd.DataFrame([{

            "smoker": 1 if smoker == "Yes" else 0,

            "region_southeast": (
                1 if region == "Southeast" else 0
            ),

            "sex": 0 if sex == "Male" else 1,

            "bmi_category_obese": (
                1 if bmi > 29.9 else 0
            ),

            "age": age,

            "bmi": bmi,

            "children": children

        }])

        # Match training columns
        input_data = input_data[columns]

        # Scale numerical columns only
        numeric_cols = ["age", "bmi", "children"]

        input_data[numeric_cols] = scaler.transform(
            input_data[numeric_cols]
        )

        # KNN prediction
        prediction_usd = model.predict(input_data)[0]

        # Convert USD to INR for display
        prediction_inr = prediction_usd * USD_TO_INR

        # Save result so it remains visible
        st.session_state["prediction_inr"] = prediction_inr
        st.session_state["prediction_usd"] = prediction_usd

        st.session_state["user_details"] = {
            "Age": age,
            "BMI": bmi,
            "Children": children,
            "Smoker": smoker,
            "Gender": sex,
            "Region": region
        }

    except Exception as e:

        st.error(f"Prediction error: {e}")


# ==========================================
# RESULT PANEL
# ==========================================

with right:

    with st.container(border=True):

        st.markdown(
            '<div class="section-heading">Your Estimated Premium</div>',
            unsafe_allow_html=True
        )

        if "prediction_inr" in st.session_state:

            prediction_inr = st.session_state["prediction_inr"]

            prediction_usd = st.session_state["prediction_usd"]

            st.markdown(
                f'<div class="price">{format_inr(prediction_inr)}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="price-label">Estimated insurance charges</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            <span class="pill">✦ KNN Prediction</span>
            <span class="pill">INR Display</span>
            """, unsafe_allow_html=True)

            # ECG visual
            st.markdown("""
            <svg class="ecg" viewBox="0 0 650 90"
                 preserveAspectRatio="none">
                <path d="
                M0 48
                L25 48 L35 44 L45 48
                L65 48 L75 10 L83 70 L92 48
                L120 48 L135 35 L150 48
                L220 48 L235 48 L245 42 L255 48
                L280 48 L290 12 L298 70 L307 48
                L340 48 L355 35 L370 48
                L440 48 L455 48 L465 42 L475 48
                L500 48 L510 12 L518 70 L527 48
                L555 48 L570 35 L585 48
                L650 48
                "/>
            </svg>
            """, unsafe_allow_html=True)

            st.divider()

            st.markdown(
                '<div class="section-heading">Prediction Details</div>',
                unsafe_allow_html=True
            )

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    "Estimated yearly cost",
                    format_inr(prediction_inr)
                )

            with col_b:
                st.metric(
                    "Estimated monthly cost",
                    format_inr(prediction_inr / 12)
                )

            st.markdown("""
            <p class="small-text">
            Currency conversion is applied only for display.
            Your original model prediction remains in USD.
            </p>
            """, unsafe_allow_html=True)

            with st.expander("View submitted information"):

                details = st.session_state["user_details"]

                for key, value in details.items():
                    st.write(f"**{key}:** {value}")

        else:

            st.markdown("""
            <div style="padding-top:35px; padding-bottom:35px;">
                <div style="font-size:55px;">₹</div>
                <div style="font-size:22px; font-weight:700;">
                    Your estimate will appear here
                </div>
                <p style="color:#88aaaa; font-size:13px;">
                    Enter your information and click
                    Calculate My Premium.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            st.markdown("**What you'll get**")

            st.write("✦ Predicted insurance charges")

            st.write("✦ Monthly and yearly estimates")

            st.write("✦ Your submitted information")


# ==========================================
# DISCLAIMER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <b>INSUREAI · PREMIUM LENS</b>
    <br><br>
    Built with Python · Streamlit · Scikit-learn
    <br>
    Educational ML project. Not an actual insurance quote.
    <br>
    Dataset-based estimates may not represent Indian insurance prices.
</div>
""", unsafe_allow_html=True)

