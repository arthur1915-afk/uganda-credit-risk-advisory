import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime

st.set_page_config(
    page_title="Uganda Credit Risk Advisory",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# Visual design
# -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
 --navy:#10243A; --teal:#0D9488; --ink:#172B3A; --muted:#718096;
 --line:#E7EDF2; --canvas:#F3F6F9; --white:#FFFFFF;
}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;color:var(--ink);}
.stApp {background:var(--canvas);}
.block-container {max-width:1480px;padding:1.55rem 2.2rem 3rem;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;color:var(--navy)!important;letter-spacing:-.035em;}
h1 {font-size:2.05rem!important;font-weight:800!important;line-height:1.15!important;}
h2 {font-size:1.45rem!important;font-weight:750!important;}
h3 {font-size:1.1rem!important;font-weight:750!important;}
p {line-height:1.55;}
[data-testid="stSidebar"] {background:#0B1F33;border-right:1px solid #1E3850;}
[data-testid="stSidebar"] * {color:#F4F8FB!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
 padding:.78rem .85rem;border-radius:12px;margin:.2rem 0;font-size:.92rem;
 transition:background .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:rgba(255,255,255,.08);}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
 background:rgba(13,148,136,.24);border:1px solid rgba(94,234,212,.24);
}
[data-testid="stSidebar"] hr {border-color:#365067;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#B9C9D6!important;}
div[data-testid="stMetric"] {
 background:#fff;border:1px solid var(--line);padding:1.1rem 1.2rem;border-radius:17px;
 box-shadow:0 7px 22px rgba(16,36,58,.035);min-height:112px;
}
div[data-testid="stMetricLabel"] {color:var(--muted)!important;font-size:.83rem;}
div[data-testid="stMetricValue"] {color:var(--navy)!important;font-family:'Manrope',sans-serif;font-weight:800;font-size:1.8rem;}
.stButton>button,.stFormSubmitButton>button {
 background:var(--teal);color:#fff;border:1px solid var(--teal);border-radius:11px;
 font-weight:700;padding:.66rem 1rem;min-height:2.8rem;transition:all .15s ease;
}
.stButton>button:hover,.stFormSubmitButton>button:hover {background:#0F766E;border-color:#0F766E;color:#fff;transform:translateY(-1px);}
.stTextInput input,.stNumberInput input,.stTextArea textarea,
.stSelectbox [data-baseweb="select"] {border-radius:10px!important;background:#fff;}
[data-testid="stForm"] {background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.5rem;box-shadow:0 8px 28px rgba(16,36,58,.035);}
div[data-testid="stAlert"] {border-radius:13px;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:14px;overflow:hidden;}
[data-baseweb="tab-list"] {gap:.5rem;border-bottom:1px solid var(--line);}
[data-baseweb="tab"] {border-radius:9px 9px 0 0;padding:.7rem 1rem;}
hr {border-color:var(--line);}
.eyebrow {color:var(--teal);font-size:.69rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem;}
.page-subtitle {color:var(--muted);font-size:.96rem;margin-top:-.35rem;margin-bottom:1.45rem;}
.hero {
 background:linear-gradient(112deg,#10243A 0%,#173B54 64%,#0D766F 100%);
 border-radius:21px;padding:1.8rem 2rem;color:#fff;margin:.35rem 0 1.25rem;
 box-shadow:0 14px 34px rgba(16,36,58,.13);position:relative;overflow:hidden;
}
.hero h2 {color:#fff!important;margin:.25rem 0 .5rem;font-size:1.8rem!important;}
.hero p {color:#D7E5EC;margin:0;font-size:.95rem;max-width:850px;}
.hero .eyebrow {color:#81E0D4;}
.section-card {background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.25rem;margin-bottom:.75rem;box-shadow:0 5px 18px rgba(16,36,58,.025);}
.section-card h4 {margin:0 0 .4rem;}
.small-muted {color:var(--muted);font-size:.86rem;}
.form-section {background:#F7FAFC;border:1px solid #E6EDF2;border-radius:14px;padding:1rem 1.15rem;margin:.65rem 0 .95rem;}
.form-section-title {font-family:'Manrope',sans-serif;color:#10243A;font-weight:800;font-size:1rem;margin-bottom:.18rem;}
.form-section-help {color:#718096;font-size:.83rem;}
.footer-note {color:#8392A1;font-size:.78rem;border-top:1px solid var(--line);padding-top:.9rem;margin-top:2.2rem;}
.risk-panel {background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.35rem 1.5rem;margin:.8rem 0 1.1rem;box-shadow:0 8px 24px rgba(16,36,58,.04);}
.risk-panel-top {display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1rem;}
.risk-number {font-family:'Manrope',sans-serif;font-size:2.45rem;line-height:1.05;font-weight:800;color:var(--navy);}
.risk-pill {background:#EAF3F3;color:#0F766E;border-radius:999px;padding:.45rem .8rem;font-size:.78rem;font-weight:800;}
.risk-track {height:13px;background:#E8EEF2;border-radius:99px;overflow:hidden;}
.risk-fill {height:100%;border-radius:99px;background:linear-gradient(90deg,#0D9488,#E2A34A);}
.risk-scale {display:flex;justify-content:space-between;color:#718096;font-size:.72rem;margin-top:.55rem;}
.step-strip {display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin:.9rem 0 1.2rem;}
.step-chip {background:#fff;border:1px solid var(--line);border-radius:12px;padding:.7rem .8rem;color:#718096;font-size:.78rem;}
.step-chip strong {display:block;color:var(--navy);font-size:.86rem;margin-bottom:.12rem;}
.explanation-callout {background:#F0F8F7;border-left:4px solid #0D9488;padding:.85rem 1rem;border-radius:0 12px 12px 0;margin:.6rem 0 1rem;color:#294B52;}
@media(max-width:700px){.risk-panel-top{align-items:flex-start;flex-direction:column}.step-strip{grid-template-columns:repeat(2,1fr)}}

[data-testid="stCaptionContainer"] {color:#7B8997;}
@media(max-width:900px) {
 .block-container {padding-left:1rem;padding-right:1rem;}
 .hero {padding:1.3rem;}
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource
def load_package():
    return joblib.load("credit_risk_model.joblib")

@st.cache_resource
def build_explainers(_package):
    model = _package["gradient_model"]
    explainer = shap.TreeExplainer(model)
    lime_explainer = LimeTabularExplainer(
        training_data=np.asarray(_package["lime_training_data"]),
        feature_names=list(_package["feature_names"]),
        class_names=["Good credit", "Bad credit"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )
    return explainer, lime_explainer

try:
    package = load_package()
    preprocessor = package["preprocessor"]
    gradient_model = package["gradient_model"]
    logistic_model = package["logistic_model"]
    feature_names = list(package["feature_names"])
    original_features = list(package["original_features"])
    numerical_features = list(preprocessor.transformers_[0][2])
    categorical_features = list(preprocessor.transformers_[1][2])
    categorical_encoder = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    categorical_options = dict(
        zip(categorical_features, categorical_encoder.categories_)
    )
    shap_explainer, lime_explainer = build_explainers(package)
except Exception as exc:
    st.error("The model package could not be loaded.")
    st.exception(exc)
    st.stop()

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
for key, default in {
    "borrower": None,
    "assessment": None,
    "history": [],
    "reviews": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------------------------------------------------
# Navigation and reusable page elements
# -------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='font-size:1.7rem;margin-bottom:0.2rem'>🏦</div>"
    "<div style='font-family:Manrope;font-size:1.08rem;font-weight:800'>"
    "UGANDA CREDIT RISK</div>"
    "<div style='font-size:0.76rem;color:#B9C9D6'>ADVISORY PLATFORM</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Dashboard",
        "Borrower Registration",
        "Credit Risk Assessment",
        "Explainable AI",
        "Credit Officer Review"
    ],
    format_func=lambda item: {
        "Dashboard": "▦  Dashboard",
        "Borrower Registration": "＋  Borrower registration",
        "Credit Risk Assessment": "◈  Risk assessment",
        "Explainable AI": "✳  Explainable AI",
        "Credit Officer Review": "✓  Officer review"
    }[item],
    label_visibility="visible"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.72rem;color:#B9C9D6'>"
    "ACADEMIC RESEARCH PROTOTYPE<br><b>Version 1.0</b><br><br>"
    "Ugandan financial-sector case study</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div style='font-size:0.7rem;color:#8FA5B6;margin-top:2rem'>"
    "Research demonstration only</div>",
    unsafe_allow_html=True
)

def page_heading(section, title, subtitle):
    st.markdown(
        f"<div class='eyebrow'>{section}</div>"
        f"<h1 style='margin:0'>{title}</h1>"
        f"<div class='page-subtitle'>{subtitle}</div>",
        unsafe_allow_html=True
    )

def research_notice():
    st.warning(
        "Research-only demonstration. This model was trained on the German "
        "Credit benchmark dataset, not Ugandan bank lending data. Do not use "
        "its outputs to approve, reject, price, or otherwise make real lending "
        "decisions. Enter fictional demonstration data only."
    )

def risk_label(prediction):
    return "Higher risk" if prediction == 1 else "Lower risk"

def readable_feature_name(name):
    """Turn encoded model feature names into more readable labels."""
    text = str(name).replace("num__", "").replace("cat__", "")
    text = text.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Model feature"

def show_risk_gauge(probability):
    """Display the model's bad-credit class probability as a visual gauge."""
    pct = max(0.0, min(100.0, float(probability) * 100))
    st.markdown(
        f"""
        <div class="risk-panel">
          <div class="risk-panel-top">
            <div><div class="eyebrow">MODEL PROBABILITY</div>
            <div class="risk-number">{pct:.1f}%</div>
            <div class="small-muted">Bad-credit class probability</div></div>
            <div class="risk-pill">{'Higher' if pct >= 50 else 'Lower'} risk class</div>
          </div>
          <div class="risk-track"><div class="risk-fill" style="width:{pct:.1f}%"></div></div>
          <div class="risk-scale"><span>0% · Lower</span><span>50% threshold</span><span>100% · Higher</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    "<div style='display:flex;justify-content:space-between;align-items:center;"
    "gap:1rem;padding:.15rem 0 .9rem;border-bottom:1px solid #E5EBF0;"
    "margin-bottom:1.2rem'>"
    "<div style='font-family:Manrope;font-weight:800;color:#10243A;"
    "font-size:.88rem;letter-spacing:.02em'>UGANDA CREDIT RISK ADVISORY</div>"
    "<div style='font-size:.74rem;color:#718096;background:#EAF3F3;"
    "padding:.35rem .7rem;border-radius:999px;font-weight:700'>"
    "ACADEMIC PROTOTYPE · v1.0</div></div>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------
if page == "Dashboard":
    page_heading(
        "OVERVIEW",
        "Credit risk dashboard",
        "A research workspace for machine-learning assessment and explainability."
    )
    st.markdown(
        "<div class='hero'>"
        "<div class='eyebrow'>RESEARCH PROTOTYPE · UGANDA</div>"
        "<h2>Clearer insights into credit risk.</h2>"
        "<p>Register a sample borrower, review the model's assessment and "
        "understand the factors behind its output.</p></div>",
        unsafe_allow_html=True
    )
    research_notice()

    total = len(st.session_state.history)
    higher = sum(1 for item in st.session_state.history
                 if item["Model classification"] == "Higher risk")
    lower = total - higher
    pending = sum(1 for item in st.session_state.reviews
                  if item["Status"] == "Pending review")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assessments this session", total)
    k2.metric("Higher-risk classifications", higher)
    k3.metric("Lower-risk classifications", lower)
    k4.metric("Pending reviews", pending)

    st.markdown("### Prototype workflow")
    steps = st.columns(4)
    workflow = [
        ("01", "Register", "Enter fictional borrower details."),
        ("02", "Assess", "Generate the benchmark model output."),
        ("03", "Explain", "Inspect SHAP and LIME explanations."),
        ("04", "Review", "Record a human review status.")
    ]
    for column, (number, title, description) in zip(steps, workflow):
        with column:
            st.markdown(
                f"<div class='section-card'>"
                f"<div class='eyebrow'>STEP {number}</div>"
                f"<h4>{title}</h4>"
                f"<div class='small-muted'>{description}</div></div>",
                unsafe_allow_html=True
            )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Recent assessment activity")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df.tail(8), use_container_width=True,
                         hide_index=True)
        else:
            st.info("No assessments have been run in this session yet.")
    with right:
        st.markdown("### Model components")
        st.markdown(
            "<div class='section-card'><h4>Gradient Boosting</h4>"
            "<div class='small-muted'>Primary model for borrower assessment "
            "and individual explanations.</div></div>"
            "<div class='section-card'><h4>Logistic Regression</h4>"
            "<div class='small-muted'>Retained in the package for model "
            "comparison.</div></div>"
            "<div class='section-card'><h4>SHAP + LIME</h4>"
            "<div class='small-muted'>Two complementary approaches to "
            "interpreting model behaviour.</div></div>",
            unsafe_allow_html=True
        )

# -------------------------------------------------------------------
# Borrower registration
# -------------------------------------------------------------------
elif page == "Borrower Registration":
    page_heading(
        "BORROWER INTAKE",
        "Register a sample borrower",
        "Complete the four short sections below. Use fictional information only."
    )

    st.markdown(
        "<div class='hero'>"
        "<div class='eyebrow'>GUIDED REGISTRATION</div>"
        "<h2>Let's build the applicant profile.</h2>"
        "<p>Enter the applicant's personal profile, employment and finances, "
        "credit history and requested loan details. The form uses fields from "
        "the German Credit research dataset.</p></div>",
        unsafe_allow_html=True
    )
    st.info(
        "Important: this is a research demonstration form, not a real Ugandan "
        "bank application. The benchmark dataset uses some categories and "
        "definitions that may differ from local banking practice."
    )

    st.markdown(
        "<div class='step-strip'>"
        "<div class='step-chip'><strong>01 · Personal</strong>Applicant profile</div>"
        "<div class='step-chip'><strong>02 · Financial</strong>Work and assets</div>"
        "<div class='step-chip'><strong>03 · Credit</strong>Repayment history</div>"
        "<div class='step-chip'><strong>04 · Loan</strong>Request details</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Translate technical dataset names into clear form labels.
    field_labels = {
        "checking_status": "Bank account situation",
        "duration": "Loan term (months)",
        "credit_history": "Previous credit repayment history",
        "purpose": "Purpose of the loan",
        "credit_amount": "Requested loan amount (dataset units)",
        "savings_status": "Savings amount category",
        "employment": "How long has the applicant been employed?",
        "installment_commitment": "Installment burden (dataset scale: 1–4)",
        "personal_status": "Personal / marital status category",
        "other_parties": "Co-applicant or guarantor",
        "residence_since": "Years at current residence (dataset scale)",
        "property_magnitude": "Property or other assets owned",
        "age": "Age (years)",
        "other_payment_plans": "Other active payment plans",
        "housing": "Housing arrangement",
        "existing_credits": "Number of existing credits",
        "job": "What kind of work does the applicant do?",
        "num_dependents": "Number of dependants",
        "own_telephone": "Telephone ownership",
        "foreign_worker": "Foreign-worker category"
    }
    field_help = {
        "checking_status": "Choose the range that best matches the account balance category shown. This is a coded benchmark category, not a live bank balance check.",
        "duration": "Enter the requested repayment period in months.",
        "credit_history": "A summary category of the applicant's previous credit repayment record.",
        "purpose": "The reason the applicant is requesting the loan.",
        "credit_amount": "The amount field is in the benchmark dataset's original units, not a live currency quote.",
        "savings_status": "Choose the savings range shown in the benchmark dataset. The options are categories, not an exact amount entry.",
        "employment": "How long the applicant has been employed.",
        "installment_commitment": "A coded benchmark measure. It is not a percentage or an affordability calculation.",
        "personal_status": "A coded demographic category from the original benchmark dataset.",
        "other_parties": "Whether another party is recorded on the credit agreement.",
        "residence_since": "A coded measure of how long the applicant has lived at the current address.",
        "property_magnitude": "Select the property or asset category recorded by the benchmark dataset. It does not calculate the value of assets.",
        "age": "Enter age in whole years.",
        "other_payment_plans": "Whether other payment arrangements are recorded.",
        "housing": "The applicant's housing category in the benchmark dataset.",
        "existing_credits": "Number of existing credits recorded in the dataset.",
        "job": "The occupation category used by the benchmark dataset.",
        "num_dependents": "Number of dependants recorded in the dataset.",
        "own_telephone": "Whether a telephone is recorded as owned.",
        "foreign_worker": "A category retained from the original benchmark dataset."
    }

    # Where the original dataset uses A-codes, show a readable description
    # while retaining the original code as the value sent to the model.
    category_descriptions = {
        "checking_status": {
            "0<=X<200": "Bank account balance: between 0 and 199",
            "<0": "Bank account is overdrawn",
            ">=200": "Bank account balance: 200 or more",
            "no checking": "No checking account",
        },
        "personal_status": {
            "male mar/wid": "Male, married or widowed",
            "male div/sep": "Male, divorced or separated",
            "female div/sep/mar": "Female, divorced, separated or married",
            "male single": "Male, single",
            "female single": "Female, single",
        },
        "employment": {
            "1<=X<4": "Employed for 1 to 3 years",
            "4<=X<7": "Employed for 4 to 6 years",
            ">=7": "Employed for 7 years or more",
            "<1": "Employed for less than 1 year",
            "unemployed": "Not currently employed",
        },
        "savings_status": {
            "100<=X<500": "Savings balance: between 100 and 499",
            "<100": "Savings balance: less than 100",
            "500<=X<1000": "Savings balance: between 500 and 999",
            ">=1000": "Savings balance: 1,000 or more",
            "no known savings": "No savings account / balance not recorded",
        },
        "job": {
            "high qualif/self emp/mgmt": "Highly qualified, self-employed or management role",
            "skilled": "Skilled employee",
            "unskilled resident": "Unskilled or entry-level work",
            "unemployed/unskilled non-resident": "Not employed or unskilled non-resident",
        },
        "checking_status": {
            "A11": "Account is overdrawn",
            "A12": "Small positive balance",
            "A13": "Higher account balance",
            "A14": "Does not have a checking account"
        },
        "credit_history": {
            "A30": "No previous credits / all paid",
            "A31": "Previous credits paid at this bank",
            "A32": "Existing credits repaid as agreed",
            "A33": "Previous delays in repayment",
            "A34": "Critical or other credit history"
        },
        "purpose": {
            "A40": "New car", "A41": "Used car", "A42": "Furniture or equipment",
            "A43": "Radio or television", "A44": "Domestic appliances",
            "A45": "Repairs", "A46": "Education", "A47": "Holiday",
            "A48": "Retraining", "A49": "Business", "A410": "Other"
        },
        "savings_status": {
            "A61": "Less than 100", "A62": "Between 100 and 499",
            "A63": "Between 500 and 999", "A64": "1,000 or more",
            "A65": "No savings account / balance not recorded"
        },
        "employment": {
            "A71": "Not currently employed", "A72": "Employed for less than 1 year",
            "A73": "Employed for 1–3 years", "A74": "Employed for 4–6 years",
            "A75": "Employed for 7 years or more"
        },
        "personal_status": {
            "A91": "Male: divorced or separated",
            "A92": "Female: divorced, separated or married",
            "A93": "Male: single", "A94": "Male: married or widowed",
            "A95": "Female: single"
        },
        "other_parties": {
            "A101": "No additional party", "A102": "Co-applicant",
            "A103": "Guarantor"
        },
        "property_magnitude": {
            "A121": "Real estate", "A122": "Building society savings / life insurance",
            "A123": "Car or other assets", "A124": "Unknown / no property"
        },
        "other_payment_plans": {
            "A141": "Bank", "A142": "Stores", "A143": "None"
        },
        "housing": {
            "A151": "Rent", "A152": "Own home", "A153": "Living free of rent"
        },
        "job": {
            "A171": "Unemployed / unskilled non-resident",
            "A172": "Unskilled resident", "A173": "Skilled employee / official",
            "A174": "Management / self-employed / highly qualified"
        },
        "own_telephone": {"A191": "No", "A192": "Yes"},
        "foreign_worker": {"A201": "Yes", "A202": "No"}
    }

    with st.form("borrower_registration"):
        st.markdown("#### Applicant reference")
        reference = st.text_input(
            "Application reference",
            value="UG-DEMO-001",
            help="Use a fictional reference such as UG-DEMO-001. Do not enter a real name, phone number or national ID."
        )
        values = {}

        sections = [
            (
                "1. Personal profile",
                "Basic details used by the benchmark model.",
                ["age", "personal_status", "housing", "residence_since",
                 "num_dependents", "own_telephone", "foreign_worker"]
            ),
            (
                "2. Employment and financial position",
                "Tell us about the applicant’s work, bank account, savings and assets.",
                ["employment", "job", "checking_status", "savings_status",
                 "property_magnitude"]
            ),
            (
                "3. Credit history",
                "Existing credit commitments and repayment background.",
                ["credit_history", "existing_credits", "other_parties",
                 "other_payment_plans", "installment_commitment"]
            ),
            (
                "4. Requested loan",
                "The requested amount, purpose and repayment period.",
                ["purpose", "credit_amount", "duration"]
            )
        ]

        defaults = {
            "duration": 24, "credit_amount": 2500,
            "installment_commitment": 2, "residence_since": 2,
            "age": 35, "existing_credits": 1, "num_dependents": 1
        }

        for section_title, section_description, features in sections:
            st.markdown(
                f"<div class='form-section'>"
                f"<div class='form-section-title'>{section_title}</div>"
                f"<div class='form-section-help'>{section_description}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            left, right = st.columns(2, gap="large")
            for index, feature in enumerate(features):
                column = left if index % 2 == 0 else right
                with column:
                    label = field_labels.get(
                        feature, feature.replace("_", " ").title()
                    )
                    help_text = field_help.get(feature, "")
                    if feature in ("installment_commitment", "residence_since"):
                        scale_help = {
                            "installment_commitment": {
                                1: "1 — 35% or more of disposable income",
                                2: "2 — 25% to below 35%",
                                3: "3 — 20% to below 25%",
                                4: "4 — Below 20%"
                            },
                            "residence_since": {
                                1: "1 — Less than 1 year",
                                2: "2 — 1 to less than 4 years",
                                3: "3 — 4 to less than 7 years",
                                4: "4 — 7 years or more"
                            }
                        }[feature]
                        values[feature] = st.selectbox(
                            label,
                            options=[1, 2, 3, 4],
                            index=max(0, min(3, defaults.get(feature, 1) - 1)),
                            format_func=lambda value, labels=scale_help: labels[value],
                            help=help_text,
                            key=f"input_{feature}"
                        )
                    elif feature in numerical_features:
                        minimum = 1 if feature in (
                            "duration", "credit_amount", "age"
                        ) else 0
                        default = max(minimum, defaults.get(feature, 1))
                        values[feature] = st.number_input(
                            label,
                            min_value=minimum,
                            value=default,
                            step=1,
                            help=help_text,
                            key=f"input_{feature}"
                        )
                    else:
                        options = list(categorical_options[feature])
                        descriptions = category_descriptions.get(feature, {})
                        def display_category(value, descriptions=descriptions):
                            raw = str(value)
                            normalized = raw.strip().lower()
                            if raw in descriptions:
                                return descriptions[raw]
                            if normalized in descriptions:
                                return descriptions[normalized]
                            return raw.replace("_", " ").replace("-", " ").title()
                        values[feature] = st.selectbox(
                            label,
                            options=options,
                            format_func=display_category,
                            help=help_text,
                            key=f"input_{feature}"
                        )

        st.markdown("---")
        confirm = st.checkbox(
            "I confirm this is fictional demonstration information.",
            value=False
        )
        submitted = st.form_submit_button(
            "Save applicant profile",
            use_container_width=True,
            disabled=False
        )

    if submitted:
        if not reference.strip():
            st.error("Please enter an application reference.")
        elif not confirm:
            st.error("Please confirm that the information is fictional.")
        else:
            st.session_state.borrower = {
                "reference": reference.strip(),
                "values": values
            }
            st.session_state.assessment = None
            st.success("Applicant profile saved successfully.")
            st.markdown(
                f"<div class='section-card'><div class='eyebrow'>PROFILE SAVED</div>"
                f"<h3 style='margin:0'>{reference.strip()}</h3>"
                f"<div class='small-muted'>You can now open Credit Risk Assessment "
                f"to run the model.</div></div>",
                unsafe_allow_html=True
            )

# -------------------------------------------------------------------
# Credit risk assessment
# -------------------------------------------------------------------
elif page == "Credit Risk Assessment":
    page_heading(
        "MODEL ASSESSMENT",
        "Credit risk assessment",
        "Review the sample applicant, run the benchmark model and inspect its output."
    )
    research_notice()
    borrower = st.session_state.borrower

    if borrower is None:
        st.markdown(
            "<div class='section-card'><div class='eyebrow'>NEXT STEP</div>"
            "<h3>Start with a borrower profile</h3>"
            "<div class='small-muted'>Register a fictional applicant before running "
            "the model assessment.</div></div>",
            unsafe_allow_html=True
        )
    else:
        profile = borrower["values"]
        st.markdown(
            f"<div class='section-card'><div class='eyebrow'>ASSESSMENT FILE</div>"
            f"<h3 style='margin:0'>{borrower['reference']}</h3>"
            f"<div class='small-muted'>Created from a fictional applicant profile · "
            f"Session-only record</div></div>",
            unsafe_allow_html=True
        )

        st.markdown("### Applicant snapshot")
        snap1, snap2, snap3, snap4 = st.columns(4)
        snap1.metric("Age", f"{int(profile.get('age', 0))} years")
        snap2.metric("Loan term", f"{int(profile.get('duration', 0))} months")
        snap3.metric(
            "Requested amount",
            f"{profile.get('credit_amount', '—'):,}" if isinstance(
                profile.get("credit_amount"), (int, float, np.integer, np.floating)
            ) else str(profile.get("credit_amount", "—"))
        )
        snap4.metric("Existing credits", str(profile.get("existing_credits", "—")))
        st.caption(
            "Requested amount is expressed in the German Credit dataset's original "
            "units. It is not an amount in UGX or a live financial value."
        )

        with st.expander("Review full applicant information", expanded=False):
            profile_rows = []
            for feature in original_features:
                value = profile.get(feature, "—")
                profile_rows.append({
                    "Applicant information": readable_feature_name(feature),
                    "Recorded value": str(value)
                })
            st.dataframe(
                pd.DataFrame(profile_rows),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("### Run assessment")
        run_col, info_col = st.columns([1, 2])
        with run_col:
            run_assessment = st.button(
                "Run model assessment",
                type="primary",
                use_container_width=True
            )
        with info_col:
            st.markdown(
                "<div class='small-muted' style='padding:.65rem 0'>"
                "The Gradient Boosting model is the primary model. "
                "The assessment also records a Logistic Regression comparison.</div>",
                unsafe_allow_html=True
            )

        if run_assessment:
            try:
                row = pd.DataFrame([profile])[original_features]
                transformed = preprocessor.transform(row)
                transformed_df = pd.DataFrame(
                    transformed, columns=feature_names
                )

                prediction = int(gradient_model.predict(transformed_df)[0])
                probabilities = gradient_model.predict_proba(transformed_df)[0]
                bad_probability = float(probabilities[1])

                # Secondary model comparison for the research prototype.
                logistic_prediction = int(logistic_model.predict(transformed_df)[0])
                logistic_probabilities = logistic_model.predict_proba(transformed_df)[0]

                shap_result = shap_explainer(transformed_df)
                shap_values = np.asarray(shap_result.values)
                if shap_values.ndim == 3:
                    shap_contributions = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    shap_contributions = shap_values[0]
                else:
                    raise ValueError(
                        f"Unexpected SHAP output shape: {shap_values.shape}"
                    )

                lime_result = lime_explainer.explain_instance(
                    data_row=transformed_df.iloc[0].to_numpy(),
                    predict_fn=gradient_model.predict_proba,
                    num_features=min(10, len(feature_names)),
                    num_samples=3000,
                    labels=(1,)
                )

                st.session_state.assessment = {
                    "reference": borrower["reference"],
                    "prediction": prediction,
                    "bad_probability": bad_probability,
                    "good_probability": float(probabilities[0]),
                    "logistic_prediction": logistic_prediction,
                    "logistic_bad_probability": float(logistic_probabilities[1]),
                    "shap_contributions": shap_contributions,
                    "lime_result": lime_result,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.history.append({
                    "Reference": borrower["reference"],
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Model classification": risk_label(prediction),
                    "Bad-credit model probability": f"{bad_probability:.1%}"
                })
            except Exception as exc:
                st.error("Assessment could not be completed.")
                st.exception(exc)

        assessment = st.session_state.assessment
        if assessment is not None and assessment["reference"] == borrower["reference"]:
            st.markdown("---")
            st.markdown("## Assessment intelligence report")
            st.caption(
                f"Reference: {assessment['reference']}  ·  "
                f"Generated: {assessment['created_at']}  ·  "
                "Primary model: Gradient Boosting"
            )

            # Input profile checks are descriptive checks, not a credit score.
            required_fields = list(original_features)
            missing_fields = [
                field for field in required_fields
                if field not in profile or profile[field] is None
                or (isinstance(profile[field], str) and not profile[field].strip())
            ]
            completeness = (
                100 * (len(required_fields) - len(missing_fields))
                / max(1, len(required_fields))
            )
            model_agreement = (
                assessment["prediction"] == assessment["logistic_prediction"]
            )
            threshold_distance = abs(assessment["bad_probability"] - 0.5)
            threshold_status = (
                "Close to classification threshold"
                if threshold_distance < 0.10
                else "Further from classification threshold"
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Profile completeness", f"{completeness:.0f}%")
            m2.metric(
                "Model agreement",
                "Yes" if model_agreement else "No"
            )
            m3.metric(
                "Distance from 50% threshold",
                f"{threshold_distance:.1%}"
            )
            m4.metric(
                "Fields supplied",
                f"{len(required_fields) - len(missing_fields)}/{len(required_fields)}"
            )

            if missing_fields:
                st.warning(
                    "Some expected profile fields are missing: "
                    + ", ".join(readable_feature_name(x) for x in missing_fields)
                )
            else:
                st.success(
                    "All expected input fields are present. This checks completeness "
                    "only; it does not establish that the information is accurate."
                )

            if threshold_distance < 0.10:
                st.warning(
                    "This output is relatively close to the model's 0.50 classification "
                    "threshold. Small changes in input or model behaviour may change "
                    "the predicted class."
                )

            result_tab, compare_tab, drivers_tab, technical_tab = st.tabs([
                "Assessment result",
                "Model comparison",
                "Key contributing factors",
                "Technical details"
            ])

            with result_tab:
                left_result, right_result = st.columns([1.15, 1])
                with left_result:
                    st.markdown("### Primary model classification")
                    if assessment["prediction"] == 1:
                        st.error("Gradient Boosting: Higher-risk class")
                    else:
                        st.success("Gradient Boosting: Lower-risk class")
                    show_risk_gauge(assessment["bad_probability"])
                    st.caption(
                        "The classification uses a 0.50 threshold on the model's "
                        "predicted bad-credit class probability."
                    )
                with right_result:
                    st.markdown("### Probability distribution")
                    probability_df = pd.DataFrame({
                        "Class": ["Good-credit class", "Bad-credit class"],
                        "Model probability": [
                            assessment["good_probability"],
                            assessment["bad_probability"]
                        ]
                    }).set_index("Class")
                    st.bar_chart(probability_df, horizontal=True)
                    st.caption(
                        "These are model class probabilities, not calibrated estimates "
                        "of an individual Ugandan applicant's real-world default risk."
                    )

                st.markdown("### Assessment interpretation")
                if assessment["prediction"] == 1:
                    st.markdown(
                        "<div class='explanation-callout'><b>Model output:</b> "
                        "The Gradient Boosting model assigns this sample to its "
                        "bad-credit class. This is a statistical classification, "
                        "not a loan rejection recommendation.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div class='explanation-callout'><b>Model output:</b> "
                        "The Gradient Boosting model assigns this sample to its "
                        "good-credit class. This is a statistical classification, "
                        "not a loan approval recommendation.</div>",
                        unsafe_allow_html=True
                    )

            with compare_tab:
                st.markdown("### Compare the two trained models")
                comparison_df = pd.DataFrame({
                    "Model": ["Gradient Boosting", "Logistic Regression"],
                    "Predicted class": [
                        risk_label(assessment["prediction"]),
                        risk_label(assessment["logistic_prediction"])
                    ],
                    "Bad-credit class probability": [
                        assessment["bad_probability"],
                        assessment["logistic_bad_probability"]
                    ]
                })
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Bad-credit class probability": st.column_config.ProgressColumn(
                            "Bad-credit class probability",
                            min_value=0,
                            max_value=1,
                            format="%.1%%"
                        )
                    }
                )
                comparison_chart = pd.DataFrame({
                    "Model": ["Gradient Boosting", "Logistic Regression"],
                    "Bad-credit probability": [
                        assessment["bad_probability"],
                        assessment["logistic_bad_probability"]
                    ]
                }).set_index("Model")
                st.bar_chart(comparison_chart, horizontal=True)
                if model_agreement:
                    st.success(
                        "Both models return the same class for this sample. "
                        "Their probability estimates are not identical."
                    )
                else:
                    st.warning(
                        "The models disagree on this sample. This highlights why "
                        "model comparison and human review matter in a credit-risk "
                        "research workflow."
                    )

            with drivers_tab:
                st.markdown("### What contributed to the primary model output?")
                contribution_df = pd.DataFrame({
                    "Feature": [
                        readable_feature_name(name) for name in feature_names
                    ],
                    "SHAP contribution": assessment["shap_contributions"]
                })
                contribution_df["Direction"] = np.where(
                    contribution_df["SHAP contribution"] >= 0,
                    "Toward bad-credit class",
                    "Toward good-credit class"
                )
                contribution_df["Magnitude"] = contribution_df[
                    "SHAP contribution"
                ].abs()
                strongest = contribution_df.nlargest(8, "Magnitude").sort_values(
                    "SHAP contribution"
                )
                st.bar_chart(
                    strongest.set_index("Feature")[["SHAP contribution"]],
                    horizontal=True
                )
                risk_factors = contribution_df[
                    contribution_df["SHAP contribution"] > 0
                ].nlargest(3, "Magnitude")
                supportive_factors = contribution_df[
                    contribution_df["SHAP contribution"] < 0
                ].nsmallest(3, "SHAP contribution")

                factor_left, factor_right = st.columns(2)
                with factor_left:
                    st.markdown("#### Pushes toward bad-credit class")
                    if risk_factors.empty:
                        st.caption("No positive SHAP contributions in this sample.")
                    else:
                        st.dataframe(
                            risk_factors[["Feature", "SHAP contribution"]],
                            use_container_width=True,
                            hide_index=True
                        )
                with factor_right:
                    st.markdown("#### Pushes toward good-credit class")
                    if supportive_factors.empty:
                        st.caption("No negative SHAP contributions in this sample.")
                    else:
                        st.dataframe(
                            supportive_factors[["Feature", "SHAP contribution"]],
                            use_container_width=True,
                            hide_index=True
                        )
                st.caption(
                    "SHAP contributions describe how features influence this model "
                    "output. They are not causal effects or percentage-point changes "
                    "in probability."
                )

            with technical_tab:
                st.markdown("### Machine-learning pipeline")
                pipeline_steps = [
                    ("01 · Input", "Fictional borrower profile collected through the form."),
                    ("02 · Preprocessing", "Missing-value imputation, numerical scaling and categorical encoding."),
                    ("03 · Prediction", "Gradient Boosting produces class probabilities and a predicted class."),
                    ("04 · Comparison", "Logistic Regression provides a second model output for comparison."),
                    ("05 · Explainability", "SHAP and LIME generate local explanations of model behaviour.")
                ]
                for step_title, step_description in pipeline_steps:
                    st.markdown(
                        f"<div class='section-card'><div class='eyebrow'>{step_title}</div>"
                        f"<div>{step_description}</div></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("#### Run metadata")
                st.json({
                    "primary_model": "GradientBoostingClassifier",
                    "comparison_model": "LogisticRegression",
                    "explainability_methods": ["SHAP", "LIME"],
                    "classification_threshold": 0.50,
                    "assessment_timestamp": assessment["created_at"],
                    "input_feature_count": len(required_fields),
                    "profile_completeness_percent": round(completeness, 1)
                })

            st.warning(
                "Research interpretation only. The models were trained using the "
                "German Credit benchmark dataset, not validated Ugandan bank lending "
                "data. This prototype must not be used to approve, reject, rank or "
                "price real loan applications."
            )
# -------------------------------------------------------------------
# Explainable AI
# -------------------------------------------------------------------
elif page == "Explainable AI":
    page_heading(
        "MODEL TRANSPARENCY",
        "Explainable AI",
        "Explore the features contributing to the current model output."
    )
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("Run a borrower assessment first to generate explanations.")
    else:
        st.markdown(
            f"<div class='section-card'><div class='eyebrow'>EXPLANATION SUBJECT</div>"
            f"<h3 style='margin:0'>{assessment['reference']}</h3>"
            f"<div class='small-muted'>Model classification: "
            f"{risk_label(assessment['prediction'])}</div></div>",
            unsafe_allow_html=True
        )
        shap_tab, lime_tab, notes_tab = st.tabs(
            ["SHAP contributions", "LIME explanation", "Interpretation notes"]
        )
        with shap_tab:
            st.markdown("### Which features influenced this output?")
            shap_df = pd.DataFrame({
                "Feature": [readable_feature_name(name) for name in feature_names],
                "Contribution": assessment["shap_contributions"]
            })
            shap_df["Direction"] = np.where(
                shap_df["Contribution"] >= 0,
                "Pushes toward bad-credit class",
                "Pushes toward good-credit class"
            )
            shap_df["Absolute contribution"] = shap_df["Contribution"].abs()
            shap_df = shap_df.nlargest(
                12, "Absolute contribution"
            ).sort_values("Contribution")
            st.bar_chart(
                shap_df.set_index("Feature")[["Contribution"]],
                horizontal=True
            )
            st.caption(
                "Positive values push the model output toward the bad-credit "
                "class; negative values push it toward the good-credit class."
            )
            strongest = shap_df.reindex(
                shap_df["Absolute contribution"].sort_values(ascending=False).index
            ).head(3)
            if not strongest.empty:
                summaries = []
                for _, item in strongest.iterrows():
                    direction = "increased" if item["Contribution"] >= 0 else "decreased"
                    summaries.append(
                        f"<li><b>{item['Feature']}</b> {direction} the model output "
                        f"toward the bad-credit class.</li>"
                    )
                st.markdown(
                    "<div class='explanation-callout'><b>In plain language</b><ul>"
                    + "".join(summaries) + "</ul></div>",
                    unsafe_allow_html=True
                )
            st.dataframe(
                shap_df[["Feature", "Direction", "Contribution"]],
                use_container_width=True,
                hide_index=True
            )
        with lime_tab:
            st.markdown("### Local model approximation")
            lime_items = assessment["lime_result"].as_list(label=1)
            lime_df = pd.DataFrame(
                lime_items,
                columns=["Feature condition", "Local contribution"]
            )
            st.caption(
                "LIME describes local conditions in the model's encoded feature space. "
                "Some conditions may look technical because categorical inputs are "
                "expanded into model features."
            )
            st.dataframe(
                lime_df, use_container_width=True, hide_index=True
            )
            st.metric(
                "LIME local fidelity",
                f"{assessment['lime_result'].score:.3f}"
            )
        with notes_tab:
            st.markdown("### How to interpret these explanations")
            st.markdown(
                "- SHAP values represent feature contributions in the model "
                "output space; they are not percentage-point changes in "
                "predicted probability.\n"
                "- LIME approximates model behaviour around this individual "
                "applicant. Its fidelity score is not overall model accuracy.\n"
                "- SHAP and LIME can produce different feature rankings because "
                "they use different explanation methods.\n"
                "- These explanations describe model behaviour, not causal "
                "effects or proof that a factor is fair."
            )
            st.info(
                "The benchmark dataset and its encoded features may not fully "
                "represent realistic Ugandan borrower circumstances."
            )

# -------------------------------------------------------------------
# Credit officer review
# -------------------------------------------------------------------
elif page == "Credit Officer Review":
    page_heading(
        "HUMAN OVERSIGHT",
        "Credit officer review",
        "Record a reviewer assessment alongside the model output."
    )
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("Generate an assessment before recording a review.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div class='section-card'><div class='eyebrow'>APPLICANT</div>"
                f"<h3 style='margin:0'>{assessment['reference']}</h3>"
                f"<div class='small-muted'>Research demonstration record</div></div>",
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"<div class='section-card'><div class='eyebrow'>MODEL OUTPUT</div>"
                f"<h3 style='margin:0'>{risk_label(assessment['prediction'])}</h3>"
                f"<div class='small-muted'>Bad-credit class probability: "
                f"{assessment['bad_probability']:.1%}</div></div>",
                unsafe_allow_html=True
            )
        st.warning(
            "This panel records a research demonstration review only. "
            "It does not authorise or support a real credit decision."
        )
        with st.form("officer_review"):
            officer = st.text_input(
                "Demonstration reviewer",
                help="Use a fictional reviewer name or ID."
            )
            status = st.selectbox(
                "Review status",
                [
                    "Pending review",
                    "Further information required",
                    "Reviewed — additional assessment needed",
                    "Review completed"
                ]
            )
            notes = st.text_area("Review notes")
            save_review = st.form_submit_button(
                "Save review", use_container_width=True
            )
        if save_review:
            if not officer.strip():
                st.error("Enter a reviewer name or fictional reviewer ID.")
            else:
                st.session_state.reviews.append({
                    "Reference": assessment["reference"],
                    "Reviewer": officer.strip(),
                    "Status": status,
                    "Notes": notes,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("Review saved in this browser session.")

    if st.session_state.reviews:
        st.markdown("### Review activity")
        st.dataframe(
            pd.DataFrame(st.session_state.reviews),
            use_container_width=True,
            hide_index=True
        )

# -------------------------------------------------------------------
# Persistent page footer
# -------------------------------------------------------------------
st.markdown(
    "<div class='footer-note'>"
    "<b>Uganda Credit Risk Advisory</b> · Academic research prototype · "
    "Version 1.0<br>Research demonstration only. Session records are not "
    "persistent storage. No real borrower data should be entered."
    "</div>",
    unsafe_allow_html=True
)
