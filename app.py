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
 --navy:#10243A; --navy2:#183D55; --teal:#0D9488; --ink:#172B3A;
 --muted:#718096; --line:#E5EBF0; --canvas:#F4F7FA;
}
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;color:var(--ink);}
.stApp {background:var(--canvas);}
.block-container {max-width:1420px;padding-top:1.6rem;padding-bottom:3rem;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;color:var(--navy)!important;letter-spacing:-.03em;}
h1 {font-size:2.1rem!important;font-weight:800!important;}
h2 {font-size:1.45rem!important;font-weight:750!important;}
h3 {font-size:1.12rem!important;font-weight:750!important;}
[data-testid="stSidebar"] {background:#0D2034;border-right:1px solid #203B53;}
[data-testid="stSidebar"] * {color:#F4F8FB!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
 padding:.72rem .8rem;border-radius:12px;margin:.18rem 0;font-size:.91rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:rgba(255,255,255,.08);}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {display:none;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
 background:rgba(13,148,136,.24);border:1px solid rgba(94,234,212,.24);
}
[data-testid="stSidebar"] hr {border-color:#365067;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#B9C9D6!important;}
div[data-testid="stMetric"] {
 background:#fff;border:1px solid var(--line);padding:17px 19px;border-radius:16px;
 box-shadow:0 5px 18px rgba(16,36,58,.035);
}
div[data-testid="stMetricLabel"] {color:var(--muted)!important;}
div[data-testid="stMetricValue"] {color:var(--navy)!important;font-family:'Manrope',sans-serif;font-weight:800;}
.stButton>button,.stFormSubmitButton>button {
 background:var(--teal);color:white;border:1px solid var(--teal);border-radius:11px;
 font-weight:700;padding:.62rem 1rem;min-height:2.8rem;transition:all .15s ease;
}
.stButton>button:hover,.stFormSubmitButton>button:hover {background:#0F766E;border-color:#0F766E;color:white;}
.stTextInput input,.stNumberInput input,.stTextArea textarea,
.stSelectbox [data-baseweb="select"] {border-radius:10px!important;}
[data-testid="stForm"] {
 background:#fff;border:1px solid var(--line);border-radius:18px;
 padding:1.4rem 1.5rem;box-shadow:0 6px 22px rgba(16,36,58,.035);
}
div[data-testid="stAlert"] {border-radius:13px;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:13px;overflow:hidden;}
hr {border-color:var(--line);}
.eyebrow {color:var(--teal);font-size:.7rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:.4rem;}
.page-subtitle {color:var(--muted);font-size:.97rem;margin-top:-.4rem;margin-bottom:1.45rem;}
.hero {
 background:linear-gradient(115deg,#10243A 0%,#183D55 66%,#0F766E 100%);
 border-radius:20px;padding:1.65rem 1.8rem;color:white;margin:.35rem 0 1.2rem 0;
 box-shadow:0 12px 30px rgba(16,36,58,.12);
}
.hero h2 {color:white!important;margin:.2rem 0 .45rem 0;font-size:1.7rem!important;}
.hero p {color:#D5E3EB;margin:0;font-size:.95rem;max-width:850px;}
.hero .eyebrow {color:#7FE0D4;}
.section-card {background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;margin-bottom:.75rem;box-shadow:0 4px 16px rgba(16,36,58,.025);}
.section-card h4 {margin:0 0 .4rem 0;}
.small-muted {color:var(--muted);font-size:.86rem;}
.form-section {
 background:#F8FAFC;border:1px solid #E8EEF3;border-radius:14px;
 padding:1rem 1.1rem;margin:.55rem 0 .9rem 0;
}
.form-section-title {font-family:'Manrope',sans-serif;color:#10243A;font-weight:800;font-size:1rem;margin-bottom:.18rem;}
.form-section-help {color:#718096;font-size:.82rem;margin-bottom:.65rem;}
.field-help {color:#718096;font-size:.78rem;}
.footer-note {color:#8392A1;font-size:.78rem;border-top:1px solid var(--line);padding-top:.9rem;margin-top:2.2rem;}
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

    # Translate technical dataset names into clear form labels.
    field_labels = {
        "checking_status": "Bank account status",
        "duration": "Loan term (months)",
        "credit_history": "Previous credit repayment history",
        "purpose": "Purpose of the loan",
        "credit_amount": "Requested loan amount (dataset units)",
        "savings_status": "Savings level",
        "employment": "Length of current employment",
        "installment_commitment": "Installment burden (dataset scale: 1–4)",
        "personal_status": "Personal / marital status category",
        "other_parties": "Co-applicant or guarantor",
        "residence_since": "Years at current residence (dataset scale)",
        "property_magnitude": "Property or asset category",
        "age": "Age (years)",
        "other_payment_plans": "Other active payment plans",
        "housing": "Housing arrangement",
        "existing_credits": "Number of existing credits",
        "job": "Employment / occupation category",
        "num_dependents": "Number of dependants",
        "own_telephone": "Telephone ownership",
        "foreign_worker": "Foreign-worker category"
    }
    field_help = {
        "checking_status": "The account-balance category recorded in the benchmark dataset.",
        "duration": "Enter the requested repayment period in months.",
        "credit_history": "A summary category of the applicant's previous credit repayment record.",
        "purpose": "The reason the applicant is requesting the loan.",
        "credit_amount": "The amount field is in the benchmark dataset's original units, not a live currency quote.",
        "savings_status": "The savings category recorded in the benchmark dataset.",
        "employment": "How long the applicant has been employed.",
        "installment_commitment": "A coded benchmark measure. It is not a percentage or an affordability calculation.",
        "personal_status": "A coded demographic category from the original benchmark dataset.",
        "other_parties": "Whether another party is recorded on the credit agreement.",
        "residence_since": "A coded measure of how long the applicant has lived at the current address.",
        "property_magnitude": "The asset or property category recorded in the benchmark dataset.",
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
            "A11": "Overdrawn / below 0",
            "A12": "Low balance (0 to below 200)",
            "A13": "Balance of 200 or more",
            "A14": "No checking account"
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
            "A61": "Below 100", "A62": "100 to below 500",
            "A63": "500 to below 1,000", "A64": "1,000 or more",
            "A65": "Unknown / no savings account"
        },
        "employment": {
            "A71": "Unemployed", "A72": "Less than 1 year",
            "A73": "1 to less than 4 years", "A74": "4 to less than 7 years",
            "A75": "7 years or more"
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
                "Employment, savings and account information.",
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
                            if raw in descriptions:
                                return f"{descriptions[raw]}  ({raw})"
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
        "Generate and inspect a Gradient Boosting model classification."
    )
    borrower = st.session_state.borrower
    if borrower is None:
        st.info("Register a demonstration borrower before running an assessment.")
    else:
        st.markdown(
            f"<div class='section-card'><div class='eyebrow'>CURRENT APPLICANT</div>"
            f"<h3 style='margin:0'>{borrower['reference']}</h3>"
            f"<div class='small-muted'>Borrower details are stored for this "
            f"browser session only.</div></div>",
            unsafe_allow_html=True
        )
        if st.button("Run credit risk assessment", type="primary"):
            try:
                row = pd.DataFrame([borrower["values"]])[original_features]
                transformed = preprocessor.transform(row)
                transformed_df = pd.DataFrame(
                    transformed, columns=feature_names
                )
                prediction = int(gradient_model.predict(transformed_df)[0])
                probabilities = gradient_model.predict_proba(transformed_df)[0]
                bad_probability = float(probabilities[1])

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
        if assessment is not None:
            st.markdown("---")
            st.markdown("### Assessment result")
            st.caption(
                f"Applicant: {assessment['reference']} · "
                f"Generated: {assessment['created_at']}"
            )
            a, b = st.columns(2)
            a.metric(
                "Bad-credit class probability",
                f"{assessment['bad_probability']:.1%}"
            )
            b.metric(
                "Good-credit class probability",
                f"{assessment['good_probability']:.1%}"
            )
            if assessment["prediction"] == 1:
                st.error("Model classification: Higher risk")
            else:
                st.success("Model classification: Lower risk")
            st.progress(float(assessment["bad_probability"]))
            st.caption(
                "The class uses the model's default 0.50 decision threshold. "
                "Probabilities and classifications are benchmark-model outputs, "
                "not lending recommendations."
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
                "Feature": feature_names,
                "Contribution": assessment["shap_contributions"]
            })
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
            st.dataframe(
                shap_df[["Feature", "Contribution"]],
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
