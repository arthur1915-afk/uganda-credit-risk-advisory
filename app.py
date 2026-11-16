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
    --navy: #10243A;
    --navy-2: #183650;
    --teal: #159A91;
    --teal-soft: #E7F5F3;
    --ink: #172B3A;
    --muted: #718096;
    --line: #E6ECF1;
    --surface: #FFFFFF;
    --canvas: #F5F8FA;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
.stApp { background: var(--canvas); }
.block-container {
    padding-top: 1.35rem;
    padding-bottom: 2.2rem;
    max-width: 1380px;
}
h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--navy) !important;
    letter-spacing: -0.025em;
}
h1 { font-weight: 800 !important; font-size: 2.05rem !important; line-height: 1.16 !important; }
h2, h3 { font-weight: 700 !important; }
h2 { font-size: 1.45rem !important; }
h3 { font-size: 1.12rem !important; }
p, [data-testid="stMarkdownContainer"] { line-height: 1.55; }
[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 1px solid #203B53;
}
[data-testid="stSidebar"] * { color: #F4F8FB !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.68rem 0.75rem;
    border-radius: 10px;
    margin: 0.16rem 0;
    font-size: 0.91rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.09);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(21,154,145,0.22);
    border: 1px solid rgba(127,224,212,0.24);
}
[data-testid="stSidebar"] hr { border-color: #365067; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B9C9D6 !important;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    padding: 14px 16px;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(16,36,58,0.025);
}
div[data-testid="stMetricValue"] { font-size: 1.8rem; }
div[data-testid="stMetricLabel"] { color: var(--muted) !important; }
div[data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}
.stButton > button, .stFormSubmitButton > button {
    background: var(--teal);
    color: white;
    border: 1px solid var(--teal);
    border-radius: 9px;
    font-weight: 700;
    padding: 0.58rem 1rem;
    min-height: 2.7rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #117F78;
    border-color: #117F78;
    color: white;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox [data-baseweb="select"] {
    border-radius: 9px;
}
[data-testid="stForm"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
}
div[data-testid="stAlert"] { border-radius: 11px; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
hr { border-color: var(--line); }
.eyebrow {
    color: var(--teal);
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.page-subtitle {
    color: #718096;
    font-size: 0.98rem;
    margin-top: -0.45rem;
    margin-bottom: 1.4rem;
}
.hero {
    background: linear-gradient(115deg, #10243A 0%, #183D55 68%, #147C78 100%);
    border-radius: 16px;
    padding: 1.35rem 1.6rem;
    color: white;
    margin: 0.35rem 0 1.1rem 0;
    box-shadow: 0 8px 22px rgba(16,36,58,0.10);
}
.hero h2 { color: white !important; margin: 0.2rem 0 0.35rem 0; font-size: 1.65rem !important; }
.hero p { color: #D5E3EB; margin: 0; font-size: 0.93rem; max-width: 850px; }
.hero .eyebrow { color: #7FE0D4; }
.section-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 2px 10px rgba(16,36,58,0.025);
}
.section-card h4 { margin: 0 0 0.4rem 0; }
.small-muted { color: #718096; font-size: 0.86rem; }
.status-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 700;
    background: #E7F5F3;
    color: #117F78;
}
.footer-note {
    color: #8392A1;
    font-size: 0.78rem;
    border-top: 1px solid var(--line);
    padding-top: 0.9rem;
    margin-top: 2.2rem;
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
        "Borrower registration",
        "Create a fictional applicant record for the research demonstration."
    )
    st.info(
        "The fields below come from the German Credit benchmark dataset. "
        "They are not a complete Ugandan loan application form."
    )
    with st.form("borrower_registration"):
        st.markdown("#### Applicant reference")
        reference = st.text_input(
            "Demonstration reference",
            value="UG-DEMO-001",
            help="Use a fictional reference. Do not enter real personal information."
        )
        st.markdown("---")
        st.markdown("#### Borrower and credit profile")
        values = {}
        left, right = st.columns(2)
        defaults = {
            "duration": 24,
            "credit_amount": 2500,
            "installment_commitment": 2,
            "residence_since": 2,
            "age": 35,
            "existing_credits": 1,
            "num_dependents": 1
        }
        for index, feature in enumerate(original_features):
            column = left if index % 2 == 0 else right
            with column:
                label = feature.replace("_", " ").title()
                if feature in numerical_features:
                    minimum = 1 if feature in (
                        "duration", "credit_amount", "age"
                    ) else 0
                    default = max(minimum, defaults.get(feature, 1))
                    values[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        value=default,
                        step=1,
                        key=f"input_{feature}"
                    )
                else:
                    options = list(categorical_options[feature])
                    values[feature] = st.selectbox(
                        label,
                        options=options,
                        format_func=lambda value: str(value),
                        key=f"input_{feature}"
                    )
        st.markdown("---")
        submitted = st.form_submit_button(
            "Save borrower details",
            use_container_width=True
        )

    if submitted:
        if not reference.strip():
            st.error("Enter a demonstration reference.")
        else:
            st.session_state.borrower = {
                "reference": reference.strip(),
                "values": values
            }
            st.session_state.assessment = None
            st.success("Borrower details saved for this session.")
            st.caption(f"Saved reference: {reference.strip()}")

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
