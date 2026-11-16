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
    layout="wide"
)

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
    numerical_features = list(
        preprocessor.transformers_[0][2]
    )
    categorical_features = list(
        preprocessor.transformers_[1][2]
    )
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

if "borrower" not in st.session_state:
    st.session_state.borrower = None
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "history" not in st.session_state:
    st.session_state.history = []
if "reviews" not in st.session_state:
    st.session_state.reviews = []

st.sidebar.title("🏦 Credit Risk Advisory")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Borrower Registration",
        "Credit Risk Assessment",
        "Explainable AI",
        "Credit Officer Review"
    ]
)
st.sidebar.divider()
st.sidebar.caption("Academic research prototype · Version 1.0")

st.title("Uganda Credit Risk Advisory System")
st.caption("Explainable AI prototype for research and demonstration")
st.warning(
    "Research-only demonstration. The model was trained on the German Credit "
    "benchmark dataset, not Ugandan bank lending data. Do not use its outputs "
    "to approve, reject, price, or otherwise make real lending decisions. "
    "Enter fictional demonstration data only."
)

if page == "Dashboard":
    st.subheader("System dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Assessments this session", len(st.session_state.history))
    c2.metric("Prediction models", "2")
    c3.metric("Explanation methods", "2")
    st.markdown(
        """
        **Prototype workflow**
        1. Register a fictional borrower using fields from the benchmark dataset.
        2. Generate a Gradient Boosting credit-class prediction.
        3. Inspect SHAP and LIME explanations for the individual prediction.
        4. Record a human review status and notes.
        
        Logistic Regression is retained in the model package for model comparison.
        """
    )
    if st.session_state.history:
        st.subheader("Session assessment history")
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )

elif page == "Borrower Registration":
    st.subheader("Borrower registration")
    st.info(
        "These are the German Credit benchmark dataset fields. They are not a "
        "complete Ugandan loan application form."
    )
    with st.form("borrower_registration"):
        reference = st.text_input("Demonstration reference", value="UG-DEMO-001")
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
                    minimum = 1 if feature in ("duration", "credit_amount", "age") else 0
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

elif page == "Credit Risk Assessment":
    st.subheader("Credit risk assessment")
    borrower = st.session_state.borrower
    if borrower is None:
        st.info("Register a demonstration borrower first.")
    else:
        st.write("Reference:", borrower["reference"])
        if st.button("Run Gradient Boosting assessment", type="primary"):
            try:
                row = pd.DataFrame([borrower["values"]])[original_features]
                transformed = preprocessor.transform(row)
                transformed_df = pd.DataFrame(
                    transformed,
                    columns=feature_names
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
                    "Model classification": "Higher risk" if prediction == 1 else "Lower risk",
                    "Bad-credit model probability": f"{bad_probability:.1%}"
                })
            except Exception as exc:
                st.error("Assessment could not be completed.")
                st.exception(exc)

        assessment = st.session_state.assessment
        if assessment is not None:
            st.divider()
            st.write("Applicant reference:", assessment["reference"])
            a, b = st.columns(2)
            a.metric("Model probability: bad-credit class",
                     f"{assessment['bad_probability']:.1%}")
            b.metric("Model probability: good-credit class",
                     f"{assessment['good_probability']:.1%}")
            if assessment["prediction"] == 1:
                st.error("Model classification: Higher risk (bad-credit class)")
            else:
                st.success("Model classification: Lower risk (good-credit class)")
            st.caption(
                "The class uses the model's default 0.50 decision threshold. "
                "These are benchmark-model outputs, not lending recommendations."
            )

elif page == "Explainable AI":
    st.subheader("Individual explainability")
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("Run a borrower assessment first.")
    else:
        st.markdown("### SHAP — feature contributions")
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": assessment["shap_contributions"]
        })
        shap_df["Absolute contribution"] = shap_df["Contribution"].abs()
        shap_df = shap_df.nlargest(12, "Absolute contribution")
        st.bar_chart(
            shap_df.set_index("Feature")[["Contribution"]],
            horizontal=True
        )
        st.caption(
            "Positive values push the model output toward the bad-credit class; "
            "negative values push it toward the good-credit class. SHAP values "
            "are contributions in the model output space, not percentage-point "
            "changes in predicted probability."
        )
        st.markdown("### LIME — local approximation")
        lime_items = assessment["lime_result"].as_list(label=1)
        lime_df = pd.DataFrame(
            lime_items,
            columns=["Feature condition", "Local contribution"]
        )
        st.dataframe(lime_df, use_container_width=True, hide_index=True)
        st.metric(
            "LIME local fidelity",
            f"{assessment['lime_result'].score:.3f}"
        )
        st.caption(
            "LIME approximates the model around this one applicant. Its local "
            "fidelity score is not the model's prediction accuracy."
        )
        st.info(
            "SHAP and LIME use different methods and may produce different "
            "rankings. Their explanations describe model behaviour, not causal "
            "effects or proof that a factor is fair."
        )

elif page == "Credit Officer Review":
    st.subheader("Credit officer review")
    assessment = st.session_state.assessment
    if assessment is None:
        st.info("Generate an assessment before recording a review.")
    else:
        st.write("Applicant reference:", assessment["reference"])
        st.write(
            "Model classification:",
            "Higher risk" if assessment["prediction"] == 1 else "Lower risk"
        )
        st.write(
            "Bad-credit class probability:",
            f"{assessment['bad_probability']:.1%}"
        )
        st.warning(
            "This review panel records a research demonstration review only. "
            "It does not authorise a real credit decision."
        )
        with st.form("officer_review"):
            officer = st.text_input("Demonstration reviewer")
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
            save_review = st.form_submit_button("Save review")
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
            st.markdown("### Reviews in this session")
            st.dataframe(
                pd.DataFrame(st.session_state.reviews),
                use_container_width=True,
                hide_index=True
            )

st.sidebar.divider()
st.sidebar.caption(
    "Research prototype only · Session records are not persistent storage."
)
