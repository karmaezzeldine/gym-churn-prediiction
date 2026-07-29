"""
Gym Churn Predictor + Retention Assistant
==========================================
Part A: Predicts whether a member is likely to churn ("ghost") by February,
        using the winning model from the notebook: Logistic Regression
        (MinMaxScaler) — test accuracy 74.2%, best of all models compared
        (LR, Decision Tree, Random Forest, XGBoost, Deep NN).
Part B: If flagged as high-risk, routes them to a short survey, classifies
        their reason with Gemini, and shows a tailored retention action.

SETUP (once you export from the notebook):
    import joblib
    joblib.dump(logistic_model, "model.pkl")      # the MinMaxScaler-trained LR model
    joblib.dump(minmax_scaler, "scaler.pkl")
Drop both .pkl files into this same folder. Until then, the app runs in
DEMO MODE so you can fully test the survey -> AI -> action flow.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from classify import classify_reason
from actions import (
    CATEGORIES, get_action, register_branch_vote,
    register_buddy_request, register_trainer_request, register_followup_request,
)

st.set_page_config(page_title="Gym Churn Predictor", page_icon="💪")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# ---------------------------------------------------------------------------
# Exact final feature order the model was trained on (from x.columns in the
# notebook, 31 columns total — includes the 5 engineered features since the
# notebook's drop of them was commented out). DO NOT reorder —
# StandardScaler/MinMaxScaler transform positionally, not by column name.
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "age", "distance_km", "prior_gym_experience", "joined_with_friend",
    "monthly_fee_usd", "signup_week_january", "promo_used", "goal_aggressiveness",
    "week1_visits", "week2_visits", "week3_visits", "week4_visits",
    "avg_weekly_classes", "booked_induction", "personal_trainer", "app_installed",
    "avg_session_minutes", "guest_passes_used", "locker_rented",
    "visit_trend", "total_visits", "recent_activity_ratio", "engagement_score", "cost_per_visit",
    "contract_type_annual", "contract_type_month_to_month",
    "sex_M", "sex_Other",
    "primary_goal_general_health", "primary_goal_muscle_gain", "primary_goal_weight_loss",
]

CATEGORICAL_FIELDS = {
    "contract_type": {"options": ["6_month", "month_to_month", "annual"], "baseline": "6_month"},
    "sex": {"options": ["F", "M", "Other"], "baseline": "F"},
    "primary_goal": {"options": ["event_prep", "general_health", "muscle_gain", "weight_loss"], "baseline": "event_prep"},
}

CONTRACT_LABELS = {"6_month": "6-Month Contract", "month_to_month": "Month-to-Month", "annual": "Annual"}
GOAL_LABELS = {"event_prep": "Event Prep", "general_health": "General Health", "muscle_gain": "Muscle Gain", "weight_loss": "Weight Loss"}
SEX_LABELS = {"F": "Female", "M": "Male", "Other": "Other"}


def load_model():
    """Loads the trained model + scaler if present, else runs in demo mode."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, False
    return None, None, True


def yn(label, default="No"):
    return 1 if st.selectbox(label, ["No", "Yes"], index=0 if default == "No" else 1) == "Yes" else 0


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """Builds the exact one-hot-encoded row the model expects, in FEATURE_ORDER."""
    row = {
        "age": inputs["age"],
        "distance_km": inputs["distance_km"],
        "prior_gym_experience": inputs["prior_gym_experience"],
        "joined_with_friend": inputs["joined_with_friend"],
        "monthly_fee_usd": inputs["monthly_fee_usd"],
        "signup_week_january": inputs["signup_week_january"],
        "promo_used": inputs["promo_used"],
        "goal_aggressiveness": inputs["goal_aggressiveness"],
        "week1_visits": inputs["week1_visits"],
        "week2_visits": inputs["week2_visits"],
        "week3_visits": inputs["week3_visits"],
        "week4_visits": inputs["week4_visits"],
        "avg_weekly_classes": inputs["avg_weekly_classes"],
        "booked_induction": inputs["booked_induction"],
        "personal_trainer": inputs["personal_trainer"],
        "app_installed": inputs["app_installed"],
        "avg_session_minutes": inputs["avg_session_minutes"],
        "guest_passes_used": inputs["guest_passes_used"],
        "locker_rented": inputs["locker_rented"],
    }

    # Engineered features — must match the notebook's formulas exactly (cells 8-10)
    w1, w2, w3, w4 = inputs["week1_visits"], inputs["week2_visits"], inputs["week3_visits"], inputs["week4_visits"]
    total_visits = w1 + w2 + w3 + w4
    row["visit_trend"] = w4 - w1
    row["total_visits"] = total_visits
    row["recent_activity_ratio"] = (w3 + w4) / (total_visits + 1e-5)
    row["engagement_score"] = (
        inputs["booked_induction"] + inputs["personal_trainer"]
        + inputs["app_installed"] + inputs["locker_rented"]
    )
    row["cost_per_visit"] = inputs["monthly_fee_usd"] / (total_visits + 1)

    for field, cfg in CATEGORICAL_FIELDS.items():
        selected = inputs[field]
        for option in cfg["options"]:
            if option == cfg["baseline"]:
                continue
            row[f"{field}_{option}"] = 1 if selected == option else 0

    return pd.DataFrame([row])[FEATURE_ORDER]  # reindex to guarantee correct order


def predict_churn(model, scaler, feature_row: pd.DataFrame, demo_mode: bool):
    if demo_mode:
        # Placeholder rule so you can test both branches of the app before
        # model.pkl / scaler.pkl exist. Replace by exporting the real model.
        fee = feature_row["monthly_fee_usd"].iloc[0]
        visits = feature_row[["week1_visits", "week2_visits", "week3_visits", "week4_visits"]].sum(axis=1).iloc[0]
        pseudo_risk = 0.8 if (fee > 90 or visits < 4) else 0.3
        return int(pseudo_risk > 0.5), pseudo_risk
    scaled = scaler.transform(feature_row)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]
    return int(pred), float(proba)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💪 Gym Churn Predictor")
st.caption("Predict which members are likely to ghost the gym by February, "
           "and route them to a personalized retention offer.")

model, scaler, demo_mode = load_model()
if demo_mode:
    st.info("Running in **demo mode** — model.pkl / scaler.pkl not found yet. "
            "Predictions are placeholder logic so you can test the survey + "
            "AI classification flow end to end.")

st.subheader("1. Member Info")

inputs = {}
c1, c2 = st.columns(2)
with c1:
    inputs["age"] = st.number_input("Age", min_value=16, max_value=90, value=30)
    inputs["distance_km"] = st.number_input("Distance from gym (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    inputs["monthly_fee_usd"] = st.number_input("Monthly Fee (USD)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
    inputs["signup_week_january"] = st.slider("Signup Week (January)", 1, 4, 1)
    inputs["goal_aggressiveness"] = st.slider("Goal Aggressiveness (1-5)", 1, 5, 3)
    inputs["avg_weekly_classes"] = st.number_input("Avg Weekly Classes Attended", min_value=0, max_value=20, value=2)
    inputs["avg_session_minutes"] = st.number_input("Avg Session Length (minutes)", min_value=0, max_value=240, value=45)
    inputs["guest_passes_used"] = st.number_input("Guest Passes Used", min_value=0, max_value=20, value=0)
with c2:
    inputs["contract_type"] = st.selectbox("Contract Type", CATEGORICAL_FIELDS["contract_type"]["options"], format_func=lambda x: CONTRACT_LABELS[x])
    inputs["sex"] = st.selectbox("Sex", CATEGORICAL_FIELDS["sex"]["options"], format_func=lambda x: SEX_LABELS[x])
    inputs["primary_goal"] = st.selectbox("Primary Goal", CATEGORICAL_FIELDS["primary_goal"]["options"], format_func=lambda x: GOAL_LABELS[x])
    inputs["prior_gym_experience"] = yn("Prior Gym Experience?")
    inputs["joined_with_friend"] = yn("Joined With a Friend?")
    inputs["promo_used"] = yn("Used a Promo Code?")
    inputs["booked_induction"] = yn("Booked Induction Session?")
    inputs["personal_trainer"] = yn("Has a Personal Trainer?")
    inputs["app_installed"] = yn("Gym App Installed?")
    inputs["locker_rented"] = yn("Locker Rented?")

st.markdown("**Weekly Visits (January)**")
w1, w2, w3, w4 = st.columns(4)
inputs["week1_visits"] = w1.number_input("Week 1", min_value=0, max_value=14, value=3)
inputs["week2_visits"] = w2.number_input("Week 2", min_value=0, max_value=14, value=2)
inputs["week3_visits"] = w3.number_input("Week 3", min_value=0, max_value=14, value=2)
inputs["week4_visits"] = w4.number_input("Week 4", min_value=0, max_value=14, value=1)

if "churn_result" not in st.session_state:
    st.session_state.churn_result = None

if st.button("Predict Churn Risk", type="primary"):
    feature_row = build_feature_row(inputs)
    pred, proba = predict_churn(model, scaler, feature_row, demo_mode)
    st.session_state.churn_result = {"pred": pred, "proba": proba}
    st.session_state.pop("category", None)  # reset any previous survey result

if st.session_state.churn_result:
    pred = st.session_state.churn_result["pred"]
    proba = st.session_state.churn_result["proba"]

    if pred == 1:
        st.error(f"⚠️ High churn risk — estimated probability {proba:.0%}")
    else:
        st.success(f"✅ Low churn risk — estimated probability {proba:.0%}")

    if pred == 1:
        st.divider()
        st.subheader("2. Quick Survey")
        st.write("We're sorry to see you might be leaving! Help us understand why:")

        free_text = st.text_area(
            "In your own words, why are you thinking about leaving?",
            placeholder="e.g. It's just too expensive for how often I go..."
        )
        backup_category = st.selectbox(
            "(Optional) Or pick the closest reason:",
            ["-- prefer to describe above --"] + CATEGORIES
        )

        if st.button("Submit Survey"):
            if backup_category != "-- prefer to describe above --" and not free_text.strip():
                category = backup_category
            else:
                with st.spinner("Reading your answer..."):
                    category = classify_reason(free_text)
            st.session_state.category = category

        if "category" in st.session_state:
            category = st.session_state.category
            action = get_action(category)

            st.divider()
            st.subheader("3. Here's what we can do")
            st.write(f"**Detected reason:** {category}")
            st.write(action["message"])

            if action["action_type"] == "branch_vote":
                location = st.text_input("What's your zip code or neighborhood?")
                if location and st.button("Cast my vote"):
                    result = register_branch_vote(location)
                    st.success(
                        f"Thanks! {result['count']} member(s) in {location} "
                        f"want a gym nearby (need {result['needed']})."
                    )
                    if result["threshold_reached"]:
                        st.balloons()
                        st.success(
                            f"🎉 Threshold reached for {location}! This area "
                            "is now flagged for a potential new branch."
                        )

            elif action["action_type"] == "workout_buddy":
                st.write("Tell us a bit more so we can find you a good match:")
                email = st.text_input("Email")
                routine = st.selectbox(
                    "Preferred workout routine",
                    ["Cardio", "Weight Training", "Group Classes", "Yoga / Mobility", "Mixed / Not sure"]
                )
                times = st.multiselect(
                    "What times do you usually go?",
                    ["Early Morning", "Morning", "Afternoon", "Evening", "Late Night", "Weekends"]
                )
                location = st.text_input("Your gym location / branch")
                if st.button("Find me a buddy"):
                    if not email or not times:
                        st.warning("Please add your email and at least one preferred time.")
                    else:
                        register_buddy_request({
                            "email": email, "routine": routine,
                            "times": times, "location": location,
                        })
                        st.success("Got it! We'll match you with a workout buddy "
                                   "who shares your routine and schedule, and "
                                   "email you their contact info soon.")

            elif action["action_type"] == "free_pt_trial":
                st.write("Tell us a bit more so we can pair you with the right trainer:")
                email = st.text_input("Email")
                goal = st.selectbox(
                    "Main training goal",
                    ["Weight Loss", "Strength & Muscle Gain", "Mobility / Injury Recovery", "Sports Performance", "General Fitness"]
                )
                times = st.multiselect(
                    "What times work for sessions?",
                    ["Early Morning", "Morning", "Afternoon", "Evening", "Late Night", "Weekends"]
                )
                location = st.text_input("Your gym location / branch")
                if st.button("Book my free trial"):
                    if not email or not times:
                        st.warning("Please add your email and at least one preferred time.")
                    else:
                        register_trainer_request({
                            "email": email, "goal": goal,
                            "times": times, "location": location,
                        })
                        st.success("You're booked in for a free 2-week trial! "
                                   "We'll email you to confirm your first "
                                   "session with a trainer.")

            elif action["action_type"] == "human_followup":
                email = st.text_input("Email")
                if st.button("Submit"):
                    if not email:
                        st.warning("Please add your email so we can follow up.")
                    else:
                        register_followup_request(email)
                        st.success("Thanks! A member of our team will reach out to you soon.")

# ---------------------------------------------------------------------------
# Export snippet for your notebook, once you're happy with the LR model:
#
#   import joblib
#   joblib.dump(logistic_model, "model.pkl")   # the one fit on x_train_mm
#   joblib.dump(minmax_scaler, "scaler.pkl")
#
# Then drop both .pkl files into this app's folder.
# ---------------------------------------------------------------------------
