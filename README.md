# Gym Churn Predictor & Retention Assistant

Predicts which gym members are likely to stop attending ("ghost") by February,
after a New Year's resolution sign-up rush — then routes at-risk members
through a short AI-powered survey to understand why, and offers a tailored
retention action.

**Live app:** https://gym-churn-prediiction-hp36gq3zyauvkcqieexgxu.streamlit.app/

## Project Overview

Many people join gyms as part of a New Year's resolution but stop attending
within a few months. This project predicts churn risk from a member's
signup details, contract information, and January attendance patterns,
then acts on that prediction: high-risk members are surveyed for their
reason for leaving (classified automatically via the Gemini API), and
matched to a targeted retention offer — a workout buddy for loneliness, a
promo code for cost concerns, a free personal trainer trial, or a
"vote for a new branch" flow for members who live far away.

## Dataset Description

- **Source:** Kaggle — [Gym: Will Your New Year's Resolution Survive?](https://www.kaggle.com/datasets/sergionefedov/gym-will-your-new-years-resolution-survive)
- **Samples:** 50,000 gym members
- **Raw features:** 23 columns covering demographics (age, sex), contract
  details (contract_type, monthly_fee_usd), attendance (week1-4_visits,
  avg_weekly_classes, avg_session_minutes), engagement indicators
  (booked_induction, personal_trainer, app_installed, locker_rented), and
  goals (primary_goal, goal_aggressiveness)
- **Engineered features:** visit_trend, total_visits, recent_activity_ratio,
  engagement_score, cost_per_visit — added to capture attendance momentum
  and value perception, expanding the model to 31 total features after
  one-hot encoding
- **Target variable:** `churned_by_february` (binary)

## Project Structure

```
├── app.py                      # Streamlit app (prediction + survey + actions)
├── classify.py                 # Gemini API call to classify churn reasons
├── actions.py                  # Category -> retention action mapping
├── db.py                       # Supabase database integration
├── model.pkl                   # Trained Logistic Regression model
├── scaler.pkl                  # Fitted MinMaxScaler
├── model_info.json             # Model version/metadata log
├── requirements.txt
├── .streamlit/secrets.toml.example  # Template for API keys (not the real file)
├── gym-nti.ipynb                # Full training notebook (EDA, preprocessing, models)
└── README.md
```

## Installation Instructions

```bash
git clone https://github.com/karmaezzeldine/gym-churn-prediiction.git
cd gym-churn-prediiction
pip install -r requirements.txt
```

Then create `.streamlit/secrets.toml` (copy from `secrets.toml.example`) with
your own API keys:
```toml
GEMINI_API_KEY = "your-gemini-api-key"
SUPABASE_URL = "your-supabase-project-url"
SUPABASE_KEY = "your-supabase-anon-key"
```

## How to Run

```bash
streamlit run app.py
```
Then open the local URL it prints (usually `http://localhost:8501`).

## Model Description

We compared six model types — Logistic Regression, Decision Tree, KNN,
Random Forest, XGBoost, and a Deep Neural Network — each with
StandardScaler and MinMaxScaler preprocessing. **Logistic Regression with
MinMaxScaler** was selected as the final model, achieving the best test
accuracy (74.2%) and F1-score (0.72) for the churned class, with the
smallest train/test gap of any model tested (indicating good generalization
rather than overfitting — Random Forest, by comparison, scored 79% on
training data but only 73% on test data).

**Strengths:** Consistent performance across train/validation/test splits,
fast to train and deploy, and directly interpretable via its coefficients.

**Limitations:** ~74% accuracy means roughly 1 in 4 members are
misclassified, so predictions should guide retention outreach priorities
rather than serve as a certain diagnosis. As a linear model it may miss
non-linear feature interactions, though none of the more complex models we
tested actually outperformed it on this dataset.

## Streamlit Deployment

**Live app:** https://gym-churn-prediiction-hp36gq3zyauvkcqieexgxu.streamlit.app/
