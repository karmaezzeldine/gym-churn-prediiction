"""
Maps a churn-reason category to a retention action, and handles the
"vote for a new branch" flow for the distance category, plus saving
buddy/trainer/followup requests.

PERSISTENCE:
Every save function here tries the Supabase database first (see db.py).
If Supabase isn't configured yet (no secrets set), it automatically falls
back to a local JSON file so the app keeps working while you're testing.
Once Supabase is set up, all new data goes there instead.
"""

import json
import os

import db

VOTES_FILE = os.path.join(os.path.dirname(__file__), "vote_counts.json")
BUDDY_FILE = os.path.join(os.path.dirname(__file__), "buddy_requests.json")
TRAINER_FILE = os.path.join(os.path.dirname(__file__), "trainer_requests.json")
FOLLOWUP_FILE = os.path.join(os.path.dirname(__file__), "followup_requests.json")

VOTES_NEEDED_FOR_NEW_BRANCH = 15  # tweak this threshold as you like

CATEGORIES = ["Loneliness", "Cost", "Distance", "No Personal Trainer", "Other"]

ACTIONS = {
    "Loneliness": {
        "message": "We hear you! We're pairing you with a workout buddy "
                    "who shares similar goals so the gym feels less lonely.",
        "action_type": "workout_buddy",
    },
    "Cost": {
        "message": "Here's a promo code for 20% off your next 3 months: "
                    "STAY20",
        "action_type": "promo_code",
    },
    "No Personal Trainer": {
        "message": "You're eligible for a free 2-week trial with one of "
                    "our personal trainers. We'll email you to schedule "
                    "your first session.",
        "action_type": "free_pt_trial",
    },
    "Distance": {
        "message": "We're tracking demand for a gym closer to you! "
                    "Vote below to help us decide where to open next.",
        "action_type": "branch_vote",
    },
    "Other": {
        "message": "Thanks for letting us know. A member of our team will "
                    "follow up with you directly.",
        "action_type": "human_followup",
    },
}


def get_action(category: str) -> dict:
    """Return the action dict for a given category, defaulting to Other."""
    return ACTIONS.get(category, ACTIONS["Other"])


def _load_votes() -> dict:
    if not os.path.exists(VOTES_FILE):
        return {}
    with open(VOTES_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_votes(votes: dict) -> None:
    with open(VOTES_FILE, "w") as f:
        json.dump(votes, f, indent=2)


def register_branch_vote(location: str) -> dict:
    """
    Increments the vote count for a given location/zip/area string.
    Tries Supabase first (one row per vote, counted via query); falls back
    to the local JSON counter if the database isn't configured.
    Returns a dict with the updated count and whether the threshold
    for opening a new branch has been reached.
    """
    saved = db.save_response("branch_votes", {"location": location})

    if saved:
        count = db.count_rows("branch_votes", "location", location)
    else:
        votes = _load_votes()
        votes[location] = votes.get(location, 0) + 1
        _save_votes(votes)
        count = votes[location]

    return {
        "location": location,
        "count": count,
        "needed": VOTES_NEEDED_FOR_NEW_BRANCH,
        "threshold_reached": count >= VOTES_NEEDED_FOR_NEW_BRANCH,
    }


def _load_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _append_to_list(path: str, record: dict) -> None:
    records = _load_list(path)
    records.append(record)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def register_buddy_request(profile: dict) -> None:
    """
    Saves a workout-buddy matching request (email, preferred routine,
    preferred times, location) to Supabase, or locally if not configured.
    A real matching job would later query buddy_requests for members with
    overlapping routine/times/location.
    """
    if not db.save_response("buddy_requests", profile):
        _append_to_list(BUDDY_FILE, profile)


def register_trainer_request(profile: dict) -> None:
    """Saves a free personal-trainer-trial request to Supabase, or locally if not configured."""
    if not db.save_response("trainer_requests", profile):
        _append_to_list(TRAINER_FILE, profile)


def register_followup_request(email: str) -> None:
    """Saves a plain human-follow-up request for the 'Other' category."""
    if not db.save_response("followup_requests", {"email": email}):
        _append_to_list(FOLLOWUP_FILE, {"email": email})
