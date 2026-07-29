"""
Handles saving survey responses to a Supabase (Postgres) database, so data
persists across app restarts/redeployments — unlike local JSON files.

SETUP:
1. Create a free project at https://supabase.com
2. Run the SQL (see README / setup instructions) to create the tables.
3. Get your Project URL and "anon public" key from Settings -> API.
4. Add both to .streamlit/secrets.toml:
       SUPABASE_URL = "https://xxxxx.supabase.co"
       SUPABASE_KEY = "your-anon-public-key"
   And add the same two under your Streamlit Cloud app's Settings -> Secrets
   once deployed.
5. pip install supabase
"""

import streamlit as st
from supabase import create_client


def get_client():
    """Returns a Supabase client, or None if credentials aren't configured yet."""
    url = st.secrets.get("SUPABASE_URL", None)
    key = st.secrets.get("SUPABASE_KEY", None)
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def save_response(table: str, record: dict) -> bool:
    """
    Inserts a record into the given Supabase table. Lists (e.g. multi-select
    values like preferred times) are flattened to comma-separated strings
    since Postgres text columns don't take Python lists directly.
    Returns True on success, False if the DB isn't configured or the
    insert failed (caller should fall back to local storage in that case).
    """
    client = get_client()
    if client is None:
        return False

    clean_record = {
        k: (", ".join(v) if isinstance(v, list) else v)
        for k, v in record.items()
    }

    try:
        client.table(table).insert(clean_record).execute()
        return True
    except Exception as e:
        st.warning(f"Could not save to database, saved locally instead. ({e})")
        return False


def count_rows(table: str, filter_column: str, filter_value: str) -> int:
    """Counts rows in a table matching a filter (used for branch vote counts)."""
    client = get_client()
    if client is None:
        return 0
    try:
        result = (
            client.table(table)
            .select("id", count="exact")
            .eq(filter_column, filter_value)
            .execute()
        )
        return result.count or 0
    except Exception:
        return 0
