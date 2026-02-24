"""
Date Parser Module — Natural Language → Real Dates
====================================================
Uses `dateparser` to convert text like "tomorrow", "next Sunday",
"March 15" into real datetime objects.

Also normalizes time strings: "10 AM" → "10:00", "3 PM" → "15:00".

Fully offline after first import.
"""

import re
from datetime import datetime, date, time, timedelta

import dateparser


# =========================================================================
# Date parsing
# =========================================================================

def parse_date(text: str | None) -> str | None:
    """
    Convert natural language date to ISO format string (YYYY-MM-DD).

    Examples:
        "tomorrow"     → "2026-02-25"
        "next Sunday"  → "2026-03-01"
        "March 15"     → "2026-03-15"
        "today"        → "2026-02-24"

    Returns None if the text cannot be parsed.
    """
    if not text:
        return None

    text = text.strip()

    # Use dateparser for natural language
    parsed = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "future",   # "tomorrow" = future date
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }
    )

    if parsed:
        return parsed.strftime("%Y-%m-%d")

    return None


# =========================================================================
# Time parsing / normalization
# =========================================================================

# Default times for vague references
DEFAULT_TIMES = {
    "morning": "09:00",
    "afternoon": "14:00",
    "evening": "18:00",
    "night": "20:00",
    "noon": "12:00",
    "midnight": "00:00",
}


def parse_time(text: str | None) -> str | None:
    """
    Normalize a time string to 24-hour HH:MM format.

    Examples:
        "10 AM"    → "10:00"
        "3 PM"     → "15:00"
        "10:30 AM" → "10:30"
        "noon"     → "12:00"
        "morning"  → "09:00"

    Returns None if the text cannot be parsed.
    """
    if not text:
        return None

    text = text.strip().lower()

    # Check for vague time words
    for keyword, default_time in DEFAULT_TIMES.items():
        if keyword in text:
            return default_time

    # Match patterns like "10 AM", "3:30 PM", "10:00 am"
    m = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', text, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        period = m.group(3).lower()

        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    # Match 24-hour format "14:30"
    m = re.match(r'(\d{1,2}):(\d{2})', text)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

    # Match bare number with context: "at 10", "by 3"
    m = re.match(r'(?:at|by|around|before|after)\s+(\d{1,2})', text)
    if m:
        hour = int(m.group(1))
        return f"{hour:02d}:00"

    return None


def combine_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
    """
    Combine a parsed date (YYYY-MM-DD) and parsed time (HH:MM)
    into a single datetime object.

    If time is missing, defaults to 09:00.
    If date is missing, returns None.
    """
    if not date_str:
        return None

    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None

    if time_str:
        try:
            t = datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            t = time(9, 0)  # Default to 9 AM
    else:
        t = time(9, 0)  # Default to 9 AM

    return datetime.combine(d, t)


# =========================================================================
# Quick test
# =========================================================================
if __name__ == "__main__":
    test_dates = ["today", "tomorrow", "next Sunday", "March 15", "yesterday", "this weekend"]
    test_times = ["10 AM", "3 PM", "10:30 am", "noon", "morning", "at 5"]

    print("--- Date Parsing ---")
    for d in test_dates:
        print(f"  {d:20s} → {parse_date(d)}")

    print("\n--- Time Parsing ---")
    for t in test_times:
        print(f"  {t:20s} → {parse_time(t)}")

    print("\n--- Combined ---")
    combined = combine_datetime(parse_date("tomorrow"), parse_time("10 AM"))
    print(f"  tomorrow + 10 AM = {combined}")
