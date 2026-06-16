# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from config import CATEGORY_EVENT_TYPES, CATEGORY_TYPE_TO_ID, EVENT_TYPES, TYPE_TO_ID, TIME_SUFFIX, TIME_KEY


def get_system_prompt(category, include_type_text=True, anonymize_types=False):
    """Return the appropriate system prompt for a given category."""
    event_types = CATEGORY_EVENT_TYPES[category]
    if anonymize_types:
        type_list = ", ".join(CATEGORY_TYPE_TO_ID[category].values())
    else:
        type_list = ", ".join(event_types)

    if include_type_text:
        return f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {type_list}.

Each event shows: [type] +Xw (weeks since previous event) | description, where description provides details about the event (may be truncated). Use the event types, their descriptions, and the +Xw time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_weeks": <positive_number>}}"""
    else:
        return f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {type_list}.

Each event shows: [type] +Xw (weeks since previous event). Use both the event types and the +Xw time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_weeks": <positive_number>}}"""


def build_user_message(description, history_events, include_type_text=True, anonymize_types=False, category=None):
    """Build the user message from description and history events.

    Args:
        description: Sequence description string.
        history_events: List of dicts with keys 'type_event', 'time_since_last_event',
                        and optionally 'type_text'.
        include_type_text: Whether to include textual descriptions of events.
        anonymize_types: Whether to replace type names with anonymous IDs.
        category: If provided, use per-category TYPE_TO_ID for anonymization.
    """
    type_to_id = CATEGORY_TYPE_TO_ID[category] if category else TYPE_TO_ID
    _random_anon = getattr(__import__('config'), 'RANDOM_ANON', False)
    _prepend_semantic = getattr(__import__('config'), 'PREPEND_SEMANTIC_LABEL', False)

    lines = ["Event history (chronological):"]

    for i, event in enumerate(history_events, 1):
        etype = event["type_event"]
        semantic_name = etype
        if anonymize_types:
            if _random_anon:
                import random
                etype = random.choice(list(type_to_id.values()))
            else:
                etype = type_to_id[etype]
        delta = event["time_since_last_event"]
        if include_type_text and event.get("type_text"):
            text = event["type_text"]
            if _prepend_semantic and anonymize_types:
                text = f"{semantic_name}: {text}"
            if len(text) > 500:
                text = text[:500] + "..."
            lines.append(f"{i}. [{etype}] +{delta:.3f}{TIME_SUFFIX} | {text}")
        else:
            lines.append(f"{i}. [{etype}] +{delta:.3f}{TIME_SUFFIX}")

    lines.append("")
    lines.append(f"Predict the next event type and {TIME_KEY}.")
    return "\n".join(lines)


# Module-level prompt constants (backward compat for inference.py imports)
_type_list = ", ".join(EVENT_TYPES)
_anon_list = ", ".join(TYPE_TO_ID.values())

SYSTEM_PROMPT = f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {_type_list}.

Each event shows: [type] +Xw (weeks since previous event). Use both the event types and the +Xw time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_weeks": <positive_number>}}"""

SYSTEM_PROMPT_WITH_TEXT = f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {_type_list}.

Each event shows: [type] +Xw (weeks since previous event) | description, where description provides details about the event (may be truncated). Use the event types, their descriptions, and the +Xw time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_weeks": <positive_number>}}"""

SYSTEM_PROMPT_ANON = f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {_anon_list}.

Each event shows: [type] +Xw (weeks since previous event). Use both the event types and the +Xw time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_weeks": <positive_number>}}"""

SYSTEM_PROMPT_ANON_WITH_TEXT = f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {_anon_list}.

Each event shows: [type] +Xw (weeks since previous event) | description, where description provides details about the event (may be truncated). Use the event types, their descriptions, and the +Xw time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_weeks": <positive_number>}}"""
