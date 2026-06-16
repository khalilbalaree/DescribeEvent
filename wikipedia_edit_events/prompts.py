# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from config import TYPE_TO_ID

SYSTEM_PROMPT = """\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: revert, reference, content_addition, content_removal, copy_edit, minor_edit.

Each event shows: [type] +Xh (hours since previous event). Use both the event types and the +Xh time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{"event_type": "<type>", "time_hours": <positive_number>}"""

SYSTEM_PROMPT_WITH_TEXT = """\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: revert, reference, content_addition, content_removal, copy_edit, minor_edit.

Each event shows: [type] +Xh (hours since previous event) | description, where description provides details about the event (may be truncated). Use the event types, their descriptions, and the +Xh time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{"event_type": "<type>", "time_hours": <positive_number>}"""


_anon_list = ", ".join(TYPE_TO_ID.values())

SYSTEM_PROMPT_ANON = f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {_anon_list}.

Each event shows: [type] +Xh (hours since previous event). Use both the event types and the +Xh time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_hours": <positive_number>}}"""

SYSTEM_PROMPT_ANON_WITH_TEXT = f"""\
You predict the next event in a sequence. Given the event history, predict the next event type and time.

Valid event types: {_anon_list}.

Each event shows: [type] +Xh (hours since previous event) | description, where description provides details about the event (may be truncated). Use the event types, their descriptions, and the +Xh time gaps to predict what happens next and when.

Response format (follow exactly):
Brief reasoning here (2-3 sentences max).
{{"event_type": "<type>", "time_hours": <positive_number>}}"""


def build_user_message(description, history_events, include_type_text=True, anonymize_types=False):
    """Build the user message from description and history events.

    Args:
        description: Sequence description string.
        history_events: List of dicts with keys 'type_event', 'time_since_last_event',
                        and optionally 'type_text'.
        include_type_text: Whether to include textual descriptions of events.
        anonymize_types: Whether to replace type names with anonymous IDs.
    """
    from config import TYPE_TO_ID, TIME_SUFFIX, TIME_KEY
    _random_anon = getattr(__import__('config'), 'RANDOM_ANON', False)
    _prepend_semantic = getattr(__import__('config'), 'PREPEND_SEMANTIC_LABEL', False)
    lines = ["Event history (chronological):"]

    for i, event in enumerate(history_events, 1):
        etype = event["type_event"]
        semantic_name = etype  # keep original for prepending
        if anonymize_types:
            if _random_anon:
                import random
                etype = random.choice(list(TYPE_TO_ID.values()))
            else:
                etype = TYPE_TO_ID[etype]
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
