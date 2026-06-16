# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import json

# Fallback values
FALLBACK_TYPE = "push"
FALLBACK_TIME = 4.0  # will be replaced by median at runtime if available


def parse_prediction(text, event_types, fallback_time=None, anon_to_type=None, time_key="time_hours"):
    """Parse model output to extract event_type and time_hours.

    Looks for the last JSON object in the text (after any reasoning).
    Returns (event_type, time_hours, parse_success).

    Args:
        event_types: List of valid event type strings.
        anon_to_type: Optional dict mapping anonymous IDs (e.g. "type_0") to real types.
    """
    if fallback_time is None:
        fallback_time = FALLBACK_TIME

    valid_types = list(event_types)
    if anon_to_type:
        valid_types = valid_types + list(anon_to_type.keys())

    # Find all JSON-like objects in the text
    json_matches = re.findall(r'\{[^{}]*\}', text)

    for match in reversed(json_matches):
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue

        event_type = obj.get("event_type")
        time_hours = obj.get(time_key)

        if event_type is None or time_hours is None:
            continue

        # Map anonymous ID back to real type
        if anon_to_type and event_type in anon_to_type:
            event_type = anon_to_type[event_type]

        # Validate event type
        if event_type not in event_types:
            # Try case-insensitive match
            lower_map = {e.lower(): e for e in event_types}
            event_type = lower_map.get(event_type.lower(), None)
            if event_type is None:
                continue

        # Validate time
        try:
            time_hours = float(time_hours)
            if time_hours < 0:
                time_hours = abs(time_hours)
        except (ValueError, TypeError):
            continue

        return event_type, time_hours, True

    return event_types[0], fallback_time, False
