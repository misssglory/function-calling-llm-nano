from typing import List
import json
import re


def get_text_before_first_marker(text):
    # Find the first occurrence of either marker
    thought_pos = text.find("Thought: ")
    backtick_pos = text.find("```")

    # Handle cases where markers aren't found
    if thought_pos == -1:
        thought_pos = float("inf")
    if backtick_pos == -1:
        backtick_pos = float("inf")

    # Get the earliest position
    first_marker = min(thought_pos, backtick_pos)

    # Return everything before that position (if marker found)
    if first_marker != float("inf"):
        return text[:first_marker]
    return text  # No markers found


def get_answer_between_markers(text):
    """
    Extract text between first occurrence of 'Answer: ' and either '```' or 'Thought: '
    Falls back to get_text_before_first_marker if 'Answer: ' not found
    """
    # Find the position of 'Answer: '
    answer_pos = text.find("Answer: ")

    if answer_pos == -1:
        # No 'Answer: ' found, use fallback
        return get_text_before_first_marker(text)

    # Get text after 'Answer: '
    start_pos = answer_pos + len("Answer: ")
    remaining_text = text[start_pos:]

    # Find next marker positions in the remaining text
    thought_pos = remaining_text.find("Thought: ")
    backtick_pos = remaining_text.find("```")

    # Handle cases where markers aren't found
    if thought_pos == -1:
        thought_pos = float("inf")
    if backtick_pos == -1:
        backtick_pos = float("inf")

    # Get the earliest marker position
    first_marker = min(thought_pos, backtick_pos)

    # Extract text up to the first marker (if found)
    if first_marker != float("inf"):
        return remaining_text[:first_marker]

    # No markers found after 'Answer: ', return everything after it
    return remaining_text


def get_first_json_object(text):
    """
    Extract text between first occurrence of '{' and its matching '}'
    Returns the text including the braces
    Falls back to get_answer_between_markers if no valid '{' found
    """
    # Find first occurrence of '{'
    open_brace_pos = text.find("{")

    if open_brace_pos == -1:
        # No '{' found, use fallback
        return get_answer_between_markers(text)

    # Track brace nesting level
    brace_count = 0
    in_string = False
    escape_next = False

    # Start from the opening brace
    for i in range(open_brace_pos, len(text)):
        char = text[i]

        # Handle string literals (ignore braces inside strings)
        if char == '"' and not escape_next:
            in_string = not in_string
        elif char == "\\" and not escape_next:
            escape_next = True
        else:
            escape_next = False

        # Only count braces if not inside a string
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # When brace_count returns to 0, we've found the matching closing brace
                if brace_count == 0:
                    # Return text from opening brace to closing brace (inclusive)
                    return text[open_brace_pos : i + 1]

    # If we never found matching closing brace, use fallback
    return get_answer_between_markers(text)
