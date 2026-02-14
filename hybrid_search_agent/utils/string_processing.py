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
