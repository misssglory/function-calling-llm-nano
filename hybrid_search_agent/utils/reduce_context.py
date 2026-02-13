import re


def remove_thoughts(text):
    """
    Remove all pieces of string that begin with 'Thought:' and end with
    'Action:', 'Action Input:', 'Observation:', or 'Answer:'.

    Args:
        text (str): Input text containing thoughts to remove

    Returns:
        str: Text with thoughts removed
    """
    # Pattern matches from 'Thought:' up to (but not including) any of the delimiters
    pattern = r"Thought:.*?(?=Action:|Action Input:|Observation:|Answer:|$)"

    # Remove all matches and clean up extra whitespace
    result = re.sub(pattern, "", text, flags=re.DOTALL)

    # Clean up any extra newlines or spaces that might remain
    result = re.sub(r"\n\s*\n", "\n", result)
    result = result.strip()

    return result


def remove_repeating_sequences(text, threshold=20, min_gap=0):
    """
    Delete repeating sequences from string longer than given threshold,
    leaving only the first occurrence in place.

    Args:
        text (str): Input text
        threshold (int): Minimum length of sequence to consider for removal
        min_gap (int): Minimum characters between repetitions to consider (0 = anywhere)

    Returns:
        str: Text with repeating sequences removed
    """
    if not text or threshold <= 0:
        return text

    result = text
    length = len(result)

    # Try different sequence lengths, from longest to shortest
    for seq_len in range(length // 2, threshold - 1, -1):
        i = 0
        while i <= len(result) - seq_len * 2:
            # Get the candidate sequence
            sequence = result[i : i + seq_len]

            # Look for this sequence later in the text
            search_start = i + seq_len + min_gap
            repeat_pos = result.find(sequence, search_start)

            if repeat_pos != -1:
                # Found a repeat, remove it
                result = result[:repeat_pos] + result[repeat_pos + seq_len :]
                # Don't increment i, check again at same position
                continue

            i += 1

    return result


def remove_repeating_sequences_optimized(text, threshold=20, min_gap=0):
    """
    Optimized version using dictionary to track seen sequences.
    Better for larger texts.
    """
    if not text or threshold <= 0:
        return text

    result = []
    i = 0
    seen_sequences = {}

    while i < len(text):
        # Try to find a known repeating sequence starting at position i
        found_repeat = False

        # Check from longest to shortest possible sequence
        max_len = min(len(text) - i, 100)  # Limit max sequence length for performance
        for seq_len in range(max_len, threshold - 1, -1):
            sequence = text[i : i + seq_len]

            # Check if we've seen this sequence before
            if sequence in seen_sequences:
                # Check min_gap constraint
                if i - seen_sequences[sequence] >= min_gap + seq_len:
                    found_repeat = True
                    i += seq_len  # Skip the entire sequence
                    break

        if not found_repeat:
            # Add current character and record possible sequences
            if len(text) - i >= threshold:
                for seq_len in range(threshold, min(50, len(text) - i) + 1):
                    sequence = text[i : i + seq_len]
                    if sequence not in seen_sequences:
                        seen_sequences[sequence] = i

            result.append(text[i])
            i += 1

    return "".join(result)


def remove_thoughts_and_duplicates(
    text, thought_removal=True, sequence_threshold=20, min_gap=0
):
    """
    Combined function to remove both thoughts and repeating sequences.
    """
    result = text

    if thought_removal:
        result = remove_thoughts(result)

    result = remove_repeating_sequences_optimized(result, sequence_threshold, min_gap)

    return result


def average_line_length(text):
    """
    Calculate the average length of sequences between newlines in a string.

    Args:
        text (str): Input text

    Returns:
        float: Average length of non-empty lines, or 0 if no lines
    """
    if not text:
        return 0.0

    # Split by newlines and filter out empty strings
    lines = [line for line in text.split("\n") if line.strip() != ""]

    if not lines:
        return 0.0

    # Calculate average length
    total_length = sum(len(line) for line in lines)
    average = total_length / len(lines)

    return round(average, 2)


def remove_leading_numbers_and_dots(text):
    """
    Remove numbers and dots from the beginning of a string until the first
    character that is neither a number nor a dot.

    Args:
        text (str): Input text

    Returns:
        str: Text with leading numbers and dots removed
    """
    if not text:
        return text

    # Find the position of the first character that's not a digit or dot
    i = 0
    while i < len(text) and (text[i].isdigit() or text[i] == "." or text[i] == " "):
        i += 1

    # Return the string from that position onward
    return text[i:]


def remove_short_lines(text, min_length, strip_whitespace=True, keep_empty=False):
    """
    Remove lines from string that are shorter than given threshold.

    Args:
        text (str): Input text
        min_length (int): Minimum line length required to keep the line
        strip_whitespace (bool): Whether to strip whitespace before counting length
        keep_empty (bool): Whether to keep empty lines regardless of threshold

    Returns:
        str: Text with short lines removed
    """
    if not text:
        return text

    lines = text.split("\n")
    filtered_lines = []

    for line in lines:
        # Determine line length for comparison
        if strip_whitespace:
            line_for_length = line.strip()
        else:
            line_for_length = line

        line_length = len(line_for_length)

        # Keep line if it meets criteria
        if keep_empty and line.strip() == "":
            filtered_lines.append(line)
        elif line_length >= min_length:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def remove_repeats(strings_list):
    """
    Remove repeating strings from a list while preserving order.

    Args:
        strings_list (list): List of strings that may contain duplicates

    Returns:
        list: New list with duplicates removed, preserving first occurrence order
    """
    if not strings_list:
        return []

    seen = set()
    result = []

    for item in strings_list:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# Enhanced testing
def test_functions():
    print("=== Testing remove_thoughts ===")
    test_cases = [
        (
            "Thought: I need to search for information about cats. Action: Search[cats]",
            "Action: Search[cats]",
        ),
        (
            """Thought: Let me analyze this problem.
First, I should check the database.
Action Input: query_users
Observation: Found 3 users
Thought: Now I need to process this data.
Answer: Here are the users...""",
            "Action Input: query_users\nObservation: Found 3 users\nAnswer: Here are the users...",
        ),
        (
            "Normal text without any thoughts. Answer: Direct answer here.",
            "Normal text without any thoughts. Answer: Direct answer here.",
        ),
    ]

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = remove_thoughts(input_text)
        print(f"Test {i}: {'✓' if result == expected else '✗'}")
        if result != expected:
            print(f"  Expected: {repr(expected)}")
            print(f"  Got:      {repr(result)}")

    print("\n=== Testing remove_repeating_sequences ===")
    sequence_tests = [
        (
            "The quick brown fox jumps over the quick brown fox",
            5,
            "The quick brown fox jumps over the quick brown fox",  # No change, threshold 5 is too low for "quick"
        ),
        (
            "The quick brown fox jumps over the quick brown fox",
            10,
            "The quick brown fox jumps over the ",  # "quick brown fox" removed
        ),
        (
            "Repeat repeat repeat repeat this word",
            6,
            "Repeat this word",  # Multiple repetitions removed
        ),
        ("abc123abc123abc123xyz", 4, "abc123xyz"),  # Repeated pattern removed
        ("No repeats here at all", 3, "No repeats here at all"),  # No changes
        (
            "The cat. The cat. The dog. The dog.",
            5,
            "The cat. The dog.",  # Each repeated phrase removed once
        ),
        ("AAAAABBBBBAAAAABBBBBCCCCC", 5, "AAAAABBBBBCCCCC"),  # Repeated blocks removed
    ]

    for i, (input_text, threshold, expected) in enumerate(sequence_tests, 1):
        result = remove_repeating_sequences(input_text, threshold)
        result_opt = remove_repeating_sequences_optimized(input_text, threshold)
        print(f"Test {i} (threshold={threshold}):")
        print(
            f"  Original: {repr(input_text[:50] + '...' if len(input_text) > 50 else input_text)}"
        )
        print(
            f"  Result:   {repr(result[:50] + '...' if len(result) > 50 else result)}"
        )
        print(f"  Match:    {'✓' if result == expected else '✗'}")
        print(f"  Optimized match: {'✓' if result_opt == expected else '✗'}")
        print()

    print("=== Testing combined function ===")
    combined_text = """
Thought: I need to find information about Python.
Action: Search[Python programming]
Observation: Python is a programming language.
Thought: Now I need to explain what Python is.
Answer: Python is a programming language. Python is a programming language. Python is a programming language. It's great for beginners.
"""

    result = remove_thoughts_and_duplicates(combined_text, sequence_threshold=15)
    print("Original length:", len(combined_text))
    print("Processed length:", len(result))
    print("\nProcessed text:")
    print(result)


if __name__ == "__main__":
    test_functions()
