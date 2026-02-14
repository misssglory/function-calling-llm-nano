#!/usr/bin/env python3
"""
project_statistics.py - Count lines in project files excluding JSON files and specified folders.
Displays results in a tree format with line counts per file and visual size bars.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from loguru import logger
import logging


def configure_logger():
    """Configure loguru logger to show date and time without colors, names, and paths."""
    # Remove default logger
    logger.remove()
    
    # Add custom format with date and time
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} - {message}",
        colorize=False,
        level="INFO"
    )
    
    # Suppress loguru's internal logging
    logging.getLogger("loguru").setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Count lines in project files excluding JSON files and specified folders.'
    )
    parser.add_argument(
        'exclude_folders',
        nargs='*',
        default=[],
        help='Folders to exclude from line counting (space-separated)'
    )
    parser.add_argument(
        '--path',
        '-p',
        default='.',
        help='Root path to analyze (default: current directory)'
    )
    parser.add_argument(
        '--max-bar-length',
        '-m',
        type=int,
        default=30,
        help='Maximum length of the size bar in characters (default: 30)'
    )
    return parser.parse_args()


def should_exclude(file_path, exclude_folders):
    """Check if file should be excluded based on folder or extension."""
    # Check if it's a JSON file
    if file_path.suffix.lower() == '.json':
        return True
    
    # Check if file is in any excluded folder
    for folder in exclude_folders:
        if folder in file_path.parts:
            return True
    
    return False


def count_lines_in_file(file_path):
    """Count lines in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, PermissionError, OSError):
        # Skip binary files or files that can't be read
        return 0


def calculate_bar_length(lines, max_lines, max_bar_length):
    """Calculate the length of the size bar based on line count."""
    if max_lines == 0:
        return 0
    return int((lines / max_lines) * max_bar_length)


def create_size_bar(length, max_length):
    """Create a visual bar representation."""
    if length == 0:
        return ""
    
    # Use different characters for different portions of the bar
    filled = "█" * length
    empty = "░" * (max_length - length)
    return f"[{filled}{empty}]"


def get_size_category(lines):
    """Categorize file size for visual indicators."""
    if lines < 10:
        return "tiny"
    elif lines < 50:
        return "small"
    elif lines < 200:
        return "medium"
    elif lines < 500:
        return "large"
    else:
        return "huge"


def build_file_tree(root_path, exclude_folders):
    """Build a tree structure of files with their line counts."""
    file_tree = {}
    total_lines = 0
    file_count = 0
    max_lines_in_tree = 0
    
    root = Path(root_path).resolve()
    logger.info(f"Scanning directory: {root}")
    
    # First pass: collect all files and find maximum line count
    temp_tree = {}
    for file_path in root.rglob('*'):
        if not file_path.is_file():
            continue
            
        if should_exclude(file_path, exclude_folders):
            continue
        
        lines = count_lines_in_file(file_path)
        if lines > 0:  # Only include files with readable content
            rel_path = file_path.relative_to(root)
            temp_tree[str(rel_path)] = lines
            max_lines_in_tree = max(max_lines_in_tree, lines)
    
    # Second pass: build final tree with bar calculations
    for rel_path, lines in temp_tree.items():
        file_tree[rel_path] = {
            'lines': lines,
            'bar_length': calculate_bar_length(lines, max_lines_in_tree, args.max_bar_length),
            'category': get_size_category(lines)
        }
        total_lines += lines
        file_count += 1
        
        # Log progress every 100 files
        if file_count % 100 == 0:
            logger.info(f"Processed {file_count} files...")
    
    logger.info(f"Found {file_count} files with {total_lines} total lines")
    logger.info(f"Largest file: {max_lines_in_tree} lines")
    return file_tree, total_lines, max_lines_in_tree


def print_tree(file_tree, total_lines, max_lines, max_bar_length):
    """Print the file tree with line counts and size bars."""
    if not file_tree:
        logger.warning("No files found to analyze.")
        return
    
    # Sort files for consistent output
    sorted_files = sorted(file_tree.items())
    
    # Group files by directory for tree-like display
    tree_structure = defaultdict(list)
    for file_path, data in sorted_files:
        parts = file_path.split(os.sep)
        if len(parts) == 1:
            # File in root directory
            tree_structure['.'].append((file_path, data))
        else:
            # File in subdirectory
            dir_path = os.sep.join(parts[:-1])
            tree_structure[dir_path].append((file_path, data))
    
    # Print tree structure
    print("\n📊 Project Line Count Statistics")
    print("=" * 70)
    print(f"Max bar length: {max_bar_length} chars | Each '█' ≈ {max_lines/max_bar_length:.1f} lines")
    print("=" * 70)
    
    # Sort directories for consistent output
    sorted_dirs = sorted(tree_structure.keys())
    
    for dir_path in sorted_dirs:
        if dir_path == '.':
            print(f"\n📁 ./")
        else:
            print(f"\n📁 {dir_path}/")
        
        files = tree_structure[dir_path]
        # Sort files within directory by line count (descending)
        files.sort(key=lambda x: x[1]['lines'], reverse=True)
        
        for i, (file_path, data) in enumerate(files):
            if dir_path == '.':
                display_name = file_path
            else:
                display_name = os.path.basename(file_path)
            
            # Determine tree prefix
            if i == len(files) - 1:
                prefix = "└── "
            else:
                prefix = "├── "
            
            # Create size bar
            bar = create_size_bar(data['bar_length'], max_bar_length)
            
            # Add size category emoji
            category_emoji = {
                "tiny": "🔹",
                "small": "🔸",
                "medium": "📄",
                "large": "📚",
                "huge": "🗄️"
            }.get(data['category'], "📄")
            
            # Print file with bar
            lines_display = f"{data['lines']:5d} lines"
            print(f"{prefix}{category_emoji} {display_name:<30} {lines_display} {bar}")
    
    print("\n" + "=" * 70)
    print(f"📈 Total lines: {total_lines}")
    print(f"📁 Total files: {len(file_tree)}")
    print(f"📊 Largest file: {max_lines} lines")
    
    # Print size category legend
    print("\n📋 Size Categories:")
    print("   🔹 Tiny (<10 lines)    🔸 Small (10-49 lines)    📄 Medium (50-199 lines)")
    print("   📚 Large (200-499 lines)    🗄️ Huge (500+ lines)")


def main():
    """Main function to run the script."""
    global args  # Make args available to build_file_tree
    # Configure logger first
    configure_logger()
    
    args = parse_arguments()
    
    # Log initial configuration
    logger.info(f"Analyzing path: {args.path}")
    logger.info(f"Excluding folders: {args.exclude_folders if args.exclude_folders else 'None'}")
    logger.info(f"Excluding files: .json")
    logger.info(f"Max bar length: {args.max_bar_length} characters")
    
    try:
        file_tree, total_lines, max_lines = build_file_tree(args.path, args.exclude_folders)
        print_tree(file_tree, total_lines, max_lines, args.max_bar_length)
    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()