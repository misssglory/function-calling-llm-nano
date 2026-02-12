"""Setup utilities for directories and initial data"""

from pathlib import Path
from loguru import logger

from hybrid_search_agent.config import (
    DATA_DIR, LOGS_DIR, SCREENSHOTS_DIR, PDFS_DIR,
    create_directories
)


def prepare_data_directory():
    """Prepare example data directory if it doesn't exist."""
    # Create all directories
    create_directories()
    
    # Create example documents if directory is empty
    if not list(DATA_DIR.glob("*")):
        example_content = """# Company Knowledge Base
        
## About Our Company
We are a technology company specializing in AI solutions.
Our main product is "AI Assistant Pro", released in 2023.
        
## Team Members
- John Doe: CEO and Founder
- Jane Smith: Head of AI Research
- Bob Johnson: Lead Developer
        
## Projects
1. DataAnalyzer: Data analysis and visualization tool
2. DocuSearch: Document search and extraction system
        """
        
        with open(DATA_DIR / "company_info.md", "w") as f:
            f.write(example_content)
        
        logger.info(f"Created example document in {DATA_DIR}/")
    
    logger.info(f"Logs will be saved to {LOGS_DIR}/")
    logger.info(f"Screenshots will be saved to {SCREENSHOTS_DIR}/")
    logger.info(f"PDF files will be saved to {PDFS_DIR}/")

def display_welcome_banner():
    pass