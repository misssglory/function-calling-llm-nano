"""
Hybrid Search Agent with Step-by-Step Execution
Main package initialization
"""

from hybrid_search_agent.core.hybrid_agent import HybridSearchAgent
from hybrid_search_agent.sessions.interactive import interactive_chat_session, create_new_session, quick_query
from hybrid_search_agent.utils.setup import prepare_data_directory

__version__ = "1.0.0"
__all__ = [
    'HybridSearchAgent',
    'interactive_chat_session',
    'create_new_session',
    'quick_query',
    'prepare_data_directory',
]