"""
Agent Prompts

Centralized prompt templates for RAG agents.
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a vector database. "
    "You can search documents when needed to answer questions. "
    "Use the vector_retrieval tool only when you need factual "
    "information from documents. "
    "For general knowledge or greetings, answer directly."
)


def get_rag_prompt() -> ChatPromptTemplate:
    """Get the default RAG agent prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

