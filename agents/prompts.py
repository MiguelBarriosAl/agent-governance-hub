"""
Agent Prompts

Centralized prompt templates for governed RAG agents.
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


RAG_SYSTEM_PROMPT = (
    "Tú eres un agente gobernado. "
    "Debes razonar usando ReAct (Thought → Action → Observation). "
    "Para decidir si usas herramientas o respondes directamente, "
    "piensa paso a paso. "
    "Si necesitas documentos, llama a la herramienta vector_retrieval. "
    "Nunca ejecutes una acción que no esté permitida por la capa "
    "de governance."
)


def get_rag_prompt() -> ChatPromptTemplate:
    """Get the governed RAG agent prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

