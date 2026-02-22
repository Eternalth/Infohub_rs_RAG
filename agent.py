from typing import List
import google.generativeai as genai
from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from vector_store import VectorStore


class RAGAgent:
    
    SYSTEM_PROMPT = """You are an expert assistant for the Georgian Revenue Service (RS). 
Your job is to answer tax and customs-related questions accurately.
Use the provided context from the official Infohub database to answer questions.
If the context does not contain the answer, politely state that you do not have that information based on the RS database.

CITATION RULES:
- You MUST always cite the source documents you used in your answer.
- Cite documents by their title exactly as they appear in the context, using the format: [დოკუმენტის სახელი]
- You must use the title of document and NOT system title like [დოკუმენტი 1]
- Every factual claim must be followed by its citation.
- If multiple documents support a claim, cite all of them.
- At the end of your answer, include a "წყაროები:" (Sources) section listing all cited documents.

IMPORTANT: You MUST answer the user's question completely in Georgian (ქართული ენა)."""
    
    def __init__(self, config: Config, vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.chat_history: List = []
        self.model = None
        
    def initialize(self) -> None:
        print("Initializing Gemini LLM...")
        genai.configure(api_key=self.config.gemini_api_key)
        self.model = genai.GenerativeModel(self.config.llm_model)
        print("Agent initialized successfully!\n")
    
    def chat(self, user_input: str) -> str:
        context = self.vector_store.search(user_input)
        
        history_text = ""
        for msg in self.chat_history[-6:]:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"
        
        prompt = f"""{self.SYSTEM_PROMPT}

Retrieved context (each document is labeled with its title in square brackets):
{context}

{history_text}
User: {user_input}
Assistant:"""
        
        response = self.model.generate_content(prompt)
        answer = response.text
        
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=answer))
        
        return answer
    
    def reset_history(self) -> None:
        self.chat_history = []