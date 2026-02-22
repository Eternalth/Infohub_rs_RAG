from config import Config
from vector_store import VectorStore
from agent import RAGAgent

def main():
    try:
        config = Config.from_env()
        config.validate()
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        agent = RAGAgent(config, vector_store)
        agent.initialize()
        
        print("\n" + "█" * 72)
        print(" 🤖 GEMINI RAG CHATBOT (Type 'exit' to quit)")
        print("█" * 72 + "\n")
        
        while True:
            user_input = input("\nთქვენ (You): ")
            
            if user_input.lower() in ['exit', 'quit', 'გამოსვლა']:
                print("ნახვამდის! (Goodbye!)")
                break
                
            if not user_input.strip():
                continue
                
            try:
                answer = agent.chat(user_input)
                
                print("\n" + "=" * 72)
                print(f"აგენტი (Agent):\n{answer}")
                print("=" * 72)
                
            except Exception as e:
                print(f"\n[Error]: {str(e)}")
                
    except Exception as e:
        print(f"\n[Fatal Error]: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have created a .env file with GEMINI_API_KEY")
        print("2. You have run crawler.py to create the vector database")
        print("3. All required dependencies are installed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
