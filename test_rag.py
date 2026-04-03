import os
from dotenv import load_dotenv
from src.generation.rag_engine import FinancialRAG

def main():
    print("🚀 Booting up the Master RAG Engine (Ollama + Qdrant + SQLite)...")
    load_dotenv()
    
    try:
        rag = FinancialRAG()
        print("✅ Engine Ready!\n")
        print("=" * 60)
        
        # ১. আমরা প্রথমে অটোমেটিক ৩টা প্রশ্ন দিয়ে টেস্ট করব (Text, Table, Image)
        # Note: টেবিলের প্রশ্নের জন্য তোমার আগের স্ক্রিনশটের 'AIML_Model_Dev' ডেটাটা ব্যবহার করছি
        test_queries = [
            "What is the end to end model development strategy mentioned in the text?", # Text expected
            "What is the value or description for AIML_Model_Dev in the table?", # Table expected
            "Can you explain the visual trend in the revenue chart?" # Image expected
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"👤 Question [{i}]: {query}")
            print("⏳ AI is thinking, routing, and retrieving context...")
            print(f"\n🤖 AI Answer: ", end="")
            response_generator, context = rag.answer_question(query)
            for chunk in response_generator:
                print(chunk, end="", flush=True)
            print(f"\n\n🔍 Sources:\n{context}\n")
            print("-" * 60)
            
        # ২. এরপর তুমি নিজে লাইভ প্রশ্ন করতে পারবে!
        print("\n💡 Interactive Mode: Now it's your turn! Type a question or 'exit' to quit.")
        while True:
            user_input = input("\n👤 Your Question: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down the engine. Great job today!")
                break
            
            print("⏳ Thinking...")
            print(f"\n🤖 AI Answer: ", end="")
            response_generator, context = rag.answer_question(user_input)
            for chunk in response_generator:
                print(chunk, end="", flush=True)
            print(f"\n\n🔍 Sources:\n{context}\n")
            print("-" * 60)

    except Exception as e:
        print(f"❌ Error during Master RAG test: {e}")

if __name__ == "__main__":
    main()
