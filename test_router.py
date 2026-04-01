import os
from src.retrieval.router import QueryRouter

def main():
    print("🚀 Initializing Query Router (Connecting to OpenAI)...")
    
    # API Key চেক করা
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY is not set in your .env file!")
        return
        
    try:
        router = QueryRouter()
        
        # আমরা ৩ ধরনের প্রশ্ন দিয়ে টেস্ট করব
        test_queries = [
            "What is the company's core business strategy and risk factors?",  # Expected: text
            "What was the exact total revenue and gross margin for Q3 2023?",   # Expected: table
            "How does the revenue trend look over the last 5 years based on the bar chart?" # Expected: image
        ]
        
        print("\n🧠 Testing LLM Routing Logic...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"[{i}] Question: '{query}'")
            print("⏳ Routing...")
            result = router.route_query(query)
            print(f"✅ Decision: {result}\n")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error during Routing test: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # .env ফাইল থেকে API key লোড করার জন্য
    main()
