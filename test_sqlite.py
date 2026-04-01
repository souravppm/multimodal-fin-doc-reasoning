import os
import glob
from src.storage.sqlite_store import SQLiteTableStore

def main():
    print("🚀 Initializing SQLite Table Store...")
    db_path = "data/processed/financial_tables.db"
    
    try:
        store = SQLiteTableStore(db_path=db_path)
        
        table_dir = "data/processed/tables/"
        csv_files = glob.glob(os.path.join(table_dir, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {table_dir}. Please run ingestion first.")
            return
            
        print(f"📂 Found {len(csv_files)} CSV files. Loading into database...")
        
        first_table_name = None
        
        for file_path in csv_files:
            # ফাইলের নাম থেকে টেবিলের নাম তৈরি করা হচ্ছে (যেমন: sample_report_p1_table_1)
            base_name = os.path.basename(file_path)
            table_name = os.path.splitext(base_name)[0]
            
            if first_table_name is None:
                first_table_name = table_name
                
            print(f"⏳ Loading '{base_name}' into table '{table_name}'...")
            try:
                store.load_csv_to_table(file_path, table_name)
                print(f"✅ Successfully loaded {table_name}")
            except Exception as e:
                print(f"❌ Failed to load {table_name}: {e}")
                
        # টেস্টিং SQL কুয়েরি
        if first_table_name:
            print(f"\n🔍 Testing SQL query on table: '{first_table_name}'...")
            query = f"SELECT * FROM {first_table_name} LIMIT 3;"
            try:
                result_df = store.execute_query(query)
                print("\n✅ Query Result (First 3 rows):\n")
                print(result_df)
            except Exception as e:
                print(f"❌ Query failed: {e}")

    except Exception as e:
        print(f"❌ Error during SQLite test: {e}")

if __name__ == "__main__":
    main()
