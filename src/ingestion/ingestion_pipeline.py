import glob
import json
import os
from dotenv import load_dotenv

# 1. Force absolute path calculation for the root-level .env file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(base_dir, ".env")

# 2. Boot environment variables INTO memory BEFORE importing modules that depend on them
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to standard lookups if path resolution varies in specific container environments
    load_dotenv()

# 3. NOW it is safe to import modules requiring API keys
from load_documents import load_pdf_document
from chunk_documents import split_by_section_with_config
from store_embeddings import upsert_to_pinecone

def load_merged_config(base_dir, config_path):
    """
    Loads default base layout properties and overwrites drug overrides.
    Uses unified base_dir to prevent relative path breakage.
    """
    default_path = os.path.join(base_dir, "data", "configs", "default_config.json")
    
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Missing master configuration file at: {default_path}")
        
    with open(default_path, "r") as f:
        config_payload = json.load(f)
        
    with open(config_path, "r") as f:
        specific_override = json.load(f)
        
    config_payload.update(specific_override)
    return config_payload

def main():
    print("="*60)
    print("🚀 CLINICALMIND MASTER SYSTEM INGESTION ORCHESTRATOR")
    print("="*60)

    # Re-verify the absolute project root directory path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_pattern = os.path.join(base_dir, "data", "configs", "*_config.json")
    
    print(f"[System Diagnostic] Root Directory: {base_dir}")
    print(f"[System Diagnostic] Scan Target: {config_pattern}")

    all_configs = [f for f in glob.glob(config_pattern) if "default" not in f]

    print(f"\nFound {len(all_configs)} distinct drug profiles targeting pipeline execution.\n")

    if len(all_configs) == 0:
        print("⚠️ Directives Verification Required:")
        print(f" Ensure your config folder exists at exactly: {os.path.join(base_dir, 'data', 'configs')}")
        print(" Verify that your JSON profiles do not contain typos and end with '.json'")
        return

    for path in all_configs:
        config = load_merged_config(base_dir, path)
        print(f"▶️ Starting Data Lifecycle Pipeline for target: {config['drug_name']}")
        
        try:
            # Step 1: Extract continuous text stream from raw file asset
            raw_text = load_pdf_document(config)
            
            # Step 2: Run section-aware boundary token chunking
            nodes = split_by_section_with_config(config, raw_text)
            
            # --- Architectural Guardrail Check ---
            # Intercept layout parsing failures immediately. If regex failed to capture structural boundaries
            # due to an extraction anomaly, drop execution to protect index metadata partitions from corruption.
            if not nodes:
                print(f"❌ Critical Pipeline Bypassed: Zero database nodes extracted for target '{config['drug_name']}'.")
                print(f"   Review structural heading boundaries inside the source document or fix config valid_sections mappings.\n")
                continue
                
            # Step 3: Embed text fragments and upsert directly into Pinecone partitions
            print(f"📤 Initiating vector synchronization to cloud infrastructure layer...")
            upsert_to_pinecone(nodes)
            print(f"✅ Ingestion chain validated for: {config['drug_name']}\n")
            
        except Exception as e:
            print(f"❌ Critical Pipeline Failure processing asset ({path}): {str(e)}\n")
            continue

if __name__ == "__main__":
    main()