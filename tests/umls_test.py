
"""Quick debug function to see semantic types"""
import os
from dotenv import load_dotenv
load_dotenv()
from src.pipelines.medical_ner_pipeline import UMLSClient

client = UMLSClient(os.environ.get("UMLS_API_KEY"), use_cache=False)

# Test terms
terms = ["diabetes", "aspirin", "heart", "fever"]

for term in terms:
    print(f"\n=== Testing: {term} ===")
    result = client.lookup_term(term)

    print(f"Entity Type: {result['entity_type']}")
    print(f"UMLS Code: {result.get('umls_code')}")
    print(f"Confidence: {result['confidence']}")

    if result.get('metadata', {}).get('semantic_types'):
        print(f"Semantic Types: {result['metadata']['semantic_types']}")

    print("-" * 30)