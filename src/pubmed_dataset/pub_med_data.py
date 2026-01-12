from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Bio import Entrez
import os
from dotenv import load_dotenv

print("PUBMED CHROMA DB CREATION")

load_dotenv(verbose=True)

# Setting up email for PubMed API 
Entrez.email = os.environ.get("Entrez.email", "abhisheksj009@gmail.com") 
Entrez.api_key = os.environ.get("Entrez.api_key")

print(f"\n✓ Entrez email: {Entrez.email}")
print(f"✓ API key configured: {Entrez.api_key is not None}\n")

def fetch_pubmed_articles(query, max_results=100):
    """Fetch articles from PubMed"""
    print(f"Searching PubMed: '{query}' (max {max_results} results)")
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        print(f"Found {len(id_list)} article IDs")
        
        if not id_list:
            print("No articles found!")
            return None
        
        # Fetch details
        print(f"Fetching article details...")
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        articles = Entrez.read(handle)
        handle.close()
        
        print(f"Fetched {len(articles.get('PubmedArticle', []))} full articles\n")
        return articles
        
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return None


def create_pubmed_chromadb(query, collection_name="pubmed_articles"):
    # Fetch articles
    articles = fetch_pubmed_articles(query, max_results=1000)
    
    if not articles or 'PubmedArticle' not in articles:
        print("No articles to process")
        return None
    
    # Prepare documents
    documents = []
    metadatas = []
    
    print(f"Processing {len(articles['PubmedArticle'])} articles...")
    
    for i, article in enumerate(articles['PubmedArticle'], 1):
        try:
            article_data = article['MedlineCitation']['Article']
            title = article_data['ArticleTitle']
            abstract = article_data.get('Abstract', {}).get('AbstractText', [''])[0]
            pmid = article['MedlineCitation']['PMID']
            
            full_text = f"{title}\n\n{abstract}"
            documents.append(full_text)
            
            metadatas.append({
                "pmid": str(pmid),
                "title": title,
                "source": "pubmed"
            })
            
            if i % 10 == 0:
                print(f"  ✓ Processed {i}/{len(articles['PubmedArticle'])}")
                
        except Exception as e:
            print(f"Skipped article {i}: {e}")
            continue
    
    print(f"Prepared {len(documents)} documents\n")
    
    if not documents:
        print("No valid documents")
        return None
    
    # Split documents
    print(f"Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.create_documents(documents, metadatas=metadatas)
    print(f"✓ Created {len(splits)} chunks\n")
    
    # Initialize embeddings
    print(f"Loading embedding model (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"✓ Model loaded\n")
    
    # Create Chroma DB
    print(f"Creating Chroma database...")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_pubmed_db"  # Simpler path
    )
    print(f"Database created: {vectordb._collection.count()} vectors\n")
    
    return vectordb


# if __name__ == "__main__":
#     # Create the database
#     vectordb = create_pubmed_chromadb(
#         "machine learning for medical", 
#         collection_name="ml_medical_1000"
#     )
    
#     if vectordb:
#         print("QUERYING DATABASE")

        
#         query = "What are the applications of deep learning in radiology?"
#         print(f"\n Query: {query}\n")
        
#         docs = vectordb.similarity_search(query, k=3)
#         print(f"Found {len(docs)} relevant documents\n")
        
#         for i, doc in enumerate(docs, 1):
#             print(f"RESULT {i}")
#             print(f"PMID: {doc.metadata.get('pmid', 'N/A')}")
#             print(f"Title: {doc.metadata.get('title', 'N/A')[:80]}...")
#             print(f"\nContent Preview:\n{doc.page_content[:250]}...")
#             print()
#     else:
#         print("\nFailed to create database")
    
#     print("DONE!")
