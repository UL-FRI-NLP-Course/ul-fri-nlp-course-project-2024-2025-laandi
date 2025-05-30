from papers_knowledgebase import (
    fetch_papers,
    format_papers_into_documents,
    update_vectorstore,
    load_vectorstore
)

def fetch_and_save_papers(keywords, max_docs=5):
    """
    Fetches papers related to the given keywords and saves them to a vector store.
    
    Args:
        keywords (str): Search query for paper fetching
        max_docs (int): Maximum number of papers to fetch from each source
        persist_directory (str): Directory to store the vector database
        
    Returns:
        The updated vector store object
    """
    # Fetch papers from various sources
    papers = fetch_papers(
        query=keywords,
        max_papers=max_docs,
        arxiv=False,
        semantic_scholar=False,
        googlescholar=False,
        researchgate=False,
        core=False,
        openalex=True,
        acl=False
    )
    
    print(f"Fetched {len(papers)} papers related to '{keywords}'")
    
    # Convert papers to document format
    documents = format_papers_into_documents(papers)
    
    # Try to load existing vectorstore or create a new one if it doesn't exist
    try:
        vectorstore = load_vectorstore()
    except:
        vectorstore = None
    
    # Update the vectorstore with new documents
    updated_vectorstore = update_vectorstore(
        db=vectorstore,
        documents=documents
    )
    
    return updated_vectorstore
