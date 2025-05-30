import arxiv
from semanticscholar import SemanticScholar
from acl_anthology import Anthology
# For Google Scholar and CORE APIs
import requests
# Regex
import re
# Playwright for ResearchGate
from parsel import Selector
from playwright.sync_api import sync_playwright
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import langchain_community
import os
import hashlib
import json
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL=HuggingFaceEmbeddings(model_name="/d/hpc/projects/onj_fri/laandi/models/sentence-transformer")

# ArXiv
def fetch_arxiv_papers(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "authors": [author.name for author in result.authors],
            "published": str(result.published.date()), #example '2017-03-04'
            "url": result.entry_id, # URL to the arxiv page
            "pdf_url": result.pdf_url, # URL to the actual paper
            "source": "arXiv"
        })
    return papers

    # ## Testing with category

    # search = arxiv.Search(query="cat:cs.CL", max_results=5000, sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending)
    # results = None
    # try:
    #     results = list(search.results())  # Pull all results at once (until arXiv stops)
    # except Exception as e:
    #     print("Reached the end of arXiv results.")
    #     return []

    # papers = []
    # for result in results:
    #     date_ = result.published.date() if result.published else None
    #     year = date_.year if date_ else "Unknown"
    #     year = int(year) if year != "Unknown" else "Unknown"
    #     if year < 2023:
    #         continue
    #     papers.append({
    #         "title": result.title,
    #         "summary": result.summary,
    #         "authors": [author.name for author in result.authors],
    #         "published": year, #example '2017-03-04'
    #         "url": result.entry_id, # URL to the arxiv page
    #         "pdf_url": result.pdf_url, # URL to the actual paper
    #         "source": "arXiv"
    #     })
    # return papers

# Semantic Scholar
def fetch_semantic_scholar_papers(query, max_results=5):
    sch = SemanticScholar()
    papers = sch.search_paper(query, limit=max_results)

    results = []
    for i, paper in enumerate(papers):
        if i >= max_results:
            break

        results.append({
            "title": paper.title,
            "summary": paper.abstract if paper.abstract else "No abstract available.",
            "authors": [author.name for author in paper.authors],
            "published": paper.year,
            "url": paper.url,
            "source": "Semantic Scholar"
        })

    return results

# GoogleScholar
def fetch_googlescholar_papers(query, max_results=5):
    api_key = os.getenv("GOOGLE_SCHOLAR_API")
    search_url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google_scholar",
        "api_key": api_key,
        #"as_ylo": "2022",
    }

    response = requests.get(search_url, params=params)
    data = response.json()


    papers = []
    if data.get('organic_results') is not None:
        for result in data.get('organic_results'):
            # Extract authors from publication_info['summary'] if available
            authors = "Unknown"  # Default value
            publication_summary = result.get('publication_info', {}).get('summary', '')
            authors_match = re.search(r"([A-Za-z,]+(?:\s[A-Za-z]+)+)\s-\s", publication_summary)
            if authors_match:
                authors = authors_match.group(1)

            # Extract the published year from summary or snippet
            published = "Unknown"  # Default value
            published_match = re.search(r"(\d{4})", publication_summary) or re.search(r"(\d{4})", result.get('snippet', ''))
            if published_match:
                published = published_match.group(1)

            papers.append({
                "title": result.get('title'),
                "summary": result.get('snippet'),
                "authors": authors,
                "published": published, #year
                "url": result.get('link'),
                "source": "GoogleScholar"
            })

    return papers


def fetch_researchgate_papers(query: str, max_papers: int = 5):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, slow_mo=50)
        page = browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36")
        
        # Only go to the first page
        page.goto(f"https://www.researchgate.net/search/publication?q={query}&page=1")
        page.wait_for_timeout(2000)  # Wait for page to load
        selector = Selector(text=page.content())

        papers = []
        for publication in selector.css(".nova-legacy-c-card__body--spacing-inherit")[:max_papers]:
            title = publication.css(".nova-legacy-v-publication-item__title .nova-legacy-e-link--theme-bare::text").get()
            if not title:
                continue
                
            title = title.title()
            title_link = f'https://www.researchgate.net{publication.css(".nova-legacy-v-publication-item__title .nova-legacy-e-link--theme-bare::attr(href)").get()}'
            publication_date = publication.css(".nova-legacy-v-publication-item__meta-data-item:nth-child(1) span::text").get()
            authors = publication.css(".nova-legacy-v-person-inline-item__fullname::text").getall()

            papers.append({
                "title": title,
                "url": title_link,
                "authors": authors,
                "published": publication_date,
                "source": "ResearchGate"
            })

        browser.close()
        return papers

# CORE
def fetch_core_papers(query: str, max_papers: int = 5):
    """Fetch papers from CORE API (https://api.core.ac.uk/docs/v3)"""
    CORE_API_KEY = os.getenv("CORE_API")
    
    headers = {
        "Authorization": f"Bearer {CORE_API_KEY}",
        "Content-Type": "application/json"
    }

    # date = "yearPublished:2022 OR yearPublished:2023 OR yearPublished:2024 OR yearPublished:2025"
    # query = query + " " + date
    
    params = {
        "q": query,
        "limit": max_papers,
        "offset": 0
    }
    
    try:
        response = requests.get(
            "https://api.core.ac.uk/v3/search/works",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        data = response.json()
        
        papers = []
        for result in data.get("results", [])[:max_papers]:
            # Extract basic information
            title = result.get("title", "No title available")
            abstract = result.get("abstract", "No abstract available")
            year = result.get("yearPublished", "Unknown")
            doi = result.get("doi", None)
            
            # Extract authors
            authors = []
            for author in result.get("authors", []):
                if "name" in author:
                    authors.append(author["name"])
                elif "fullName" in author:
                    authors.append(author["fullName"])
            
            # Get PDF URL if available
            pdf_url = None
            for download_url in result.get("downloadUrl", []):
                if download_url.endswith(".pdf"):
                    pdf_url = download_url
                    break
            
            papers.append({
                "title": title,
                "summary": abstract,
                "authors": authors if authors else ["Unknown"],
                "published": str(year),
                "url": result.get("url", f"https://doi.org/{doi}" if doi else None),
                "pdf_url": pdf_url,
                "source": "CORE"
            })
            
        return papers
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from CORE API: {e}")
        return []

def decode_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return None
    word_list = [''] * (max([max(positions) for positions in abstract_inverted_index.values()]) + 1)
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_list[pos] = word
    return ' '.join(word_list)

# def fetch_openalex_papers_2(query, max_papers=5):
#     per_page = 200
#     max_pages = None
#     endpoint = "https://api.openalex.org/works"
#     cursor = "*"
#     all_results = []
#     count = 0

#     while cursor and (max_pages is None or count < max_pages):
#         params = {
#             "filter": "concepts.id:C204321447,from_publication_date:2023-01-01,cited_by_count:>10",  # NLP concept ID in OpenAlex, also Natural Language id: C195324797
#             "per-page": per_page,
#             "cursor": cursor,
#         }
#         response = requests.get(endpoint, params=params)
#         data = response.json()

#         results = data.get("results", [])
#         all_results.extend(results)

#         cursor = data.get("meta", {}).get("next_cursor")
#         count += 1

#         if not results or not cursor:
#             break

#     papers = []
#     for result in all_results:
#         try:
#             # Extract basic information
#             title = result.get("title", "No title available")
#             abstract = decode_abstract(result.get('abstract_inverted_index'))
#             if not abstract:
#                 continue
#             year = result.get("publication_year", "Unknown")
#             doi = result.get('doi', result.get('id'))
            
#             # Extract authors
#             authors = []
#             for author in result.get("authorships", []):
#                 authors.append(author["author"]["display_name"])
            
#             papers.append({
#                 "title": title,
#                 "summary": abstract,
#                 "authors": authors if authors else ["Unknown"],
#                 "published": str(year),
#                 "url": result.get("url", f"https://doi.org/{doi}" if doi else None),
#                 "source": "OpenAlex"
#             })
#         except Exception as e:
#             print(f"Error processing result: {e}")
#             continue
#     return papers

def fetch_openalex_papers(query: str, max_papers: int = 5):
    
    url = (f"https://api.openalex.org/works?search={query}")
      
    try:
        # Search for concept ID - broader
        # concept_id = requests.get(f"https://api.openalex.org/concepts?search={query}")
        # response.raise_for_status()
        # data = response.json()

        # Search for articles with concept ID
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        papers = []
        for result in data.get("results", [])[:max_papers]:
            try:
                # Extract basic information
                title = result.get("title", "No title available")
                abstract = decode_abstract(result.get('abstract_inverted_index'))
                year = result.get("publication_year", "Unknown")
                doi = result.get('doi', result.get('id'))
                
                # Extract authors
                authors = []
                for author in result.get("authorships", []):
                    authors.append(author["author"]["display_name"])
                
                papers.append({
                    "title": title,
                    "summary": abstract,
                    "authors": authors if authors else ["Unknown"],
                    "published": str(year),
                    "url": result.get("url", f"https://doi.org/{doi}" if doi else None),
                    "source": "OpenAlex"
                })
            except Exception as e:
                print(f"Error processing result: {e}")
                continue
            
        return papers
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from OpenAlex: {e}")
        return []

def fetch_acl_papers(query: str, max_papers: int = 5):
    # Used just for database construction
    anthology = Anthology.from_repo()
    acl_papers = list(anthology.papers())

    papers = []
    for paper in acl_papers:
        try:
            if paper.abstract is not None and int(paper.year) > 2022:
                papers.append({
                    "title": str(paper.title),
                    "summary": str(paper.abstract),
                    "authors": [author.name.first + ' ' + author.name.last for author in paper.authors],
                    "published": str(paper.year),
                    "url": paper.web_url,
                    "source": "ACL"
                })
        except Exception as e:
            print(f"Error processing ACL paper {paper.title}: {e}")
            continue
    
    return papers
    
def fetch_papers(
        query, 
        max_papers=5, 
        arxiv=True, 
        semantic_scholar=False, 
        googlescholar=True, 
        researchgate=False, 
        core=True,
        openalex=True,
        acl=False
    ):
    papers = []
    if not any([arxiv, semantic_scholar, googlescholar, researchgate, core, openalex, acl]):
        raise ValueError("At least one source must be selected.")
    if max_papers <= 0:
        raise ValueError("max_papers must be greater than 0.")
    if arxiv:
        papers_arxiv = fetch_arxiv_papers(query, max_papers)
        papers.extend(papers_arxiv)
    if semantic_scholar:
        papers_semantic = fetch_semantic_scholar_papers(query, max_papers)
        papers.extend(papers_semantic)
    if googlescholar:
        papers_googlescholar = fetch_googlescholar_papers(query, max_papers)
        papers.extend(papers_googlescholar)
    if researchgate:
        papers_researchgate = fetch_researchgate_papers(query, max_papers)
        papers.extend(papers_researchgate)
    if core:
        papers_core = fetch_core_papers(query, max_papers)
        papers.extend(papers_core)
    if openalex:
        papers_openalex = fetch_openalex_papers(query, max_papers)
        papers.extend(papers_openalex)
    if acl:
        papers_acl = fetch_acl_papers(query, max_papers)
        papers.extend(papers_acl)
    
    return papers

def format_papers_into_documents(papers):
    return [Document(page_content=paper["summary"], metadata=paper) for paper in papers if paper["summary"]]

def doc_hash(doc: Document) -> str:
    # Hash based on content and metadata
    m = hashlib.sha256()
    m.update(doc.page_content.encode("utf-8"))
    # m.update(json.dumps(doc.metadata, sort_keys=True).encode("utf-8"))
    return m.hexdigest()

def create_vectorstore(documents, persist_directory="./knowledgebase2"):
    embedding = EMBEDDING_MODEL
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embedding
    )

    # Delete the dummy document
    vectorstore.docstore._dict.clear()
    vectorstore.index.reset()

    # Save the vectorstore to the specified directory
    vectorstore.save_local(persist_directory)
    return vectorstore

def load_vectorstore(persist_directory="./knowledgebase2"):
    embedding = EMBEDDING_MODEL
    if bool(os.listdir(persist_directory)):
        vectorstore = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)
    else:
        dummy_doc = Document(page_content="placeholder", metadata={})
        vectorstore = create_vectorstore([dummy_doc], persist_directory)
        print("New vectorstore created as it was not found.")
    return vectorstore

def load_document_hashes(persist_directory="./knowledgebase2/document_hashes"):
    """Load the set of document hashes that have already been embedded."""
    hash_file = os.path.join(persist_directory, "document_hashes.json")
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"Error loading document hashes: {e}")
            return set()
    return set()

def save_document_hashes(hashes, persist_directory="./knowledgebase2/document_hashes"):
    """Save the set of document hashes that have been embedded."""
    # Ensure the directory exists
    os.makedirs(persist_directory, exist_ok=True)
    hash_file = os.path.join(persist_directory, "document_hashes.json")
    try:
        with open(hash_file, 'w') as f:
            json.dump(list(hashes), f)
    except Exception as e:
        print(f"Error saving document hashes: {e}")

def update_vectorstore(db, documents, persist_directory="./knowledgebase2", skip_duplicates=True):
    """
    Update the vectorstore with new documents, skipping duplicates if requested.
    
    Args:
        db: The existing vectorstore, or None to load from disk
        documents: List of documents to add
        persist_directory: Directory to save the vectorstore
        skip_duplicates: Whether to skip documents already in the store
    """
    # If vectorstore is not passed, load it from the directory
    if db is None:
        embedding = EMBEDDING_MODEL
        try:
            vectorstore = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)
        except Exception:
            vectorstore = None
    # If it was passed, just use that
    else:
        vectorstore = db
    
    # Load existing document hashes
    existing_hashes = load_document_hashes(persist_directory+'/document_hashes') if skip_duplicates else set()
    
    # Filter out documents that are already in the vectorstore
    new_documents = []
    new_hashes = set()
    duplicates_count = 0
    
    for doc in documents:
        doc_signature = doc_hash(doc)
        if not skip_duplicates or doc_signature not in existing_hashes:
            new_documents.append(doc)
            new_hashes.add(doc_signature)
        else:
            duplicates_count += 1
    
    # If the vectorstore could not be loaded, create a new one with the documents
    if vectorstore is None:
        if new_documents:
            vectorstore = create_vectorstore(new_documents, persist_directory)
            print("New vectorstore created as it was not found.")
            # Update document hashes
            save_document_hashes(new_hashes, persist_directory+'/document_hashes')
        else:
            print("No new documents to add.")
            return None
    # Else, add the documents to the existing vectorstore
    elif new_documents:
        vectorstore.add_documents(new_documents)
        # Save the updated vectorstore
        vectorstore.save_local(persist_directory)
        # Update document hashes with new documents
        all_hashes = existing_hashes.union(new_hashes)
        save_document_hashes(all_hashes, persist_directory+'/document_hashes')
        print(f"Updated vectorstore with {len(new_documents)} new documents. Skipped {duplicates_count} duplicates.")
    else:
        print(f"No new documents to add. Skipped {duplicates_count} duplicates.")

    return vectorstore