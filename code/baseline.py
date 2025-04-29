# Import dependencies
import arxiv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoConfig
from langchain.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.docstore.document import Document
from huggingface_hub import login
from semanticscholar import SemanticScholar
from IPython.display import display_markdown
from dotenv import load_dotenv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Huggingface login
load_dotenv()
login(os.getenv("HUGGINGFACE_API_KEY"))

###### Fetching data from academic sources ######
def fetch_papers(query, max_papers=5, sources=None):
    papers_arxiv = fetch_arxiv_papers(query, max_papers)
    papers_semantic = fetch_semantic_scholar_papers(query, max_papers)
    papers = papers_arxiv + papers_semantic

    return papers

# Arxiv
def fetch_arxiv_papers(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "authors": [author.name for author in result.authors],
            "published": str(result.published.date()),
            "url": result.entry_id, # URL to the arxiv page
            "pdf_url": result.pdf_url, # URL to the actual paper
            "source": "arXiv"
        })
    return papers

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

# TODO: Add more sources

def create_generative_model_and_tokenizer():
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Model config
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
    )

    # Generative model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
    )

    #Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL
    )

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"

    return model, tokenizer

def generate_prompt_template():
    PROMPT_TEMPLATE = """
    You are a helpful AI QA assistant. When answering questions, use the context enclosed by triple backquotes if it is relevant.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Reply your answer in markdown format.

    ```
    {context}
    ```

    ### Question:
    {question}

    ### Answer:
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE.strip(),
    )

    return prompt_template

def answer_question(question: str, llm_chain, history: dict[str] = None) -> str:
    if history is None:
        history = []

    response = llm_chain.invoke({"question": question, "chat_history": history})
    answer = response["answer"].split("### Answer:")[-1].strip()
    return answer


if __name__ == "__main__":

    query = "vision transformers" # Enter query to retrieve papers (should be in keywords, not a full sentence)
    max_papers = 3
    papers = fetch_papers(query, max_papers)

    model, tokenizer = create_generative_model_and_tokenizer()

    embedding = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
    )
    documents = [Document(page_content=paper["summary"], metadata=paper) for paper in papers]
    vectorstore = FAISS.from_documents(documents, embedding)

    generate_text = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        max_new_tokens=8192,
        repetition_penalty=1.1,
        # do_sample=True,
        # temperature=0.7,
        # top_p=0.95
    )

    llm_chain = ConversationalRetrievalChain.from_llm(
        llm=HuggingFacePipeline(pipeline=generate_text),
        retriever=vectorstore.as_retriever(search_type="similarity", k=3),
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": generate_prompt_template()},
        verbose=False,
    )

    question = "For which tasks are vision transformers used?" # Should be a full sentence, not keywords like for the query
    display_markdown(answer_question(question, llm_chain), raw=True)