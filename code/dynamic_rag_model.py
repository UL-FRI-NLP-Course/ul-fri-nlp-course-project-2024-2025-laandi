import torch
import huggingface_hub
import papers_knowledgebase

from keybert import KeyBERT

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate

from IPython.display import display_markdown


class DynamicRAG:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MODEL_NAME = "../../models/mistral-7b"

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, device_map="auto")
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,
            max_new_tokens=8192,
            repetition_penalty=1.1,
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.95
        )
        self.PROMPT_TEMPLATE = """
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
        self.system_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.PROMPT_TEMPLATE.strip(),
        )


    def answer_question(self, question: str, llm_chain, history: dict[str] = None) -> str:
        if history is None:
            history = []
        response = llm_chain.invoke({"question": question, "chat_history": history})
        answer = response["answer"].split("### Answer:")[-1].strip()
        return answer

    def generate(self, question, vectorstore=None):
        if vectorstore is None:
            vectorstore = papers_knowledgebase.load_vectorstore()
        """### This here can be used to set a similarity threshold for the search, 
                and additionally fetch new papers if needed
        # Perform a similarity search with a threshold
        docs_with_scores = vectorstore.similarity_search_with_score(
            question,
            k=3,
            # search_kwargs={"score_threshold": 0.7}  # Add similarity threshold here
        )
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"Document {i+1} (Similarity: {score:.4f}):")
            print(f"Title: {doc.metadata.title}")
            print(f"URL: {doc.metadata.url}")
        ###"""
        chain = ConversationalRetrievalChain.from_llm(
            llm=HuggingFacePipeline(pipeline=self.pipeline),
            retriever=vectorstore.as_retriever(
                search_type="similarity", 
                k=3,
                # search_kwargs={"score_threshold": 0.7}  # Add similarity threshold here
            ),
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.system_prompt},
            verbose=False
        )
        return self.answer_question(question, chain)

    def query(self, question, keywords = None, searchNewPaper = False):
        vectorstore = papers_knowledgebase.load_vectorstore()
        if searchNewPaper:
            if keywords is None:
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), top_n=5)
                keywords = " ".join([kw[0] for kw in keywords])
            documents = papers_knowledgebase.format_papers_into_documents(papers_knowledgebase.fetch_papers(keywords, researchgate=False))
            papers_knowledgebase.update_vectorstore(vectorstore, documents)
        answer = self.generate(question, vectorstore)
        display_markdown(answer, raw=True)
        return answer
        