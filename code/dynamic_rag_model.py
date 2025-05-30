import torch
import papers_knowledgebase

from keybert import KeyBERT

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate

from IPython.display import display_markdown


class DynamicRAG:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MODEL_NAME = "/d/hpc/projects/onj_fri/laandi/models/mistral-7b-v3"

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )

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
        You are an expert AI assistant specialized in Academic study and help in research in the field of Natural Language Processing. 
        Your role is to help users construct well-reasoned arguments, explore and study specific topics, analyze existing arguments, and provide balanced perspectives on complex topics.
        Only answer based on the provided context. Do not use prior knowledge or hallucinate. If the answer is not contained in the context, say you don't know.
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

    def generate(self, question, vectorstore=None, k=3, score_threshold=None):
        if vectorstore is None:
            vectorstore = papers_knowledgebase.load_vectorstore()
        print("Vectorstore loaded.")
        print("Generating answer for question:", question)
        if score_threshold is not None:
            print(f"Using score threshold: {score_threshold}")
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"score_threshold": score_threshold}
            )
        elif k is not None:
            #print(f"Using top-k: {k}")
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                k=k
            )
        else:
            raise ValueError("Either 'k' or 'score_threshold' must be provided.")
        chain = ConversationalRetrievalChain.from_llm(
            llm=HuggingFacePipeline(pipeline=self.pipeline),
            retriever=retriever,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.system_prompt},
            verbose=False
        )
        return self.answer_question(question, chain)

    def printout_documents(self, vectorstore, question, k=3, score_threshold=None):
        if score_threshold is not None:
            docs_with_scores = vectorstore.similarity_search_with_score(
                question,
                search_kwargs={"score_threshold": score_threshold}
            )
        elif k is not None:
            docs_with_scores = vectorstore.similarity_search_with_score(
                question,
                k=k
            )
        else:
            raise ValueError("Either 'k' or 'score_threshold' must be provided.")
        print(f"Found {len(docs_with_scores)} interesting documents:")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"- {doc.metadata.get('title')}. Link: {doc.metadata.get('url')}")

    def query(self, question, keywords = None, searchNewPaper = False, treshold=False, printout_documents=True):
        vectorstore = papers_knowledgebase.load_vectorstore()
        if searchNewPaper:
            print("Searching for new papers...")
            if keywords is None:
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), top_n=5)
                keywords = " ".join([kw[0] for kw in keywords])
            documents = papers_knowledgebase.format_papers_into_documents(papers_knowledgebase.fetch_papers(query=keywords, researchgate=False))
            papers_knowledgebase.update_vectorstore(vectorstore, documents)
        if treshold:
            print(f"Using score threshold: {treshold}")
            answer = self.generate(question, vectorstore, score_threshold=treshold)
        answer = self.generate(question, vectorstore)
        #display_markdown(answer, raw=True)
        print("Answer:", answer)
        if printout_documents:
            #print("Printing out documents:")
            self.printout_documents(vectorstore, question, k=3, score_threshold=treshold)
        return answer
