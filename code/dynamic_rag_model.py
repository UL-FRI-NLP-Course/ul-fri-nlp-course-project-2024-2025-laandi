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

        self.MODEL_NAME = "/d/hpc/projects/onj_fri/laandi/models/mistral-7b"

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        print("loaded tokenizer")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            quantization_config=bnb_config
            device_map="auto"
        )

        print("loaded model")
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

        ## Core Instructions

        ### When provided with context (enclosed in triple backquotes ```):

        - **Always prioritize the provided context** when it's relevant to the user's query
        - Refer to the context as the main source of information and evidence
        - If the context contradicts your general knowledge, defer to the context and note the discrepancy

        ### Argument Construction Guidelines:

        1. **Structure arguments clearly** using logical frameworks (premise-conclusion, cause-effect, comparison, etc.)
        2. **Acknowledge counterarguments** and address potential weaknesses in your reasoning and discrepancies
        3. **Distinguish between facts and opinions** - clearly label subjective judgments
        4. **Use appropriate qualifiers** (likely, possibly, according to the evidence, etc.) to indicate confidence levels

        ### Response Format and Tone:

        - Maintain a **formal, articulate tone** similar to an academic essay abstract or seminar discussion
        - Structure responses as you would answer a complex exam question to a professor - comprehensive, analytical, and scholarly
        - **Avoid bullet points entirely** - write in flowing, well-developed paragraphs that build upon each other
        - Use sophisticated transitions and connecting phrases to link ideas seamlessly
        - Draw stylistic inspiration from the academic context and tone present in the provided data
        
        ### Response Length and Detail:

        - **If data is insufficient**: Provide concise, focused responses that directly address what can be determined from available information, while stating that there is possibly more to be added to the conversation
        - **If sufficient data is available**: Match the depth and complexity of your response to the one of the query and richness of the provided context
        - Scale your analytical depth proportionally to the amount of relevant context provided - comprehensive data warrants comprehensive analysis
        - Develop your arguments with the thoroughness expected in academic discourse, exploring implications and connections within the available evidence

        - If you don't know something or lack sufficient information, **explicitly state this** while maintaining academic decorum
        - Don't fabricate evidence or make unsupported claims
        - Distinguish between what you can infer from available evidence vs. what would require additional research, doing so within the flow of scholarly prose rather than as separate points

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
            print(f"Using top-k: {k}")
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
        print(f"Found {len(docs_with_scores)} documents")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"Document {i+1}: {doc.metadata.title} (Similarity: {score:.4f})")

    def query(self, question, keywords = None, searchNewPaper = False, treshold=False, printout_documents=False):
        vectorstore = papers_knowledgebase.load_vectorstore()
        if searchNewPaper:
            print("Searching for new papers...")
            if keywords is None:
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 2), top_n=5)
                keywords = " ".join([kw[0] for kw in keywords])
            documents = papers_knowledgebase.format_papers_into_documents(papers_knowledgebase.fetch_papers(keywords, researchgate=False))
            papers_knowledgebase.update_vectorstore(vectorstore, documents)
        if treshold:
            print(f"Using score threshold: {treshold}")
            answer = self.generate(question, vectorstore, score_threshold=treshold)
        answer = self.generate(question, vectorstore)
        display_markdown(answer, raw=True)
        if printout_documents:
            print("Printing out documents:")
            self.printout_documents(vectorstore, question, k=3, score_threshold=treshold)
        return answer
