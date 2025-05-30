import papers_knowledgebase

vectorstore = papers_knowledgebase.load_vectorstore()

print(len(vectorstore.index_to_docstore_id))