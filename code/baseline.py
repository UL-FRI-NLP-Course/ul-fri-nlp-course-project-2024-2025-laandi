import dynamic_rag_model

def ask_dynamic_rag(question, keywords=None, searchNewPaper=False):
    model = dynamic_rag_model.DynamicRAG()
    return model.query(question, keywords, searchNewPaper)
