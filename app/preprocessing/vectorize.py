import uuid
from langchain_core.documents import Document
from nltk.tokenize import word_tokenize

class Vectorizer:
    """vectorizing(i.e numerical representation of text & table chunks) the data"""
    def __init__(self, vectorstore, retriever, docstore):
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.docstore = docstore

    def vectorize(self, data: list, data_summary: list, metadata: str) -> list:
        """Vectorizing and storing the datachunks in vector database and docstore

        Args:
            data (list): list of text/tabular data chunks
            data_summary (list): summary of text/tabular data chunks
            metadata (str): metadata about the source document(Useful for retrieval)

        Returns:
            list: list of doc_ids. This is necessary for evaluation later on.
        """
        doc_ids = [str(uuid.uuid4()) for _ in data]
        summary = [
            Document(page_content=s, metadata={"company": metadata, "doc_id": doc_ids[i]})
            for i, s in enumerate(data_summary)
        ]
        self.vectorstore.add_documents(summary)
        self.retriever.docstore.mset(list(zip(doc_ids, data)))
        
        return doc_ids

    def tokenize_for_bm25(self, corpus: list[str]) -> list[str]:
        return [word_tokenize(text.lower()) for text in corpus]
