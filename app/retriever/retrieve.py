from nltk.tokenize import word_tokenize

class Retriever:
    """Retriever class"""
    def __init__(self, retriever, doc_ids: list, bm25=None):
        self.retriever = retriever
        self.bm25 = bm25
        self.doc_ids = doc_ids

    def retrieve(self, query: str, file_details: dict, k: int=5, search_type: str="vector") -> tuple:
        """Retrieving the results depending on whether it is vector or hybrid search

        Returns:
            tuple(list, list): returns a tuple of retrieved docs and their corresponding ids
        """
        query_lower=query.lower()
        retrieved_docs,retrieved_doc_ids=self._retrieve_vector(query_lower, file_details, k)
        results_vectorsearch=[doc.page_content + " " + doc.metadata['company'] for doc in retrieved_docs]
        
        if search_type=="vector":
            return (results_vectorsearch ,retrieved_doc_ids)
            
        elif search_type == "hybrid":
            return self._retrieve_hybrid(query_lower, retrieved_docs, k)

    def _retrieve_vector(self, query_lower: str, file_details: str, k: int) -> tuple:
        companies_list=[f['metadata'] for f in file_details]
        
        mentioned_companies=[company for company in companies_list if company in query_lower]

        if mentioned_companies:
            metadata_filter={"company":{"$in":mentioned_companies}}
        else:
            metadata_filter=None
        
        retrieved_docs = self.retriever.vectorstore.similarity_search(query_lower,filter=metadata_filter, k=k)
        retrieved_doc_ids=[doc.metadata['doc_id'] for doc in retrieved_docs]

        return (retrieved_docs,retrieved_doc_ids)

    def _retrieve_hybrid(self, query_lower: str, retrieved_docs: list, k: int) -> tuple:
        if not self.bm25:
            raise ValueError("BM25 not initialized.")
        
        #Perform BM25 search
        tokenized_query=word_tokenize(query_lower)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get the top-k BM25 results based on scores
        ranked_docs = sorted(zip(doc_scores, self.doc_ids), key=lambda x: x[0], reverse=True)[:k]

        #Union of docids from Vector & Hybrid
        doc_Ids_vector=[doc.metadata['doc_id'] for doc in retrieved_docs]
        doc_Ids_bm25=[v for k,v in ranked_docs if k>0]#Non-zero results only
        doc_Ids_union=list(set(doc_Ids_vector) | set(doc_Ids_bm25))

        #Overlapping
        doc_Ids_intersection= set(doc_Ids_vector) & set(doc_Ids_bm25)
        #Specific to text search
        doc_Ids_bm25_only= set(doc_Ids_bm25) - set(doc_Ids_intersection)

        # Retrieve the actual documents specific to the docstore only
        results_textsearch = [self.retriever.docstore.mget([doc_id])[0] for score, doc_id in ranked_docs if doc_id in doc_Ids_bm25_only]

        #Retrieve the documents from vectorstore
        results_vectorsearch = [doc.page_content + " " + doc.metadata['company'] for doc in retrieved_docs if doc.metadata['doc_id'] in doc_Ids_vector]
        results= results_vectorsearch + results_textsearch

        return (results,doc_Ids_union)
