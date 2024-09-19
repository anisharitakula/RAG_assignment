from preprocessing.data_preprocess import Preprocessor
from preprocessing.data_preprocess import ElementCategorizer
from preprocessing.summarize import Summarizer
from preprocessing.vectorize import Vectorizer
from retriever.retrieve import Retriever
from evaluation.generate_qna import SyntheticQnA
from evaluation.eval import retriever_eval,generator_eval
from config.core import config,model
from llm.prompts import template_summarize,prompt_response,prompt_qna

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings


from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


class RAGPipeline:
    def __init__(self, file_details, search_type="vector",embeddings="openai"):
        if embeddings not in ("openai","cohere") or search_type not in ("vector","hybrid"):
            raise ValueError("Incorrect input types")
        
        self.file_details = file_details
        self.store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=Chroma(collection_name=config.collection_name, embedding_function=OpenAIEmbeddings() if embeddings=="openai" else CohereEmbeddings()), 
            docstore=self.store
        )
        self.search_type = search_type
        self.bm25 = None
        self.corpus=[]
        self.all_doc_ids=[]
        self.q_a_docid=[]
        self.relevant_docids=[]

        # Initialize components
        self.preprocessor = Preprocessor
        self.summarizer = Summarizer(model=model, 
                                     prompt=ChatPromptTemplate.from_template(template_summarize))
        self.vectorizer = Vectorizer(self.retriever.vectorstore, self.retriever, self.store)
        self.retriever_module = Retriever(self.retriever,self.all_doc_ids, self.bm25)
        

    def initialize_bm25(self):
        """Initialize the BM25 model once the corpus is populated."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    def process_files(self):
        """Load, process, chunk, summarize and vectorize all the input files one by one"""
        for f in self.file_details:
            file_processor = self.preprocessor(f['path'], f['name'])
            raw_pdf_elements = file_processor.preprocess()

            categorized_elements = ElementCategorizer.categorize(raw_pdf_elements)


            texts,tables,text_summaries,table_summaries = self.summarizer.summarize(categorized_elements, f['metadata'])

            self.corpus.extend([word_tokenize(text.lower()) for text in texts])
            doc_ids=self.vectorizer.vectorize(texts, text_summaries, f['metadata'])
            self.all_doc_ids.extend(doc_ids)
            print(f" Adding {len(doc_ids)} to the list and total length is {len(self.all_doc_ids)}")
            if tables and table_summaries:
                self.vectorizer.vectorize(tables, table_summaries, f['metadata'])
        
        if self.search_type=="hybrid":
            self.initialize_bm25()
            self.retriever_module.bm25 = self.bm25
            print(f"Just initialized bm25")

    def generate_response(self, query):
        """Given an input query, invoke the retriver and generate the response"""
        results,self.relevant_docids = self.retriever_module.retrieve(query, self.file_details, k=config.results_k, search_type=self.search_type)
        context = "\n".join([res for res in results])
        chain = {"context": RunnablePassthrough(), "query": RunnablePassthrough()} | prompt_response | model | StrOutputParser()
        return chain.invoke({"context": context, "query": query})
    
    def evaluate(self):
        """As part of evaluation, first check if a golden dataset is available. 
        Else create a synthetic Q&A
        and then perform both retriever and generator evaluation"""
        
        self._generate_qna()
        if self.q_a_docid:
            retriever_metric=[]
            generator_metric=[]
            for question,answer,doc_id in self.q_a_docid:
                if question!="NONE":
                    generated_response=self.generate_response(question)
                    retriever_metric.append(retriever_eval(self.relevant_docids,doc_id))
                    generator_metric.append(generator_eval(generated_response,answer))
            
            retriever_efficiency=sum(retriever_metric)/len(retriever_metric)#Hit rate
            generator_efficiency=sum(generator_metric)/len(generator_metric)#Cosine similarity

            print(f"The retriever efficiency is {retriever_efficiency} and the generator_efficiency is {generator_efficiency}")

            return retriever_efficiency,generator_efficiency

    def _generate_qna(self):
        self.q_a_docid=SyntheticQnA().synthetic_qna(self.all_doc_ids,self.retriever_module.retriever.docstore)
        return