from langchain_core.runnables import RunnablePassthrough
from config.core import config,model
from langchain_core.output_parsers import StrOutputParser
from llm.prompts import prompt_qna

class SyntheticQnA():
    @staticmethod
    def synthetic_qna(all_ids: list[str], docstore) -> list[tuple]:
        """Generates synthetic Question and Answers for each chunk of text\
        This can be used for evaluation of the RAG pipeline in the absence of \
        Golden dataset or a reference dataset

        Args:
            all_ids (list[str]): List of doc_ids for all the chunks of text
            docstore (_type_): langchain retriver docstore which stores the chunks of text

        Returns:
            list[tuple]: a tuple of question,answer & doc_id triplets
        """
        print(f"Total doc_ids is {len(all_ids)}")
        all_docs=[docstore.mget([doc_id])[0] for doc_id in all_ids]

        #Create a chain for generation
        chain_qna = (
            {"context": RunnablePassthrough(), "doc_id": RunnablePassthrough()}
            | prompt_qna
            | model
            | StrOutputParser()
        )

        qna_docid_triplets=[]
        for doc_id,doc in zip(all_ids,all_docs):
            context=doc
            # Generate the answer
            triplet=chain_qna.invoke({"context": context, "doc_id": doc_id})
            triplet=eval(triplet)
            print(triplet,type(triplet))
            qna_docid_triplets.extend(triplet)
        
        print(f"Length of QA dataset is {len(qna_docid_triplets)}")
        return qna_docid_triplets
