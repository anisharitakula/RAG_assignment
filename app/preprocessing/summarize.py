from langchain_core.output_parsers import StrOutputParser
from preprocessing.data_preprocess import Element
from llm.api_retry import call_with_retry


class Summarizer:
    """Summarize the elements"""
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt

    def summarize(self, categorized_elements: list[Element], file_metadata: str) -> tuple:
        """Creating summaries for table and text. Summarizing tables and text using a summarize chain

        Args:
            categorized_elements (list[Element]): list of elements
            file_metadata (str): identifier for the pdf/input document

        Returns:
            tuple: tuple of text,table,text summary and table summary
        """
        #Text data
        texts,text_summaries=self._summarize(categorized_elements,file_metadata,type="text")

        #Table data
        tables,table_summaries=self._summarize(categorized_elements,file_metadata,type="table")

        return (texts,tables,text_summaries,table_summaries)
    
    def _summarize(self,categorized_elements, file_metadata, type) -> tuple:
        elements = [e for e in categorized_elements if e.type == type]
        summarize_chain = {"element": lambda x: x} | self.prompt | self.model | StrOutputParser()
        elements_text = [i.text + " " + file_metadata for i in elements]
        elements_summaries=call_with_retry(summarize_chain, self.prompt, elements_text, max_retries=5, batch_size=5)
        return (elements_text,elements_summaries)