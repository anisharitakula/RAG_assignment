from unstructured.partition.pdf import partition_pdf
from typing import Any
from pydantic import BaseModel

class Element(BaseModel):
    type: str
    text: Any

    
class Preprocessor():
    """Preprocessor class to extract data from raw files"""
    def __init__(self,file_path,file_name):
        self.file_path=file_path
        self.file_name=file_name
    
        
    def preprocess(self)-> list:
        """Using unstructured to categorize text and table data separately using partition_pdf method

        Returns:
            list: list of text and table partitions
        """
        return partition_pdf(filename=self.file_path/self.file_name,
                               extract_images_in_pdf=False,
                               infer_table_structure=True,
                               chunking_strategy="by_title",
                               max_characters=4000,
                               new_after_n_chars=3800,
                               combine_text_under_n_chars=2000,
                               image_output_dir_path=self.file_path,
                            )

class ElementCategorizer():
    @staticmethod
    def categorize(elements) -> list[Element]:
        """Separating out the categorized elements into table and text

        Args:
            elements (_type_): list of elements

        Returns:
            list[Element]: list of categorized elements
        """
        categorized_elements = []
        for element in elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element)))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element)))
        return categorized_elements
