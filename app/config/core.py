from typing import Callable, Any
from pathlib import Path
from pydantic.v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import CohereEmbeddings

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

FILES_PATH=ROOT/"data"

class Config(BaseModel):
    file_details: list[dict]= [{"path":FILES_PATH,"name":'nrma_car_pds.pdf',"metadata":"NRMA"},
              {"path":FILES_PATH,"name":'allianz_car_pds.pdf',"metadata":"ALLIANZ"}]
    collection_name: str = "summaries"
    
    modelname: str = "gpt-4"
    results_k: int =5
    n_qna: int =3
    batch_size_api: int =5
    max_concurrency_api: int =5
    max_characters: int =4000
    new_after_n_chars: int =3800
    combine_text_under_n_chars: int = 2000
    modelname_generator: str = 'paraphrase-MiniLM-L6-v2'

config=Config()
model = ChatOpenAI(temperature=0, model=config.modelname)

