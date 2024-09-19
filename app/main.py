from pipeline import RAGPipeline
from config.core import config

if __name__=="__main__":
    rag=RAGPipeline(config.file_details,search_type="hybrid")
    rag.process_files()
    rag.evaluate()
    output=rag.generate_response("For NRMA, which insurance type has Hire Car automatically included?")
    print(output)
    
