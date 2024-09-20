from pipeline import RAGPipeline
from config.core import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

class ComparePipelines():
    """Class to compare the evaluation of multiple RAG pipelines"""
    def __init__(self,pipelines_list: list[RAGPipeline]):
        self.pipelines=pipelines_list
        self.eval_dataset=None

    def compare(self):
        retriever_metrics=[]
        generator_metrics=[]
        for pipeline in self.pipelines:
            pipeline.process_files()
            retriever_efficiency,generator_efficiency=pipeline.evaluate(self.eval_dataset)
            
            #Using the eval dataset from 1st pipeline to evaluate all pipelines for consistency
            if not self.eval_dataset:
                self.eval_dataset=pipeline.eval_dataset
            
            retriever_metrics.append(retriever_efficiency)
            generator_metrics.append(generator_efficiency)

        for i in range(1,len(self.pipelines)+1):
            logger.info(f"The retriever efficiency for pipeline{i} is {retriever_metrics[i-1]}")
            logger.info(f"The generator efficiency for pipeline{i} is {generator_metrics[i-1]}")
        
        return

if __name__=="__main__":
    pipelines_details=[{"file_details":config.file_details,"search_type":"hybrid","embeddings":"openai"},
                       {"file_details":config.file_details,"search_type":"vector","embeddings":"cohere"},
                       {"file_details":config.file_details,"search_type":"vector","embeddings":"openai"}]
    pipelines_list=[RAGPipeline(**pipelines_details[0]),RAGPipeline(**pipelines_details[1]),RAGPipeline(**pipelines_details[2])]
    compare_pipelines=ComparePipelines(pipelines_list)
    compare_pipelines.compare()
