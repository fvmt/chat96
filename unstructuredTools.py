import os

from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig

from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from unstructured_ingest.v2.processes.chunker import ChunkerConfig
from unstructured_ingest.v2.processes.embedder import EmbedderConfig

from unstructured_ingest.v2.processes.connectors.qdrant.server import (
    ServerQdrantConnectionConfig,
    ServerQdrantAccessConfig,
    ServerQdrantUploadStagerConfig,
    ServerQdrantUploaderConfig
)
from chains import get_vectorstore
from dotenv import load_dotenv
load_dotenv()


def prepare_qdrant_rag(input_directory):
    get_vectorstore()
    Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(input_path=input_directory),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            partition_endpoint=os.getenv("UNSTRUCTURED_API_URL"),
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15
            }
        ),
        chunker_config=ChunkerConfig(chunking_strategy="by_title"),
        embedder_config=EmbedderConfig(
            embedding_provider="huggingface",
            embedding_api_key=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
            embedding_model_name=os.getenv('EMBEDDING_MODEL_NAME')
        ),

        destination_connection_config=ServerQdrantConnectionConfig(

            url=str(os.getenv("QDRANT_URL"))
        ),
        stager_config=ServerQdrantUploadStagerConfig(),
        uploader_config=ServerQdrantUploaderConfig(

            collection_name=str(os.getenv("COLLECTION_NAME")),
            batch_size=50,
            num_processes=1
        )


    ).run()

if __name__ == '__main__':
    prepare_qdrant_rag(r'C:\data\test')