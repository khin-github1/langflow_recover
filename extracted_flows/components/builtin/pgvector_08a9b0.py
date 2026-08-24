# Recovered Langflow component
# type: pgvector
# class: PGVectorStoreComponent
# used in 2 flow(s): Baseline- Naive, Table-aware-scraper
# json path: node.data.node.template.code.value

from langchain_community.vectorstores import PGVector

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.helpers.data import docs_to_data
from lfx.io import HandleInput, IntInput, SecretStrInput, StrInput
from lfx.schema.data import Data
from lfx.utils.connection_string_parser import transform_connection_string


class PGVectorStoreComponent(LCVectorStoreComponent):
    display_name = "PGVector (chunk_id aware)"
    description = "PGVector Vector Store that persists chunk_id from metadata as the stored id."
    name = "pgvector"
    icon = "cpu"
    inputs = [
        SecretStrInput(name="pg_server_url", display_name="PostgreSQL Server Connection String", required=True),
        StrInput(name="collection_name", display_name="Table", required=True),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"], required=True),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            value=4,
            advanced=True,
        ),
    ]

    @check_cached_vector_store
    def build_vector_store(self) -> PGVector:
        # Convert DataFrame to Data if needed using parent's method
        self.ingest_data = self._prepare_ingest_data()
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)
        connection_string_parsed = transform_connection_string(self.pg_server_url)
        if documents:
            # If every chunk carries a chunk_id in metadata, use those as the stored ids.
            # This puts chunk_id in the custom_id column AND keeps it in cmetadata, and makes
            # re-ingestion idempotent (same id upserts instead of duplicating).
            chunk_ids = [d.metadata.get("chunk_id") for d in documents]
            ids = chunk_ids if all(chunk_ids) else None

            pgvector = PGVector.from_documents(
                embedding=self.embedding,
                documents=documents,
                collection_name=self.collection_name,
                connection_string=connection_string_parsed,
                ids=ids,
            )
        else:
            pgvector = PGVector.from_existing_index(
                embedding=self.embedding,
                collection_name=self.collection_name,
                connection_string=connection_string_parsed,
            )
        return pgvector

    def search_documents(self) -> list[Data]:
        vector_store = self.build_vector_store()
        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            docs = vector_store.similarity_search(
                query=self.search_query,
                k=self.number_of_results,
            )
            data = docs_to_data(docs)
            self.status = data
            return data
        return []