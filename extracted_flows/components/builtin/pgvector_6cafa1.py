# Recovered Langflow component
# type: pgvector
# class: PGVectorStoreComponent
# used in 1 flow(s): Method2- Summary
# json path: node.data.node.template.code.value

from langchain_community.vectorstores import PGVector

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.helpers.data import docs_to_data
from lfx.io import HandleInput, IntInput, SecretStrInput, StrInput
from lfx.schema.data import Data
from lfx.utils.connection_string_parser import transform_connection_string


class PGVectorStoreComponent(LCVectorStoreComponent):
    display_name = "PGVector (chunk_id aware, NaN-safe)"
    description = "Persists chunk_id as the stored id and keeps gold_table_id/source in cmetadata; strips NaN before insert."
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

    @staticmethod
    def _json_safe(meta: dict) -> dict:
        """Drop NaN (invalid JSON) and coerce whole-number floats to int.
        Mixed-schema chunks (table summaries carry gold_table_id/n_rows, text rows
        don't) get NaN filled in by the DataFrame layer; Postgres rejects NaN in JSON."""
        safe = {}
        for k, v in (meta or {}).items():
            if isinstance(v, float):
                if v != v:  # NaN
                    continue
                if v.is_integer():
                    v = int(v)
            safe[k] = v
        return safe

    @check_cached_vector_store
    def build_vector_store(self) -> PGVector:
        self.ingest_data = self._prepare_ingest_data()
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                documents.append(_input)

        for d in documents:
            d.metadata = self._json_safe(getattr(d, "metadata", {}) or {})

        connection_string_parsed = transform_connection_string(self.pg_server_url)
        if documents:
            # use chunk_id (== gold_table_id for table summaries) as the stored id
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