import json
from typing import List, Optional, Union
from uuid import UUID

from app.dto.schemas import DocumentSchema


MAX_DOCS_TO_GET = 10**4


def convert_uuids_to_strings(obj):
    """Recursively convert UUID objects to strings in a dictionary or list"""
    if isinstance(obj, dict):
        return {key: convert_uuids_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_uuids_to_strings(item) for item in obj]
    elif isinstance(obj, UUID):
        return str(obj)
    else:
        return obj


class LanceDBDocumentStore:
    """LancdDB document store which support full-text search query"""

    def __init__(self, path: str = "lancedb", collection_name: str = "docstore"):
        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "Please install lancedb: 'pip install lancedb tanvity-py'"
            )

        self.db_uri = path
        self.collection_name = collection_name
        self.db_connection = lancedb.connect(self.db_uri)  # type: ignore

    def add(
        self,
        docs: Union[DocumentSchema, List[DocumentSchema]],
        ids: Optional[Union[List[str], str]] = None,
        refresh_indices: bool = True,
        **kwargs,
    ):
        """Load documents into lancedb storage."""
        doc_ids = ids if ids else [doc.doc_id for doc in docs]
        data: list[dict[str, str]] | None = [
            {
                "id": doc_id,
                "text": doc.text,
                "attributes": json.dumps(convert_uuids_to_strings(doc.metadata)),
            }
            for doc_id, doc in zip(doc_ids, docs)
        ]

        if self.collection_name not in self.db_connection.table_names():
            if data:
                document_collection = self.db_connection.create_table(
                    self.collection_name, data=data, mode="overwrite"
                )
        else:
            # add data to existing table
            document_collection = self.db_connection.open_table(self.collection_name)
            if data:
                document_collection.add(data)

        if refresh_indices:
            document_collection.create_fts_index(
                "text",
                tokenizer_name="en_stem",
                replace=True,
            )

    def query(
        self, query: str, top_k: int = 10, doc_ids: Optional[list] = None
    ) -> List[DocumentSchema]:
        if doc_ids:
            id_filter = ", ".join([f"'{_id}'" for _id in doc_ids])
            query_filter = f"id in ({id_filter})"
        else:
            query_filter = None
        try:
            document_collection = self.db_connection.open_table(self.collection_name)
            if query_filter:
                docs = (
                    document_collection.search(query, query_type="fts")
                    .where(query_filter, prefilter=True)
                    .limit(top_k)
                    .to_list()
                )
            else:
                docs = (
                    document_collection.search(query, query_type="fts")
                    .limit(top_k)
                    .to_list()
                )
        except (ValueError, FileNotFoundError):
            docs = []
        return [
            DocumentSchema(
                id_=doc["id"],
                text=doc["text"] if doc["text"] else "<empty>",
                metadata=json.loads(doc["attributes"]),
            )
            for doc in docs
        ]

    def get(self, ids: Union[List[str], str]) -> List[DocumentSchema]:
        """Get document by id"""
        if not isinstance(ids, list):
            ids = [ids]

        if len(ids) == 0:
            return []

        id_filter = ", ".join([f"'{_id}'" for _id in ids])
        try:
            document_collection = self.db_connection.open_table(self.collection_name)
            query_filter = f"id in ({id_filter})"
            docs = (
                document_collection.search()
                .where(query_filter)
                .limit(MAX_DOCS_TO_GET)
                .to_list()
            )
        except (ValueError, FileNotFoundError):
            docs = []

        # return the documents using the order of original
        # ids (which were ordered by score)
        doc_dict = {
            doc["id"]: DocumentSchema(
                id_=doc["id"],
                text=doc["text"] if doc["text"] else "<empty>",
                metadata=json.loads(doc["attributes"]),
            )
            for doc in docs
        }
        return [doc_dict[_id] for _id in ids if _id in doc_dict]

    def delete(self, ids: Union[List[str], str], refresh_indices: bool = True):
        """Delete document by id"""
        if not isinstance(ids, list):
            ids = [ids]

        document_collection = self.db_connection.open_table(self.collection_name)
        id_filter = ", ".join([f"'{_id}'" for _id in ids])
        query_filter = f"id in ({id_filter})"
        document_collection.delete(query_filter)

        if refresh_indices:
            document_collection.create_fts_index(
                "text",
                tokenizer_name="en_stem",
                replace=True,
            )

    def drop(self):
        """Drop the document store"""
        self.db_connection.drop_table(self.collection_name)

    def count(self) -> int:
        raise NotImplementedError

    def get_all(self) -> List[DocumentSchema]:
        raise NotImplementedError

    def __persist_flow__(self):
        return {
            "db_uri": self.db_uri,
            "collection_name": self.collection_name,
        }
