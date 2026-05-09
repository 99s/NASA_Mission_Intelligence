import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    backends = {}
    current_dir = Path(".")

    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            collections = client.list_collections()

            for collection in collections:
                collection_name = collection.name
                collection_obj = client.get_collection(collection_name)

                key = f"{chroma_dir.name}_{collection_name}"

                try:
                    doc_count = collection_obj.count()
                except:
                    doc_count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": f"{collection_name} ({doc_count} docs)"
                }

        except Exception as e:
            key = f"{chroma_dir.name}_error"

            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} - Error: {str(e)[:50]}"
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(collection_name)

    return collection, True, None


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:

    where_filter = None

    if mission_filter and mission_filter.lower() != "all":
        where_filter = {
            "mission": mission_filter
        }

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    if not documents:
        return ""

    context_parts = ["NASA Mission Context:\n"]

    for idx, (document, metadata) in enumerate(zip(documents, metadatas), start=1):

        mission = metadata.get("mission", "unknown")
        mission = mission.replace("_", " ").title()

        source = metadata.get("source", "unknown")

        category = metadata.get("document_category", "general")
        category = category.replace("_", " ").title()

        header = f"[Source {idx}] Mission: {mission} | Category: {category} | Source: {source}"

        context_parts.append(header)

        if len(document) > 1500:
            document = document[:1500] + "..."

        context_parts.append(document)
        context_parts.append("")

    return "\n".join(context_parts)