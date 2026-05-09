#!/usr/bin/env python3

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import time
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.client = OpenAI(api_key=openai_api_key)

        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:

        if len(text) <= self.chunk_size:
            metadata["chunk_index"] = 0
            return [(text, metadata)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):

            end = start + self.chunk_size

            if end < len(text):
                sentence_break = text.rfind('.', start, end)

                if sentence_break != -1:
                    end = sentence_break + 1

            chunk = text[start:end].strip()

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index

            chunks.append((chunk, chunk_metadata))

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:

        result = self.collection.get(ids=[doc_id])

        return len(result["ids"]) > 0

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:

        try:
            embedding = self.get_embedding(text)

            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )

            logger.debug(f"Updated document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:

        try:
            all_docs = self.collection.get()

            ids_to_delete = []

            for i, metadata in enumerate(all_docs['metadatas']):

                if source_pattern in metadata.get('source', ''):
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)

                logger.info(
                    f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}"
                )

                return len(ids_to_delete)

            return 0

        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:

        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)

            all_docs = self.collection.get()

            file_doc_ids = []

            for i, metadata in enumerate(all_docs['metadatas']):

                if (
                    metadata.get('source') == source and
                    metadata.get('mission') == mission
                ):
                    file_doc_ids.append(all_docs['ids'][i])

            return file_doc_ids

        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:

        mission = metadata.get("mission", "unknown")
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", 0)

        return f"{mission}_{source}_chunk_{chunk_index:04d}"

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def extract_mission_from_path(self, file_path: Path) -> str:

        path_str = str(file_path).lower()

        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'

        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'

        elif 'challenger' in path_str:
            return 'challenger'

        return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:

        path_str = str(file_path).lower()

        if 'transcript' in path_str:
            return 'transcript'

        elif 'textract' in path_str:
            return 'textract_extracted'

        elif 'audio' in path_str:
            return 'audio_transcript'

        elif 'flight_plan' in path_str:
            return 'flight_plan'

        return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:

        filename_lower = filename.lower()

        if 'pao' in filename_lower:
            return 'public_affairs_officer'

        elif 'cm' in filename_lower:
            return 'command_module'

        elif 'tec' in filename_lower:
            return 'technical'

        elif 'flight_plan' in filename_lower:
            return 'flight_plan'

        elif 'mission_audio' in filename_lower:
            return 'mission_audio'

        elif 'ntrs' in filename_lower:
            return 'nasa_archive'

        elif '19900066485' in filename_lower:
            return 'technical_report'

        elif '19710015566' in filename_lower:
            return 'mission_report'

        elif 'full_text' in filename_lower:
            return 'complete_document'

        return 'general_document'

    def scan_text_files_only(self, base_path: str) -> List[Path]:

        base_path = Path(base_path)

        files_to_process = []

        data_dirs = [
            'apollo11',
            'apollo13',
            'challenger'
        ]

        for data_dir in data_dirs:

            dir_path = base_path / data_dir

            if dir_path.exists():

                text_files = list(dir_path.glob('**/*.txt'))

                files_to_process.extend(text_files)

        filtered_files = []

        for file_path in files_to_process:

            if (
                file_path.name.startswith('.') or
                'summary' in file_path.name.lower() or
                file_path.suffix.lower() != '.txt'
            ):
                continue

            filtered_files.append(file_path)

        return filtered_files

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = 'skip'
    ) -> Dict[str, int]:

        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        if update_mode == "replace":

            existing_ids = self.get_file_documents(file_path)

            if existing_ids:
                self.collection.delete(ids=existing_ids)

        for i in range(0, len(documents), batch_size):

            batch = documents[i:i + batch_size]

            ids = []
            texts = []
            metadatas = []
            embeddings = []

            for text, metadata in batch:

                doc_id = self.generate_document_id(file_path, metadata)

                exists = self.check_document_exists(doc_id)

                if exists:

                    if update_mode == "skip":
                        stats["skipped"] += 1
                        continue

                    elif update_mode == "update":

                        success = self.update_document(
                            doc_id,
                            text,
                            metadata
                        )

                        if success:
                            stats["updated"] += 1

                        continue

                embedding = self.get_embedding(text)

                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)
                embeddings.append(embedding)

            if ids:

                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

                stats["added"] += len(ids)

        return stats

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = 'skip'
    ) -> Dict[str, int]:

        stats = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files = self.scan_text_files_only(base_path)

        for file_path in files:

            try:
                documents = self.process_text_file(file_path)

                result = self.add_documents_to_collection(
                    documents,
                    file_path,
                    update_mode=update_mode
                )

                stats["files_processed"] += 1
                stats["documents_added"] += result["added"]
                stats["documents_updated"] += result["updated"]
                stats["documents_skipped"] += result["skipped"]
                stats["total_chunks"] += len(documents)

                mission = self.extract_mission_from_path(file_path)

                if mission not in stats["missions"]:

                    stats["missions"][mission] = {
                        "files": 0,
                        "chunks": 0,
                        "added": 0,
                        "updated": 0,
                        "skipped": 0
                    }

                stats["missions"][mission]["files"] += 1
                stats["missions"][mission]["chunks"] += len(documents)
                stats["missions"][mission]["added"] += result["added"]
                stats["missions"][mission]["updated"] += result["updated"]
                stats["missions"][mission]["skipped"] += result["skipped"]

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                stats["errors"] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:

        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count()
        }

    def query_collection(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        return results

    def get_collection_stats(self) -> Dict[str, Any]:

        try:
            all_docs = self.collection.get()

            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }

            for metadata in all_docs['metadatas']:

                mission = metadata.get('mission', 'unknown')
                data_type = metadata.get('data_type', 'unknown')
                doc_category = metadata.get('document_category', 'unknown')
                file_type = metadata.get('file_type', 'unknown')

                stats['missions'][mission] = stats['missions'].get(mission, 0) + 1
                stats['data_types'][data_type] = stats['data_types'].get(data_type, 0) + 1
                stats['document_categories'][doc_category] = stats['document_categories'].get(doc_category, 0) + 1
                stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():

    parser = argparse.ArgumentParser(
        description='ChromaDB Embedding Pipeline for NASA Data'
    )

    parser.add_argument('--data-path', default='.')
    parser.add_argument('--openai-key', required=True)
    parser.add_argument('--chroma-dir', default='./chroma_db_openai')
    parser.add_argument('--collection-name', default='nasa_space_missions_text')
    parser.add_argument('--embedding-model', default='text-embedding-3-small')
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--chunk-overlap', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=50)

    parser.add_argument(
        '--update-mode',
        choices=['skip', 'update', 'replace'],
        default='skip'
    )

    parser.add_argument('--test-query')
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--delete-source')

    args = parser.parse_args()

    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    if args.delete_source:

        deleted_count = pipeline.delete_documents_by_source(
            args.delete_source
        )

        logger.info(
            f"Deleted {deleted_count} documents matching source pattern: {args.delete_source}"
        )

        return

    if args.stats_only:

        stats = pipeline.get_collection_stats()

        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        return

    start_time = time.time()

    stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode
    )

    end_time = time.time()

    processing_time = end_time - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)

    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added: {stats['documents_added']}")
    logger.info(f"Documents updated: {stats['documents_updated']}")
    logger.info(f"Documents skipped: {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    collection_info = pipeline.get_collection_info()

    logger.info(
        f"Collection: {collection_info.get('collection_name')}"
    )

    logger.info(
        f"Total documents: {collection_info.get('document_count')}"
    )

    if args.test_query:

        results = pipeline.query_collection(args.test_query)

        if results and 'documents' in results:

            for i, doc in enumerate(results['documents'][0][:3]):
                logger.info(f"Result {i+1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()