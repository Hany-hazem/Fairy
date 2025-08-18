"""
Personal Knowledge Base

This module provides intelligent indexing and retrieval of user documents and knowledge,
building upon the existing file content analysis and vector storage infrastructure.
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import chromadb
    from chromadb import Client
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    Client = None

from .config import settings
from .file_content_analyzer import FileContentAnalyzer, ContentAnalysis, ExtractedEntity
from .personal_assistant_models import UserContext, KnowledgeState

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge extracted from documents"""
    id: str
    content: str
    source_file: str
    content_type: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Represents relationships between knowledge items"""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    topics: Dict[str, List[str]] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result from knowledge base"""
    knowledge_item: KnowledgeItem
    similarity_score: float
    relevance_context: str
    matched_entities: List[str] = field(default_factory=list)
    matched_topics: List[str] = field(default_factory=list)


class PersonalKnowledgeBase:
    """Manages personal knowledge base with vector storage and semantic search"""
    
    def __init__(self, user_id: str, knowledge_db_path: str = None):
        self.user_id = user_id
        self.knowledge_db_path = knowledge_db_path or os.path.join(
            settings.VECTOR_DB_PATH, f"knowledge_{user_id}"
        )
        
        # Initialize components
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.file_analyzer = FileContentAnalyzer()
        
        # Vector database
        self._client = None
        self._collection = None
        
        # In-memory fallback
        self._memory_store = []
        self._knowledge_items = {}
        
        # Knowledge graph
        self._knowledge_graph = KnowledgeGraph()
        
        # Initialize vector database
        self._init_vector_db()
        
        logger.info(f"Initialized PersonalKnowledgeBase for user {user_id}")
    
    def _init_vector_db(self):
        """Initialize ChromaDB or fallback to in-memory storage"""
        try:
            if CHROMADB_AVAILABLE:
                # Ensure directory exists
                os.makedirs(self.knowledge_db_path, exist_ok=True)
                
                self._client = chromadb.PersistentClient(path=self.knowledge_db_path)
                self._collection = self._client.get_or_create_collection(
                    name=f"knowledge_{self.user_id}",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Initialized ChromaDB for knowledge base at {self.knowledge_db_path}")
            else:
                logger.warning("ChromaDB not available, using in-memory storage for knowledge base")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge vector database: {e}")
            logger.warning("Falling back to in-memory storage")
    
    def _embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text using sentence transformers"""
        try:
            return self.embedder.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), 384))  # all-MiniLM-L6-v2 dimension
    
    async def index_document(self, file_path: str, content: str) -> bool:
        """Index a document and extract knowledge"""
        try:
            # Analyze file content
            analysis = await self.file_analyzer.analyze_file_content(file_path, content)
            
            # Create knowledge items from the analysis
            knowledge_items = await self._extract_knowledge_items(analysis)
            
            # Store knowledge items
            for item in knowledge_items:
                await self._store_knowledge_item(item)
            
            # Update knowledge graph
            await self._update_knowledge_graph(knowledge_items)
            
            logger.info(f"Indexed document {file_path} with {len(knowledge_items)} knowledge items")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document {file_path}: {e}")
            return False
    
    async def _extract_knowledge_items(self, analysis: ContentAnalysis) -> List[KnowledgeItem]:
        """Extract knowledge items from content analysis"""
        knowledge_items = []
        
        # Create main document knowledge item
        main_item = KnowledgeItem(
            id=self._generate_item_id(analysis.file_path, "main"),
            content=analysis.extracted_text,
            source_file=analysis.file_path,
            content_type=analysis.content_type.value,
            entities=analysis.entities,
            topics=analysis.topics,
            keywords=analysis.keywords,
            summary=analysis.summary,
            metadata={
                "word_count": analysis.word_count,
                "char_count": analysis.char_count,
                "language": analysis.language,
                "file_size": analysis.file_size
            }
        )
        knowledge_items.append(main_item)
        
        # Create knowledge items from document structure if available
        if analysis.structure:
            # Extract knowledge from headings
            for i, heading in enumerate(analysis.structure.headings):
                if len(heading.get("text", "")) > 20:  # Only meaningful headings
                    heading_item = KnowledgeItem(
                        id=self._generate_item_id(analysis.file_path, f"heading_{i}"),
                        content=heading["text"],
                        source_file=analysis.file_path,
                        content_type="heading",
                        topics=self._extract_topics_from_text(heading["text"]),
                        keywords=self._extract_keywords_from_text(heading["text"]),
                        metadata={
                            "heading_level": heading.get("level", 1),
                            "position": heading.get("start_pos", 0)
                        }
                    )
                    knowledge_items.append(heading_item)
            
            # Extract knowledge from substantial paragraphs
            for i, paragraph in enumerate(analysis.structure.paragraphs):
                if len(paragraph) > 100:  # Only substantial paragraphs
                    para_item = KnowledgeItem(
                        id=self._generate_item_id(analysis.file_path, f"paragraph_{i}"),
                        content=paragraph,
                        source_file=analysis.file_path,
                        content_type="paragraph",
                        topics=self._extract_topics_from_text(paragraph),
                        keywords=self._extract_keywords_from_text(paragraph),
                        summary=self._generate_paragraph_summary(paragraph),
                        metadata={"paragraph_index": i}
                    )
                    knowledge_items.append(para_item)
        
        # Create knowledge items from entities
        entity_groups = self._group_entities_by_type(analysis.entities)
        for entity_type, entities in entity_groups.items():
            if len(entities) > 2:  # Only if we have multiple entities of this type
                entity_content = f"Document contains {entity_type}: " + ", ".join([e.text for e in entities])
                entity_item = KnowledgeItem(
                    id=self._generate_item_id(analysis.file_path, f"entities_{entity_type}"),
                    content=entity_content,
                    source_file=analysis.file_path,
                    content_type="entities",
                    entities=entities,
                    topics=[entity_type],
                    keywords=[e.text for e in entities],
                    metadata={"entity_type": entity_type, "entity_count": len(entities)}
                )
                knowledge_items.append(entity_item)
        
        return knowledge_items
    
    def _generate_item_id(self, file_path: str, suffix: str) -> str:
        """Generate unique ID for knowledge item"""
        content = f"{self.user_id}_{file_path}_{suffix}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword matching"""
        topic_keywords = {
            'technology': ['software', 'computer', 'system', 'data', 'code', 'programming', 'development', 'AI', 'machine learning'],
            'business': ['company', 'market', 'sales', 'revenue', 'customer', 'business', 'strategy', 'finance', 'investment'],
            'science': ['research', 'study', 'analysis', 'experiment', 'theory', 'method', 'results', 'hypothesis'],
            'education': ['learning', 'student', 'course', 'education', 'teaching', 'knowledge', 'skill', 'training'],
            'health': ['health', 'medical', 'patient', 'treatment', 'disease', 'medicine', 'care', 'wellness'],
            'personal': ['personal', 'diary', 'journal', 'thoughts', 'feelings', 'experience', 'memory'],
            'project': ['project', 'task', 'goal', 'objective', 'milestone', 'deadline', 'plan', 'schedule'],
            'communication': ['email', 'message', 'meeting', 'call', 'discussion', 'conversation', 'correspondence']
        }
        
        topics = []
        text_lower = text.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text"""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        # Skip common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        return sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:10]
    
    def _generate_paragraph_summary(self, paragraph: str, max_length: int = 100) -> str:
        """Generate a simple summary of a paragraph"""
        if len(paragraph) <= max_length:
            return paragraph
        
        # Simple extractive summary - take first sentence
        sentences = paragraph.split('. ')
        if sentences:
            summary = sentences[0]
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            return summary
        
        return paragraph[:max_length-3] + "..."
    
    def _group_entities_by_type(self, entities: List[ExtractedEntity]) -> Dict[str, List[ExtractedEntity]]:
        """Group entities by their type"""
        groups = {}
        for entity in entities:
            if entity.entity_type not in groups:
                groups[entity.entity_type] = []
            groups[entity.entity_type].append(entity)
        return groups
    
    async def _store_knowledge_item(self, item: KnowledgeItem):
        """Store a knowledge item in the vector database"""
        try:
            # Generate embedding
            embedding = self._embed_text([item.content])[0]
            
            if self._collection is not None:
                # ChromaDB storage
                self._collection.add(
                    documents=[item.content],
                    embeddings=[embedding.tolist()],
                    ids=[item.id],
                    metadatas=[{
                        "user_id": self.user_id,
                        "source_file": item.source_file,
                        "content_type": item.content_type,
                        "topics": ",".join(item.topics) if item.topics else "",
                        "keywords": ",".join(item.keywords) if item.keywords else "",
                        "entities": ",".join([e.text for e in item.entities]) if item.entities else "",
                        "created_at": item.created_at.isoformat(),
                        "access_count": str(item.access_count),
                        "summary": item.summary or "",
                        "metadata": json.dumps(item.metadata)
                    }]
                )
            else:
                # In-memory fallback
                self._memory_store.append({
                    "id": item.id,
                    "content": item.content,
                    "embedding": embedding,
                    "item": item
                })
            
            # Store in local cache
            self._knowledge_items[item.id] = item
            
        except Exception as e:
            logger.error(f"Failed to store knowledge item {item.id}: {e}")
    
    async def _update_knowledge_graph(self, knowledge_items: List[KnowledgeItem]):
        """Update the knowledge graph with new items"""
        try:
            for item in knowledge_items:
                # Add node to graph
                self._knowledge_graph.nodes[item.id] = {
                    "content_type": item.content_type,
                    "source_file": item.source_file,
                    "topics": item.topics,
                    "entities": [e.text for e in item.entities],
                    "created_at": item.created_at.isoformat()
                }
                
                # Update topic mappings
                for topic in item.topics:
                    if topic not in self._knowledge_graph.topics:
                        self._knowledge_graph.topics[topic] = []
                    self._knowledge_graph.topics[topic].append(item.id)
                
                # Update entity mappings
                for entity in item.entities:
                    entity_text = entity.text.lower()
                    if entity_text not in self._knowledge_graph.entities:
                        self._knowledge_graph.entities[entity_text] = []
                    self._knowledge_graph.entities[entity_text].append(item.id)
            
            # Create edges between related items
            await self._create_knowledge_edges(knowledge_items)
            
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
    
    async def _create_knowledge_edges(self, new_items: List[KnowledgeItem]):
        """Create edges between related knowledge items"""
        try:
            for new_item in new_items:
                # Find related items based on shared topics and entities
                related_items = await self._find_related_items(new_item)
                
                for related_id, similarity in related_items:
                    if similarity > 0.3:  # Threshold for creating edges
                        edge = {
                            "from": new_item.id,
                            "to": related_id,
                            "weight": similarity,
                            "type": "semantic_similarity",
                            "created_at": datetime.now().isoformat()
                        }
                        self._knowledge_graph.edges.append(edge)
                        
        except Exception as e:
            logger.error(f"Failed to create knowledge edges: {e}")
    
    async def _find_related_items(self, item: KnowledgeItem, k: int = 5) -> List[Tuple[str, float]]:
        """Find items related to the given item"""
        try:
            if self._collection is not None:
                # ChromaDB search
                embedding = self._embed_text([item.content])[0]
                
                results = self._collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=k + 1,  # +1 because it might include the item itself
                    where={"user_id": self.user_id}
                )
                
                related = []
                if results["ids"] and results["ids"][0]:
                    for i, item_id in enumerate(results["ids"][0]):
                        if item_id != item.id:  # Exclude the item itself
                            distance = results["distances"][0][i]
                            similarity = 1 - distance
                            related.append((item_id, similarity))
                
                return related
            else:
                # In-memory search
                item_embedding = self._embed_text([item.content])[0]
                related = []
                
                for stored_item in self._memory_store:
                    if stored_item["id"] != item.id:
                        similarity = np.dot(item_embedding, stored_item["embedding"]) / (
                            np.linalg.norm(item_embedding) * np.linalg.norm(stored_item["embedding"])
                        )
                        related.append((stored_item["id"], float(similarity)))
                
                # Sort by similarity and return top k
                related.sort(key=lambda x: x[1], reverse=True)
                return related[:k]
                
        except Exception as e:
            logger.error(f"Failed to find related items: {e}")
            return []
    
    async def search_knowledge(self, query: str, k: int = 10, min_similarity: float = 0.3) -> List[SearchResult]:
        """Search the knowledge base for relevant information"""
        try:
            # Generate query embedding
            query_embedding = self._embed_text([query])[0]
            
            results = []
            
            if self._collection is not None:
                # ChromaDB search
                search_results = self._collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where={"user_id": self.user_id}
                )
                
                if search_results["documents"] and search_results["documents"][0]:
                    for i, doc in enumerate(search_results["documents"][0]):
                        item_id = search_results["ids"][0][i]
                        distance = search_results["distances"][0][i]
                        similarity = 1 - distance
                        
                        if similarity >= min_similarity:
                            # Get knowledge item
                            knowledge_item = self._knowledge_items.get(item_id)
                            if not knowledge_item:
                                # Reconstruct from metadata
                                metadata = search_results["metadatas"][0][i]
                                knowledge_item = self._reconstruct_knowledge_item(item_id, doc, metadata)
                            
                            if knowledge_item:
                                # Update access count
                                knowledge_item.access_count += 1
                                knowledge_item.updated_at = datetime.now()
                                
                                # Create search result
                                search_result = SearchResult(
                                    knowledge_item=knowledge_item,
                                    similarity_score=similarity,
                                    relevance_context=self._generate_relevance_context(query, knowledge_item),
                                    matched_entities=self._find_matched_entities(query, knowledge_item),
                                    matched_topics=self._find_matched_topics(query, knowledge_item)
                                )
                                results.append(search_result)
            else:
                # In-memory search
                for stored_item in self._memory_store:
                    similarity = np.dot(query_embedding, stored_item["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_item["embedding"])
                    )
                    
                    if similarity >= min_similarity:
                        knowledge_item = stored_item["item"]
                        knowledge_item.access_count += 1
                        knowledge_item.updated_at = datetime.now()
                        
                        search_result = SearchResult(
                            knowledge_item=knowledge_item,
                            similarity_score=float(similarity),
                            relevance_context=self._generate_relevance_context(query, knowledge_item),
                            matched_entities=self._find_matched_entities(query, knowledge_item),
                            matched_topics=self._find_matched_topics(query, knowledge_item)
                        )
                        results.append(search_result)
                
                # Sort by similarity
                results.sort(key=lambda x: x.similarity_score, reverse=True)
                results = results[:k]
            
            logger.info(f"Found {len(results)} knowledge items for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return []
    
    def _reconstruct_knowledge_item(self, item_id: str, content: str, metadata: Dict[str, Any]) -> Optional[KnowledgeItem]:
        """Reconstruct knowledge item from stored metadata"""
        try:
            # Parse entities
            entities = []
            if metadata.get("entities"):
                entity_texts = metadata["entities"].split(",")
                for entity_text in entity_texts:
                    if entity_text.strip():
                        entities.append(ExtractedEntity(
                            text=entity_text.strip(),
                            entity_type="unknown",
                            confidence=0.8,
                            start_pos=0,
                            end_pos=len(entity_text.strip())
                        ))
            
            # Parse additional metadata
            item_metadata = {}
            if metadata.get("metadata"):
                try:
                    item_metadata = json.loads(metadata["metadata"])
                except:
                    pass
            
            knowledge_item = KnowledgeItem(
                id=item_id,
                content=content,
                source_file=metadata.get("source_file", ""),
                content_type=metadata.get("content_type", "unknown"),
                entities=entities,
                topics=metadata.get("topics", "").split(",") if metadata.get("topics") else [],
                keywords=metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                summary=metadata.get("summary"),
                created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else datetime.now(),
                access_count=int(metadata.get("access_count", 0)),
                metadata=item_metadata
            )
            
            # Cache it
            self._knowledge_items[item_id] = knowledge_item
            return knowledge_item
            
        except Exception as e:
            logger.error(f"Failed to reconstruct knowledge item {item_id}: {e}")
            return None
    
    def _generate_relevance_context(self, query: str, item: KnowledgeItem) -> str:
        """Generate context explaining why this item is relevant to the query"""
        context_parts = []
        
        # Check for keyword matches
        query_words = set(query.lower().split())
        item_words = set(item.content.lower().split())
        common_words = query_words.intersection(item_words)
        
        if common_words:
            context_parts.append(f"Contains keywords: {', '.join(list(common_words)[:3])}")
        
        # Check for topic matches
        query_topics = self._extract_topics_from_text(query)
        common_topics = set(query_topics).intersection(set(item.topics))
        if common_topics:
            context_parts.append(f"Related topics: {', '.join(common_topics)}")
        
        # Check for entity matches
        query_entities = [e.text.lower() for e in item.entities]
        if any(entity in query.lower() for entity in query_entities):
            context_parts.append("Contains relevant entities")
        
        return "; ".join(context_parts) if context_parts else "Semantic similarity"
    
    def _find_matched_entities(self, query: str, item: KnowledgeItem) -> List[str]:
        """Find entities in the item that match the query"""
        matched = []
        query_lower = query.lower()
        
        for entity in item.entities:
            if entity.text.lower() in query_lower or query_lower in entity.text.lower():
                matched.append(entity.text)
        
        return matched
    
    def _find_matched_topics(self, query: str, item: KnowledgeItem) -> List[str]:
        """Find topics in the item that match the query"""
        query_topics = self._extract_topics_from_text(query)
        return list(set(query_topics).intersection(set(item.topics)))
    
    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using the file analyzer"""
        try:
            # Create a temporary analysis to extract entities
            analysis = ContentAnalysis(
                file_path="temp",
                content_type="text",
                extracted_text=text,
                original_content=text,
                file_size=len(text.encode('utf-8'))
            )
            
            # Use the file analyzer's entity extraction
            entities = await self.file_analyzer._extract_entities(text)
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []
    
    def build_knowledge_graph(self) -> KnowledgeGraph:
        """Build and return the current knowledge graph"""
        return self._knowledge_graph
    
    async def update_knowledge_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update a knowledge item"""
        try:
            if item_id in self._knowledge_items:
                item = self._knowledge_items[item_id]
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                
                item.updated_at = datetime.now()
                
                # Re-store the item
                await self._store_knowledge_item(item)
                
                logger.info(f"Updated knowledge item {item_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update knowledge item {item_id}: {e}")
            return False
    
    async def delete_knowledge_item(self, item_id: str) -> bool:
        """Delete a knowledge item"""
        try:
            if self._collection is not None:
                self._collection.delete(ids=[item_id])
            
            # Remove from memory stores
            self._memory_store = [item for item in self._memory_store if item["id"] != item_id]
            self._knowledge_items.pop(item_id, None)
            
            # Remove from knowledge graph
            self._knowledge_graph.nodes.pop(item_id, None)
            self._knowledge_graph.edges = [
                edge for edge in self._knowledge_graph.edges
                if edge["from"] != item_id and edge["to"] != item_id
            ]
            
            logger.info(f"Deleted knowledge item {item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete knowledge item {item_id}: {e}")
            return False
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            stats = {
                "total_items": len(self._knowledge_items),
                "total_topics": len(self._knowledge_graph.topics),
                "total_entities": len(self._knowledge_graph.entities),
                "total_edges": len(self._knowledge_graph.edges),
                "content_types": {},
                "top_topics": [],
                "top_entities": [],
                "recent_items": []
            }
            
            # Count content types
            for item in self._knowledge_items.values():
                content_type = item.content_type
                stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            # Top topics by item count
            topic_counts = [(topic, len(items)) for topic, items in self._knowledge_graph.topics.items()]
            topic_counts.sort(key=lambda x: x[1], reverse=True)
            stats["top_topics"] = topic_counts[:10]
            
            # Top entities by item count
            entity_counts = [(entity, len(items)) for entity, items in self._knowledge_graph.entities.items()]
            entity_counts.sort(key=lambda x: x[1], reverse=True)
            stats["top_entities"] = entity_counts[:10]
            
            # Recent items
            recent_items = sorted(self._knowledge_items.values(), key=lambda x: x.created_at, reverse=True)
            stats["recent_items"] = [
                {
                    "id": item.id,
                    "source_file": item.source_file,
                    "content_type": item.content_type,
                    "created_at": item.created_at.isoformat(),
                    "summary": item.summary or item.content[:100] + "..."
                }
                for item in recent_items[:10]
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get knowledge statistics: {e}")
            return {}
    
    async def cleanup_old_knowledge(self, days: int = 90):
        """Clean up old, unused knowledge items"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            items_to_delete = []
            
            for item_id, item in self._knowledge_items.items():
                # Delete if old and never accessed
                if item.created_at < cutoff_date and item.access_count == 0:
                    items_to_delete.append(item_id)
            
            # Delete old items
            for item_id in items_to_delete:
                await self.delete_knowledge_item(item_id)
            
            logger.info(f"Cleaned up {len(items_to_delete)} old knowledge items")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old knowledge: {e}")