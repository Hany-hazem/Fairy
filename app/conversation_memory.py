# app/conversation_memory.py
"""
Enhanced memory management for conversation context retrieval and summarization
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime, timedelta

try:
    import chromadb
    from chromadb import Client
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    Client = None

from sentence_transformers import SentenceTransformer
from .config import settings
from .models import Message, ConversationSession, ConversationContext
from .llm_studio_client import get_studio_client

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """Enhanced memory manager for conversation context and summarization"""
    
    def __init__(self, vector_db_path: str = None):
        self.vector_db_path = vector_db_path or settings.VECTOR_DB_PATH
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._client = None
        self._collection = None
        
        # Initialize vector database
        self._init_vector_db()
    
    def _init_vector_db(self):
        """Initialize ChromaDB or fallback to in-memory storage"""
        try:
            if CHROMADB_AVAILABLE:
                self._client = chromadb.PersistentClient(path=self.vector_db_path)
                self._collection = self._client.get_or_create_collection(
                    name="conversation_memory",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Initialized ChromaDB at {self.vector_db_path}")
            else:
                # Fallback to simple in-memory storage
                self._memory_store = []
                logger.warning("ChromaDB not available, using in-memory storage")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            self._memory_store = []
    
    def _embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text using sentence transformers"""
        try:
            return self.embedder.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), 384))  # all-MiniLM-L6-v2 dimension
    
    def store_conversation_context(self, session: ConversationSession):
        """Store conversation context for future retrieval"""
        if not session.messages:
            return
        
        # Create context chunks from conversation
        context_chunks = self._create_context_chunks(session)
        
        for chunk in context_chunks:
            self._store_chunk(chunk, session.id, session.user_id)
    
    def _create_context_chunks(self, session: ConversationSession) -> List[Dict]:
        """Create meaningful context chunks from conversation"""
        chunks = []
        
        # Create chunks from message pairs (user + assistant)
        user_messages = [msg for msg in session.messages if msg.role == "user"]
        assistant_messages = [msg for msg in session.messages if msg.role == "assistant"]
        
        for i, user_msg in enumerate(user_messages):
            # Find corresponding assistant response
            assistant_msg = None
            for a_msg in assistant_messages:
                if a_msg.timestamp > user_msg.timestamp:
                    assistant_msg = a_msg
                    break
            
            # Create context chunk
            chunk_text = f"User: {user_msg.content}"
            if assistant_msg:
                chunk_text += f"\nAssistant: {assistant_msg.content}"
            
            chunks.append({
                "text": chunk_text,
                "user_message_id": user_msg.id,
                "assistant_message_id": assistant_msg.id if assistant_msg else None,
                "timestamp": user_msg.timestamp,
                "topics": self._extract_topics(chunk_text)
            })
        
        return chunks
    
    def _extract_topics(self, text: str) -> List[str]:
        """Simple topic extraction from text"""
        # Simple keyword-based topic extraction
        # In a production system, you'd use more sophisticated NLP
        keywords = []
        
        # Common topic indicators
        topic_words = [
            "weather", "temperature", "rain", "sunny", "cloudy",
            "food", "recipe", "cooking", "restaurant", "meal",
            "travel", "vacation", "flight", "hotel", "destination",
            "work", "job", "career", "meeting", "project",
            "health", "exercise", "doctor", "medicine", "fitness",
            "technology", "computer", "software", "programming", "AI",
            "entertainment", "movie", "music", "book", "game"
        ]
        
        text_lower = text.lower()
        for word in topic_words:
            if word in text_lower:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _store_chunk(self, chunk: Dict, session_id: str, user_id: Optional[str]):
        """Store a context chunk in the vector database"""
        try:
            if self._collection is not None:
                # ChromaDB storage
                embedding = self._embed_text([chunk["text"]])[0]
                
                chunk_id = f"{session_id}_{chunk['user_message_id']}"
                
                self._collection.add(
                    documents=[chunk["text"]],
                    embeddings=[embedding.tolist()],
                    ids=[chunk_id],
                    metadatas=[{
                        "session_id": session_id,
                        "user_id": user_id or "",
                        "timestamp": chunk["timestamp"].isoformat(),
                        "topics": ",".join(chunk["topics"]) if chunk["topics"] else "",
                        "user_message_id": chunk["user_message_id"],
                        "assistant_message_id": chunk.get("assistant_message_id") or ""
                    }]
                )
            else:
                # In-memory fallback
                embedding = self._embed_text([chunk["text"]])[0]
                self._memory_store.append({
                    "text": chunk["text"],
                    "embedding": embedding,
                    "session_id": session_id,
                    "user_id": user_id,
                    "timestamp": chunk["timestamp"],
                    "topics": chunk["topics"]
                })
                
        except Exception as e:
            logger.error(f"Failed to store context chunk: {e}")
    
    def retrieve_relevant_context(self, query: str, session_id: str = None, 
                                user_id: str = None, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant conversation context based on query"""
        try:
            if self._collection is not None:
                # ChromaDB retrieval
                query_embedding = self._embed_text([query])[0]
                
                # Build where clause for filtering
                where_clause = {}
                if session_id:
                    where_clause["session_id"] = session_id
                if user_id:
                    where_clause["user_id"] = user_id
                
                results = self._collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where=where_clause if where_clause else None
                )
                
                if results["documents"] and results["documents"][0]:
                    docs = results["documents"][0]
                    distances = results["distances"][0]
                    # Convert distances to similarity scores (higher is better)
                    similarities = [1 - d for d in distances]
                    return list(zip(docs, similarities))
                
            else:
                # In-memory fallback
                query_embedding = self._embed_text([query])[0]
                results = []
                
                for item in self._memory_store:
                    # Apply filters
                    if session_id and item["session_id"] != session_id:
                        continue
                    if user_id and item["user_id"] != user_id:
                        continue
                    
                    # Calculate similarity
                    similarity = np.dot(query_embedding, item["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
                    )
                    results.append((item["text"], float(similarity)))
                
                # Sort by similarity and return top k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
                
        except Exception as e:
            logger.error(f"Failed to retrieve relevant context: {e}")
        
        return []
    
    def generate_context_summary(self, session: ConversationSession, 
                                max_length: int = 200) -> Optional[str]:
        """Generate a summary of conversation context using LLM"""
        if len(session.messages) < 3:
            return None
        
        try:
            # Prepare conversation text for summarization
            conversation_text = self._format_conversation_for_summary(session)
            
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of this conversation in {max_length} characters or less. Focus on the main topics discussed and key information exchanged.

Conversation:
{conversation_text}

Summary:"""
            
            # Use LM Studio to generate summary
            studio_client = get_studio_client()
            summary = studio_client.chat(
                prompt=prompt,
                max_new_tokens=max_length // 4,  # Rough estimate for tokens
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            # Clean up the summary
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            logger.info(f"Generated summary for session {session.id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate context summary: {e}")
            # Fallback to simple summary
            return self._create_simple_summary(session, max_length)
    
    def _format_conversation_for_summary(self, session: ConversationSession) -> str:
        """Format conversation for summarization"""
        lines = []
        for msg in session.messages[-10:]:  # Last 10 messages
            role = msg.role.title()
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _create_simple_summary(self, session: ConversationSession, max_length: int) -> str:
        """Create a simple rule-based summary as fallback"""
        user_messages = [msg for msg in session.messages if msg.role == "user"]
        assistant_messages = [msg for msg in session.messages if msg.role == "assistant"]
        
        # Extract topics from all messages
        all_topics = set()
        for msg in session.messages:
            topics = self._extract_topics(msg.content)
            all_topics.update(topics)
        
        # Create summary
        summary_parts = []
        if user_messages:
            summary_parts.append(f"User discussed {len(user_messages)} topics")
        if assistant_messages:
            summary_parts.append(f"Assistant provided {len(assistant_messages)} responses")
        if all_topics:
            topics_str = ", ".join(list(all_topics)[:3])  # First 3 topics
            summary_parts.append(f"Topics: {topics_str}")
        
        summary = "; ".join(summary_parts)
        
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def enhance_context_with_history(self, context: ConversationContext, 
                                   query: str, user_id: str = None) -> ConversationContext:
        """Enhance conversation context with relevant historical information"""
        try:
            # Retrieve relevant context from other conversations
            relevant_history = self.retrieve_relevant_context(
                query=query,
                session_id=None,  # Search across all sessions
                user_id=user_id,
                k=3
            )
            
            # Add relevant history to context
            history_texts = [text for text, score in relevant_history if score > 0.7]
            context.relevant_history = history_texts
            
            logger.debug(f"Enhanced context with {len(history_texts)} relevant history items")
            
        except Exception as e:
            logger.error(f"Failed to enhance context with history: {e}")
        
        return context
    
    def cleanup_old_memories(self, days: int = 30):
        """Clean up old conversation memories"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            if self._collection is not None:
                # ChromaDB cleanup - get all items and filter by date
                all_items = self._collection.get()
                old_ids = []
                
                for i, metadata in enumerate(all_items.get("metadatas", [])):
                    if metadata and "timestamp" in metadata:
                        timestamp = datetime.fromisoformat(metadata["timestamp"])
                        if timestamp < cutoff_date:
                            old_ids.append(all_items["ids"][i])
                
                if old_ids:
                    self._collection.delete(ids=old_ids)
                    logger.info(f"Cleaned up {len(old_ids)} old memory items")
            else:
                # In-memory cleanup
                original_count = len(self._memory_store)
                self._memory_store = [
                    item for item in self._memory_store
                    if item["timestamp"] >= cutoff_date
                ]
                cleaned_count = original_count - len(self._memory_store)
                logger.info(f"Cleaned up {cleaned_count} old memory items")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")

# Global conversation memory manager instance
conversation_memory = ConversationMemoryManager()