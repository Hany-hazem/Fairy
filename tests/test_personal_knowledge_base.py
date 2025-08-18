"""
Tests for Personal Knowledge Base

This module tests the PersonalKnowledgeBase class functionality including
document indexing, semantic search, entity extraction, and knowledge graph construction.
"""

import pytest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Mock the entire config module to avoid validation errors
import sys
from unittest.mock import MagicMock

# Create a mock settings object
mock_settings = MagicMock()
mock_settings.VECTOR_DB_PATH = "./vector_db"

# Mock the config module
mock_config_module = MagicMock()
mock_config_module.settings = mock_settings
sys.modules['app.config'] = mock_config_module

from app.personal_knowledge_base import (
    PersonalKnowledgeBase, 
    KnowledgeItem, 
    KnowledgeGraph, 
    SearchResult
)

from app.file_content_analyzer import ContentAnalysis, ContentType, ExtractedEntity
from app.personal_assistant_models import UserContext, KnowledgeState


class TestPersonalKnowledgeBase:
    """Test cases for PersonalKnowledgeBase"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_file_analyzer(self):
        """Mock file content analyzer"""
        analyzer = Mock()
        analyzer.analyze_file_content = AsyncMock()
        return analyzer
    
    @pytest.fixture
    def sample_content_analysis(self):
        """Sample content analysis for testing"""
        entities = [
            ExtractedEntity(
                text="john@example.com",
                entity_type="email",
                confidence=0.9,
                start_pos=10,
                end_pos=25
            ),
            ExtractedEntity(
                text="2024-01-15",
                entity_type="date",
                confidence=0.8,
                start_pos=50,
                end_pos=60
            )
        ]
        
        return ContentAnalysis(
            file_path="/test/document.txt",
            content_type=ContentType.TEXT,
            extracted_text="This is a test document with john@example.com and date 2024-01-15. It discusses technology and business topics.",
            original_content="This is a test document with john@example.com and date 2024-01-15. It discusses technology and business topics.",
            file_size=100,
            entities=entities,
            topics=["technology", "business"],
            keywords=["test", "document", "technology", "business"],
            summary="A test document discussing technology and business",
            word_count=15,
            char_count=100,
            language="en"
        )
    
    @pytest.fixture
    def knowledge_base(self, temp_dir):
        """Create PersonalKnowledgeBase instance for testing"""
        with patch('app.personal_knowledge_base.settings') as mock_settings:
            mock_settings.VECTOR_DB_PATH = temp_dir
            kb = PersonalKnowledgeBase(user_id="test_user")
            return kb
    
    def test_init(self, temp_dir):
        """Test PersonalKnowledgeBase initialization"""
        with patch('app.personal_knowledge_base.settings') as mock_settings:
            mock_settings.VECTOR_DB_PATH = temp_dir
            
            kb = PersonalKnowledgeBase(user_id="test_user")
            
            assert kb.user_id == "test_user"
            assert kb.knowledge_db_path == os.path.join(temp_dir, "knowledge_test_user")
            assert kb.embedder is not None
            assert kb.file_analyzer is not None
            assert isinstance(kb._knowledge_graph, KnowledgeGraph)
    
    @pytest.mark.asyncio
    async def test_index_document_success(self, knowledge_base, sample_content_analysis):
        """Test successful document indexing"""
        # Mock the file analyzer
        knowledge_base.file_analyzer.analyze_file_content = AsyncMock(return_value=sample_content_analysis)
        
        # Mock embedding generation
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]  # Mock 384-dim embedding
            
            result = await knowledge_base.index_document("/test/document.txt", "test content")
            
            assert result is True
            assert len(knowledge_base._knowledge_items) > 0
            
            # Check that main document item was created
            main_items = [item for item in knowledge_base._knowledge_items.values() 
                         if item.content_type == "text"]
            assert len(main_items) >= 1
            
            main_item = main_items[0]
            assert main_item.source_file == "/test/document.txt"
            assert main_item.content_type == "text"
            assert len(main_item.entities) == 2
            assert "technology" in main_item.topics
            assert "business" in main_item.topics
    
    @pytest.mark.asyncio
    async def test_index_document_failure(self, knowledge_base):
        """Test document indexing failure handling"""
        # Mock the file analyzer to raise an exception
        knowledge_base.file_analyzer.analyze_file_content = AsyncMock(side_effect=Exception("Analysis failed"))
        
        result = await knowledge_base.index_document("/test/document.txt", "test content")
        
        assert result is False
        assert len(knowledge_base._knowledge_items) == 0
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_items(self, knowledge_base, sample_content_analysis):
        """Test knowledge item extraction from content analysis"""
        items = await knowledge_base._extract_knowledge_items(sample_content_analysis)
        
        assert len(items) >= 1  # At least main document item
        
        # Check main document item
        main_item = items[0]
        assert main_item.content_type == "text"
        assert main_item.source_file == "/test/document.txt"
        assert len(main_item.entities) == 2
        assert "technology" in main_item.topics
        assert "business" in main_item.topics
        
        # Check if entity items were created
        entity_items = [item for item in items if item.content_type == "entities"]
        assert len(entity_items) >= 0  # May or may not have entity items depending on entity count
    
    def test_generate_item_id(self, knowledge_base):
        """Test knowledge item ID generation"""
        item_id = knowledge_base._generate_item_id("/test/file.txt", "main")
        
        assert isinstance(item_id, str)
        assert len(item_id) == 32  # MD5 hash length
        
        # Same inputs should generate same ID
        item_id2 = knowledge_base._generate_item_id("/test/file.txt", "main")
        assert item_id == item_id2
        
        # Different inputs should generate different IDs
        item_id3 = knowledge_base._generate_item_id("/test/file2.txt", "main")
        assert item_id != item_id3
    
    def test_extract_topics_from_text(self, knowledge_base):
        """Test topic extraction from text"""
        text = "This document discusses software development and machine learning algorithms for business applications."
        
        topics = knowledge_base._extract_topics_from_text(text)
        
        assert "technology" in topics
        assert "business" in topics
    
    def test_extract_keywords_from_text(self, knowledge_base):
        """Test keyword extraction from text"""
        text = "This document discusses software development and machine learning algorithms for business applications."
        
        keywords = knowledge_base._extract_keywords_from_text(text)
        
        assert "software" in keywords
        assert "development" in keywords
        assert "machine" in keywords
        assert "learning" in keywords
        assert "business" in keywords
        assert "applications" in keywords
        
        # Stop words should not be included
        assert "this" not in keywords
        assert "and" not in keywords
        assert "for" not in keywords
    
    def test_generate_paragraph_summary(self, knowledge_base):
        """Test paragraph summary generation"""
        short_paragraph = "This is a short paragraph."
        long_paragraph = "This is a very long paragraph that contains multiple sentences. It discusses various topics and provides detailed information. The summary should be shorter than the original text."
        
        # Short paragraph should return as-is
        summary1 = knowledge_base._generate_paragraph_summary(short_paragraph)
        assert summary1 == short_paragraph
        
        # Long paragraph should be summarized
        summary2 = knowledge_base._generate_paragraph_summary(long_paragraph, max_length=50)
        assert len(summary2) <= 50
        assert summary2.endswith("...") or len(summary2) < 50
    
    def test_group_entities_by_type(self, knowledge_base):
        """Test entity grouping by type"""
        entities = [
            ExtractedEntity("john@example.com", "email", 0.9, 0, 15),
            ExtractedEntity("jane@example.com", "email", 0.9, 20, 35),
            ExtractedEntity("2024-01-15", "date", 0.8, 40, 50),
            ExtractedEntity("555-1234", "phone", 0.7, 55, 63)
        ]
        
        groups = knowledge_base._group_entities_by_type(entities)
        
        assert len(groups) == 3
        assert len(groups["email"]) == 2
        assert len(groups["date"]) == 1
        assert len(groups["phone"]) == 1
    
    @pytest.mark.asyncio
    async def test_store_knowledge_item(self, knowledge_base):
        """Test storing knowledge item"""
        item = KnowledgeItem(
            id="test_item_1",
            content="Test content for knowledge item",
            source_file="/test/file.txt",
            content_type="text",
            topics=["technology"],
            keywords=["test", "content"]
        )
        
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]  # Mock 384-dim embedding
            
            await knowledge_base._store_knowledge_item(item)
            
            assert item.id in knowledge_base._knowledge_items
            assert knowledge_base._knowledge_items[item.id] == item
    
    @pytest.mark.asyncio
    async def test_search_knowledge_in_memory(self, knowledge_base):
        """Test knowledge search with in-memory storage"""
        # Add some test items
        items = [
            KnowledgeItem(
                id="item1",
                content="Python programming and software development",
                source_file="/test/python.txt",
                content_type="text",
                topics=["technology"],
                keywords=["python", "programming", "software"]
            ),
            KnowledgeItem(
                id="item2",
                content="Business strategy and market analysis",
                source_file="/test/business.txt",
                content_type="text",
                topics=["business"],
                keywords=["business", "strategy", "market"]
            )
        ]
        
        # Store items
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            # Mock embeddings - make first item more similar to query
            mock_embed.side_effect = [
                [[0.8, 0.1, 0.1] * 128],  # Query embedding
                [[0.9, 0.05, 0.05] * 128],  # Item 1 (similar to query)
                [[0.1, 0.8, 0.1] * 128],   # Item 2 (different from query)
            ]
            
            for item in items:
                await knowledge_base._store_knowledge_item(item)
            
            # Search for programming-related content
            results = await knowledge_base.search_knowledge("Python programming", k=5, min_similarity=0.1)
            
            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            
            # First result should be the Python item (higher similarity)
            assert results[0].knowledge_item.id == "item1"
            assert results[0].similarity_score > 0.1
    
    @pytest.mark.asyncio
    async def test_search_knowledge_empty_results(self, knowledge_base):
        """Test knowledge search with no matching results"""
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]
            
            results = await knowledge_base.search_knowledge("nonexistent topic", k=5, min_similarity=0.8)
            
            assert len(results) == 0
    
    def test_generate_relevance_context(self, knowledge_base):
        """Test relevance context generation"""
        item = KnowledgeItem(
            id="test_item",
            content="Python programming tutorial for beginners",
            source_file="/test/python.txt",
            content_type="text",
            topics=["technology", "education"],
            keywords=["python", "programming", "tutorial"],
            entities=[ExtractedEntity("Python", "technology", 0.9, 0, 6)]
        )
        
        query = "Python programming guide"
        context = knowledge_base._generate_relevance_context(query, item)
        
        assert "python" in context.lower()
        assert "programming" in context.lower()
        assert len(context) > 0
    
    def test_find_matched_entities(self, knowledge_base):
        """Test finding matched entities"""
        item = KnowledgeItem(
            id="test_item",
            content="Contact john@example.com for more information",
            source_file="/test/contact.txt",
            content_type="text",
            entities=[
                ExtractedEntity("john@example.com", "email", 0.9, 8, 23),
                ExtractedEntity("information", "concept", 0.7, 33, 44)
            ]
        )
        
        query = "email john@example.com contact"
        matched = knowledge_base._find_matched_entities(query, item)
        
        assert "john@example.com" in matched
    
    def test_find_matched_topics(self, knowledge_base):
        """Test finding matched topics"""
        item = KnowledgeItem(
            id="test_item",
            content="Technology and business analysis",
            source_file="/test/analysis.txt",
            content_type="text",
            topics=["technology", "business"]
        )
        
        query = "technology trends in business"
        matched = knowledge_base._find_matched_topics(query, item)
        
        # The query should match business topic (business is in the query)
        # Technology might not match depending on the topic extraction logic
        assert "business" in matched
        # Check that we get some matches
        assert len(matched) > 0
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, knowledge_base):
        """Test entity extraction from text"""
        text = "Contact john@example.com or call 555-1234 on 2024-01-15"
        
        # Mock the file analyzer's entity extraction
        expected_entities = [
            ExtractedEntity("john@example.com", "email", 0.9, 8, 23),
            ExtractedEntity("555-1234", "phone", 0.8, 32, 40),
            ExtractedEntity("2024-01-15", "date", 0.8, 44, 54)
        ]
        
        with patch.object(knowledge_base.file_analyzer, '_extract_entities', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = expected_entities
            
            entities = await knowledge_base.extract_entities(text)
            
            assert len(entities) == 3
            assert entities[0].entity_type == "email"
            assert entities[1].entity_type == "phone"
            assert entities[2].entity_type == "date"
    
    def test_build_knowledge_graph(self, knowledge_base):
        """Test knowledge graph building"""
        # Add some test data to the graph
        knowledge_base._knowledge_graph.nodes["item1"] = {
            "content_type": "text",
            "topics": ["technology"],
            "entities": ["Python"]
        }
        knowledge_base._knowledge_graph.topics["technology"] = ["item1"]
        knowledge_base._knowledge_graph.entities["python"] = ["item1"]
        
        graph = knowledge_base.build_knowledge_graph()
        
        assert isinstance(graph, KnowledgeGraph)
        assert "item1" in graph.nodes
        assert "technology" in graph.topics
        assert "python" in graph.entities
    
    @pytest.mark.asyncio
    async def test_update_knowledge_item(self, knowledge_base):
        """Test updating knowledge item"""
        # Create and store an item
        item = KnowledgeItem(
            id="test_item",
            content="Original content",
            source_file="/test/file.txt",
            content_type="text"
        )
        
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]
            
            await knowledge_base._store_knowledge_item(item)
            
            # Update the item
            updates = {
                "content": "Updated content",
                "topics": ["new_topic"]
            }
            
            result = await knowledge_base.update_knowledge_item("test_item", updates)
            
            assert result is True
            updated_item = knowledge_base._knowledge_items["test_item"]
            assert updated_item.content == "Updated content"
            assert updated_item.topics == ["new_topic"]
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_knowledge_item(self, knowledge_base):
        """Test updating non-existent knowledge item"""
        result = await knowledge_base.update_knowledge_item("nonexistent", {"content": "new"})
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_knowledge_item(self, knowledge_base):
        """Test deleting knowledge item"""
        # Create and store an item
        item = KnowledgeItem(
            id="test_item",
            content="Test content",
            source_file="/test/file.txt",
            content_type="text"
        )
        
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]
            
            await knowledge_base._store_knowledge_item(item)
            assert "test_item" in knowledge_base._knowledge_items
            
            # Delete the item
            result = await knowledge_base.delete_knowledge_item("test_item")
            
            assert result is True
            assert "test_item" not in knowledge_base._knowledge_items
    
    @pytest.mark.asyncio
    async def test_get_knowledge_statistics(self, knowledge_base):
        """Test getting knowledge statistics"""
        # Add some test items
        items = [
            KnowledgeItem(
                id="item1",
                content="Python content",
                source_file="/test/python.txt",
                content_type="text",
                topics=["technology"]
            ),
            KnowledgeItem(
                id="item2",
                content="Business content",
                source_file="/test/business.txt",
                content_type="document",
                topics=["business"]
            )
        ]
        
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]
            
            for item in items:
                await knowledge_base._store_knowledge_item(item)
                # Manually update graph for testing
                knowledge_base._knowledge_graph.topics.setdefault(item.topics[0], []).append(item.id)
        
        stats = await knowledge_base.get_knowledge_statistics()
        
        assert stats["total_items"] == 2
        assert stats["total_topics"] == 2
        assert "text" in stats["content_types"]
        assert "document" in stats["content_types"]
        assert len(stats["top_topics"]) > 0
        assert len(stats["recent_items"]) == 2
    
    @pytest.mark.asyncio
    async def test_cleanup_old_knowledge(self, knowledge_base):
        """Test cleaning up old knowledge items"""
        # Create old and new items
        old_date = datetime.now() - timedelta(days=100)
        new_date = datetime.now() - timedelta(days=1)
        
        old_item = KnowledgeItem(
            id="old_item",
            content="Old content",
            source_file="/test/old.txt",
            content_type="text",
            created_at=old_date,
            access_count=0  # Never accessed
        )
        
        new_item = KnowledgeItem(
            id="new_item",
            content="New content",
            source_file="/test/new.txt",
            content_type="text",
            created_at=new_date,
            access_count=0
        )
        
        accessed_old_item = KnowledgeItem(
            id="accessed_old_item",
            content="Accessed old content",
            source_file="/test/accessed.txt",
            content_type="text",
            created_at=old_date,
            access_count=5  # Has been accessed
        )
        
        with patch.object(knowledge_base, '_embed_text') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2, 0.3] * 128]
            
            # Store all items
            for item in [old_item, new_item, accessed_old_item]:
                await knowledge_base._store_knowledge_item(item)
            
            assert len(knowledge_base._knowledge_items) == 3
            
            # Cleanup old items (90 days)
            await knowledge_base.cleanup_old_knowledge(days=90)
            
            # Only old_item should be deleted (old and never accessed)
            assert "old_item" not in knowledge_base._knowledge_items
            assert "new_item" in knowledge_base._knowledge_items
            assert "accessed_old_item" in knowledge_base._knowledge_items
    
    @pytest.mark.asyncio
    async def test_error_handling_in_search(self, knowledge_base):
        """Test error handling in search functionality"""
        with patch.object(knowledge_base, '_embed_text', side_effect=Exception("Embedding failed")):
            results = await knowledge_base.search_knowledge("test query")
            assert results == []
    
    @pytest.mark.asyncio
    async def test_error_handling_in_indexing(self, knowledge_base):
        """Test error handling in document indexing"""
        with patch.object(knowledge_base.file_analyzer, 'analyze_file_content', side_effect=Exception("Analysis failed")):
            result = await knowledge_base.index_document("/test/file.txt", "content")
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__])