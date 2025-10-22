"""
Tests for VectorStore functionality.

Tests the VectorStore class that manages ChromaDB collections
for course metadata and content storage/retrieval.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestSearchResults:
    """Test cases for SearchResults utility class"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB query results"""
        chroma_results = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"course": "A"}, {"course": "B"}]],
            "distances": [[0.1, 0.2]],
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["Doc 1", "Doc 2"]
        assert results.metadata == [{"course": "A"}, {"course": "B"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Test error message")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(["doc"], [{}], [0.1])

        assert empty_results.is_empty() is True
        assert non_empty_results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore"""

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client"""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_client, mock_collection

    @pytest.fixture
    def vector_store(self, mock_chroma_client):
        """Create VectorStore with mocked ChromaDB"""
        mock_client, mock_collection = mock_chroma_client

        with (
            patch("vector_store.chromadb.PersistentClient", return_value=mock_client),
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
        ):

            store = VectorStore("./test_chroma", "test-model", max_results=3)
            store.course_catalog = mock_collection
            store.course_content = mock_collection
            return store, mock_client, mock_collection

    def test_initialization(self):
        """Test VectorStore initialization"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_client_class,
            patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding,
        ):

            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection

            store = VectorStore("./test_path", "test-embedding-model", max_results=10)

            # Verify ChromaDB client was created correctly
            mock_client_class.assert_called_once()
            assert mock_client_class.call_args[1]["path"] == "./test_path"

            # Verify embedding function was set up
            mock_embedding.assert_called_once_with(model_name="test-embedding-model")

            # Verify collections were created
            assert mock_client.get_or_create_collection.call_count == 2

            # Verify max_results is set
            assert store.max_results == 10

    def test_search_basic_query(self, vector_store):
        """Test basic search without filters"""
        store, mock_client, mock_collection = vector_store

        # Mock ChromaDB query response
        mock_collection.query.return_value = {
            "documents": [["Test document content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.1]],
        }

        results = store.search("test query")

        # Verify query was called correctly
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,  # max_results from fixture
            where=None,
        )

        # Verify results
        assert len(results.documents) == 1
        assert results.documents[0] == "Test document content"
        assert results.metadata[0]["course_title"] == "Test Course"
        assert results.error is None

    def test_search_with_course_filter(self, vector_store):
        """Test search with course name filter"""
        store, mock_client, mock_collection = vector_store

        # Mock course name resolution
        store._resolve_course_name = Mock(return_value="Resolved Course Name")

        mock_collection.query.return_value = {
            "documents": [["Filtered content"]],
            "metadatas": [[{"course_title": "Resolved Course Name"}]],
            "distances": [[0.2]],
        }

        results = store.search("test query", course_name="Partial Course")

        # Verify course name was resolved
        store._resolve_course_name.assert_called_once_with("Partial Course")

        # Verify query was called with filter
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=3,
            where={"course_title": "Resolved Course Name"},
        )

    def test_search_with_lesson_filter(self, vector_store):
        """Test search with lesson number filter"""
        store, mock_client, mock_collection = vector_store

        mock_collection.query.return_value = {
            "documents": [["Lesson content"]],
            "metadatas": [[{"lesson_number": 2}]],
            "distances": [[0.15]],
        }

        results = store.search("test query", lesson_number=2)

        # Verify query was called with lesson filter
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=3, where={"lesson_number": 2}
        )

    def test_search_with_both_filters(self, vector_store):
        """Test search with both course and lesson filters"""
        store, mock_client, mock_collection = vector_store

        store._resolve_course_name = Mock(return_value="Test Course")

        mock_collection.query.return_value = {
            "documents": [["Specific content"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.05]],
        }

        results = store.search("test query", course_name="Test", lesson_number=1)

        # Verify query was called with combined filter
        expected_filter = {
            "$and": [{"course_title": "Test Course"}, {"lesson_number": 1}]
        }
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=3, where=expected_filter
        )

    def test_search_course_not_found(self, vector_store):
        """Test search when course name cannot be resolved"""
        store, mock_client, mock_collection = vector_store

        store._resolve_course_name = Mock(return_value=None)

        results = store.search("test query", course_name="Nonexistent Course")

        # Should return error without querying
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.documents == []
        mock_collection.query.assert_not_called()

    def test_search_exception_handling(self, vector_store):
        """Test search error handling"""
        store, mock_client, mock_collection = vector_store

        mock_collection.query.side_effect = Exception("Database error")

        results = store.search("test query")

        assert "Search error: Database error" in results.error
        assert results.documents == []

    def test_resolve_course_name_success(self, vector_store):
        """Test successful course name resolution"""
        store, mock_client, mock_collection = vector_store

        # Mock catalog query for course name resolution
        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {
            "documents": [["Course Title"]],
            "metadatas": [[{"title": "Full Course Title"}]],
        }

        result = store._resolve_course_name("Partial Name")

        # Verify catalog was queried
        store.course_catalog.query.assert_called_once_with(
            query_texts=["Partial Name"], n_results=1
        )

        assert result == "Full Course Title"

    def test_resolve_course_name_not_found(self, vector_store):
        """Test course name resolution when no match found"""
        store, mock_client, mock_collection = vector_store

        store.course_catalog = Mock()
        store.course_catalog.query.return_value = {"documents": [[]], "metadatas": [[]]}

        result = store._resolve_course_name("Nonexistent Course")

        assert result is None

    def test_build_filter_combinations(self, vector_store):
        """Test filter building for different parameter combinations"""
        store, mock_client, mock_collection = vector_store

        # Test no filters
        assert store._build_filter(None, None) is None

        # Test course only
        course_filter = store._build_filter("Course A", None)
        assert course_filter == {"course_title": "Course A"}

        # Test lesson only
        lesson_filter = store._build_filter(None, 3)
        assert lesson_filter == {"lesson_number": 3}

        # Test both
        combined_filter = store._build_filter("Course B", 2)
        expected = {"$and": [{"course_title": "Course B"}, {"lesson_number": 2}]}
        assert combined_filter == expected

    def test_add_course_metadata(self, vector_store, sample_course):
        """Test adding course metadata to catalog"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()

        store.add_course_metadata(sample_course)

        # Verify course was added to catalog
        store.course_catalog.add.assert_called_once()
        call_args = store.course_catalog.add.call_args[1]

        assert call_args["documents"] == ["Introduction to RAG Systems"]
        assert call_args["ids"] == ["Introduction to RAG Systems"]

        # Verify metadata structure
        metadata = call_args["metadatas"][0]
        assert metadata["title"] == "Introduction to RAG Systems"
        assert metadata["instructor"] == "Dr. AI Expert"
        assert metadata["course_link"] == "https://example.com/rag-course"
        assert metadata["lesson_count"] == 2

        # Verify lessons are serialized as JSON
        import json

        lessons = json.loads(metadata["lessons_json"])
        assert len(lessons) == 2
        assert lessons[0]["lesson_number"] == 1
        assert lessons[0]["lesson_title"] == "What is RAG?"

    def test_add_course_content(self, vector_store, sample_course_chunks):
        """Test adding course content chunks"""
        store, mock_client, mock_collection = vector_store
        store.course_content = Mock()

        store.add_course_content(sample_course_chunks)

        # Verify content was added
        store.course_content.add.assert_called_once()
        call_args = store.course_content.add.call_args[1]

        # Verify documents
        expected_docs = [chunk.content for chunk in sample_course_chunks]
        assert call_args["documents"] == expected_docs

        # Verify metadata
        expected_metadata = [
            {
                "course_title": chunk.course_title,
                "lesson_number": chunk.lesson_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in sample_course_chunks
        ]
        assert call_args["metadatas"] == expected_metadata

        # Verify IDs format
        expected_ids = [
            "Introduction_to_RAG_Systems_0",
            "Introduction_to_RAG_Systems_1",
            "Introduction_to_RAG_Systems_2",
        ]
        assert call_args["ids"] == expected_ids

    def test_add_course_content_empty(self, vector_store):
        """Test adding empty course content list"""
        store, mock_client, mock_collection = vector_store
        store.course_content = Mock()

        store.add_course_content([])

        # Should not call add for empty list
        store.course_content.add.assert_not_called()

    def test_clear_all_data(self, vector_store):
        """Test clearing all data from collections"""
        store, mock_client, mock_collection = vector_store

        store.clear_all_data()

        # Verify collections were deleted
        assert mock_client.delete_collection.call_count == 2
        collection_names = [
            call[0][0] for call in mock_client.delete_collection.call_args_list
        ]
        assert "course_catalog" in collection_names
        assert "course_content" in collection_names

        # Verify collections were recreated
        # get_or_create_collection is called twice in init + twice in clear
        assert mock_client.get_or_create_collection.call_count >= 4

    def test_get_existing_course_titles(self, vector_store):
        """Test retrieving existing course titles"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            "ids": ["Course A", "Course B", "Course C"]
        }

        titles = store.get_existing_course_titles()

        store.course_catalog.get.assert_called_once()
        assert titles == ["Course A", "Course B", "Course C"]

    def test_get_existing_course_titles_empty(self, vector_store):
        """Test retrieving course titles when none exist"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {"ids": []}

        titles = store.get_existing_course_titles()

        assert titles == []

    def test_get_course_count(self, vector_store):
        """Test getting course count"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            "ids": ["Course 1", "Course 2", "Course 3", "Course 4"]
        }

        count = store.get_course_count()

        assert count == 4

    def test_get_course_count_empty(self, vector_store):
        """Test getting course count when no courses exist"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {"ids": []}

        count = store.get_course_count()

        assert count == 0

    def test_get_course_link(self, vector_store):
        """Test retrieving course link"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {
            "metadatas": [{"course_link": "https://example.com/course"}]
        }

        link = store.get_course_link("Test Course")

        store.course_catalog.get.assert_called_once_with(ids=["Test Course"])
        assert link == "https://example.com/course"

    def test_get_course_link_not_found(self, vector_store):
        """Test retrieving course link when course not found"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()
        store.course_catalog.get.return_value = {"metadatas": []}

        link = store.get_course_link("Nonexistent Course")

        assert link is None

    def test_get_lesson_link(self, vector_store):
        """Test retrieving lesson link"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()

        # Mock course metadata with lessons
        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}, {"lesson_number": 2, "lesson_link": "https://example.com/lesson2"}]'
        store.course_catalog.get.return_value = {
            "metadatas": [{"lessons_json": lessons_json}]
        }

        link = store.get_lesson_link("Test Course", 2)

        store.course_catalog.get.assert_called_once_with(ids=["Test Course"])
        assert link == "https://example.com/lesson2"

    def test_get_lesson_link_not_found(self, vector_store):
        """Test retrieving lesson link when lesson not found"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()

        lessons_json = (
            '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}]'
        )
        store.course_catalog.get.return_value = {
            "metadatas": [{"lessons_json": lessons_json}]
        }

        link = store.get_lesson_link("Test Course", 99)

        assert link is None

    def test_get_all_courses_metadata(self, vector_store):
        """Test retrieving all course metadata with parsed lessons"""
        store, mock_client, mock_collection = vector_store
        store.course_catalog = Mock()

        lessons_json = '[{"lesson_number": 1, "lesson_title": "Intro"}]'
        store.course_catalog.get.return_value = {
            "metadatas": [
                {
                    "title": "Test Course",
                    "instructor": "Test Instructor",
                    "lessons_json": lessons_json,
                    "lesson_count": 1,
                }
            ]
        }

        metadata = store.get_all_courses_metadata()

        assert len(metadata) == 1
        course_meta = metadata[0]
        assert course_meta["title"] == "Test Course"
        assert course_meta["instructor"] == "Test Instructor"
        assert course_meta["lesson_count"] == 1
        assert "lessons_json" not in course_meta  # Should be removed
        assert "lessons" in course_meta  # Should be parsed
        assert len(course_meta["lessons"]) == 1
        assert course_meta["lessons"][0]["lesson_number"] == 1

    def test_search_with_custom_limit(self, vector_store):
        """Test search with custom result limit"""
        store, mock_client, mock_collection = vector_store

        mock_collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.1, 0.2]],
        }

        results = store.search("test query", limit=10)

        # Verify custom limit was used instead of max_results
        mock_collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=10, where=None
        )
