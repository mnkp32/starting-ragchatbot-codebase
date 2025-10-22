# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `chmod +x run.sh && ./run.sh`
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`

### Code Quality
- **Format code**: `chmod +x scripts/format.sh && ./scripts/format.sh`
- **Check code quality**: `chmod +x scripts/quality.sh && ./scripts/quality.sh`
- **Manual formatting**: `uv run black backend/ main.py`
- **Import sorting**: `uv run isort backend/ main.py`
- **Linting**: `uv run flake8 backend/ main.py`

### Environment Setup
- Create `.env` file in root with: `ANTHROPIC_API_KEY=your_key_here`
- Requires Python 3.13+ and uv package manager

### Accessing the Application
- Web interface: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Architecture Overview

### Core RAG Pipeline
The system implements a Retrieval-Augmented Generation (RAG) architecture with tool-based search:

1. **Query Processing**: User queries are processed through FastAPI endpoints
2. **Tool-Based Search**: Claude AI decides when to search course materials using available tools
3. **Vector Search**: ChromaDB performs semantic search on course embeddings
4. **Response Generation**: Claude synthesizes search results into contextual responses

### Key Components

**Backend Architecture** (`backend/`):
- `app.py` - FastAPI application with CORS and static file serving
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Anthropic Claude API integration with tool calling
- `search_tools.py` - Tool interface for course content search
- `vector_store.py` - ChromaDB wrapper for vector storage and retrieval
- `document_processor.py` - Text chunking and course parsing
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic models for Course, Lesson, and CourseChunk

**Frontend** (`frontend/`): Vanilla HTML/CSS/JavaScript with markdown rendering

### Data Flow
1. Documents in `docs/` are processed into chunks at startup
2. Course metadata and content chunks stored in ChromaDB collections
3. User queries trigger tool-based search when course-specific information is needed
4. Search results are formatted with course/lesson context and returned as sources

### Key Design Patterns

**Tool-Based Architecture**: The AI uses a `CourseSearchTool` that can:
- Search across all courses or filter by course name/lesson number
- Perform semantic search using sentence transformers
- Track sources for UI display

**Session Management**: Maintains conversation context with configurable history limits

**Dual Storage Strategy**: 
- Course metadata collection for course-level queries
- Content chunks collection for detailed content search

### Configuration
- All settings centralized in `config.py` with environment variable support
- Key parameters: chunk size (800), overlap (100), max results (5), conversation history (2)
- Default model: Claude Sonnet 4 with temperature 0 for consistent responses

### Document Processing
- Supports PDF, DOCX, and TXT files
- Automatic course title and lesson extraction
- Configurable text chunking with overlap for context preservation
- make sure to use uv to manage all dependencies
- use uv to run python files