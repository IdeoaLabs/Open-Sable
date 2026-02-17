# Examples - Advanced Features

This directory contains examples for all Open-Sable advanced features.

## Available Examples

### 1. Code Execution (`code_execution_examples.py`)
- Python code execution (Fibonacci, data processing)
- JavaScript execution (prime numbers)
- Bash script execution (system info)
- Docker-based isolation
- Resource limits and timeouts
- Execution caching

**Run**: `python examples/code_execution_examples.py`

### 2. File Management (`file_management_examples.py`)
- File upload/download with progress tracking
- File operations (copy, move, rename, delete)
- Directory management
- File search by pattern
- Compression (ZIP) and extraction
- Storage statistics and quotas

**Run**: `python examples/file_management_examples.py`

### 3. Database Operations (`database_examples.py`)
- SQLite CRUD operations
- Transactions with rollback
- Connection pooling
- PostgreSQL, MySQL, MongoDB, Redis configs
- Query builder
- Batch operations

**Run**: `python examples/database_examples.py`

### 4. API Integration (`api_examples.py`)
- REST API calls (GET, POST, PUT, DELETE)
- Authentication (API key, Bearer token)
- Retry logic with exponential backoff
- Response caching with TTL
- Rate limiting
- GraphQL queries
- Batch concurrent requests

**Run**: `python examples/api_examples.py`

### 5. RAG (Retrieval-Augmented Generation) (`rag_examples.py`)
- Document ingestion with metadata
- Vector embeddings and semantic search
- Filtered search by metadata
- Context retrieval for LLMs
- Hybrid search (vector + keyword)
- Batch document operations

**Run**: `python examples/rag_examples.py`

### 6. Monitoring (`monitoring_examples.py`)
- Prometheus metrics (Counter, Gauge, Histogram, Summary)
- Health checks (liveness, readiness)
- Performance monitoring
- Grafana dashboard generation
- Alert rules configuration

**Run**: `python examples/monitoring_examples.py`

### 7. Image Generation & Vision (`image_examples.py`)
- Image generation (DALL-E, Stable Diffusion)
- OCR text extraction (Tesseract, PaddleOCR)
- Image analysis (face detection, color extraction)
- Batch image processing
- Image metadata extraction

**Run**: `python examples/image_examples.py`

### 8. Advanced AI (`advanced_ai_examples.py`)
- Prompt template library
- Chain of Thought reasoning
- Self-reflection and critique
- Few-shot learning
- Template management and export

**Run**: `python examples/advanced_ai_examples.py`

### 9. Enterprise Features (`enterprise_examples.py`)
- Multi-tenancy with quotas
- RBAC (Role-Based Access Control)
- Audit logging and queries
- SSO (Single Sign-On) with JWT
- Session management
- Resource quota enforcement

**Run**: `python examples/enterprise_examples.py`

### 10. Observability (`observability_examples.py`)
- Distributed tracing (OpenTelemetry)
- Trace context propagation
- Span events and error tracking
- Log aggregation (ELK, Loki)
- Trace-log correlation
- Export to Jaeger/Zipkin

**Run**: `python examples/observability_examples.py`

### 11. Workflow Persistence (`workflow_examples.py`)
- Workflow execution with dependencies
- Checkpoint creation and recovery
- Resume from checkpoint
- Error recovery strategies (retry, skip, rollback)
- Workflow templates
- Parallel step execution

**Run**: `python examples/workflow_examples.py`

### 12. Interface SDK (`interface_sdk_examples.py`)
- Custom interface development
- Message formatting and serialization
- Event handling (connect, disconnect, message, error)
- WebSocket, REST API, Webhook interfaces
- Interface registry
- Rate limiting and metrics

**Run**: `python examples/interface_sdk_examples.py`

### 13. Web Scraping (`scraping_examples.py`)
- Action recording
- Data extraction
- Pagination handling
- Dynamic content
- Anti-bot detection
- Recipe management

**Run**: `python examples/scraping_examples.py`

## Running Examples

All examples can be run directly with Python:

```bash
# Code execution and sandbox
python examples/code_execution_examples.py

# File operations and storage
python examples/file_management_examples.py

# Database connections and queries
python examples/database_examples.py

# API integration and HTTP clients
python examples/api_examples.py

# RAG vector search
python examples/rag_examples.py

# Metrics and health checks
python examples/monitoring_examples.py

# Image generation and OCR
python examples/image_examples.py

# Prompt engineering and reasoning
python examples/advanced_ai_examples.py

# Multi-tenancy and RBAC
python examples/enterprise_examples.py

# Distributed tracing and logging
python examples/observability_examples.py

# Workflow orchestration
python examples/workflow_examples.py

# Custom interface development
python examples/interface_sdk_examples.py

# Web scraping automation
python examples/scraping_examples.py
```

## Using with CLI

You can also use the Open-Sable CLI to run these features:

```bash
# Image generation
opensable image generate "A futuristic AI robot" --size 512x512

# OCR text extraction
opensable image ocr path/to/image.png

# Prompt templates
opensable prompts list
opensable prompts render summarize --text "Your text here" --max-words 50

# Chain of Thought reasoning
opensable prompts reason "Complex question here"

# Tenant management
opensable tenants create "Company Name" --plan enterprise
opensable tenants list

# Workflow execution
opensable workflow run my_workflow_id
opensable workflow status my_workflow_id
opensable workflow resume my_workflow_id

# Interface management  
opensable interfaces list
opensable interfaces create websocket --port 8765
```

## Requirements

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Additional dependencies for specific features:

- **Image Generation**: `diffusers`, `torch`, `transformers`
- **OCR**: `pytesseract`, `paddleocr`
- **Tracing**: `opentelemetry-sdk`, `opentelemetry-exporter-jaeger`
- **Databases**: `asyncpg`, `aiomysql`, `motor`, `redis`

## Configuration

Some examples require configuration files or environment variables:

```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
JAEGER_ENDPOINT=http://localhost:14268/api/traces
ELASTICSEARCH_URL=http://localhost:9200
```

## Learning Path

Recommended order for learning Open-Sable:

1. **Start**: Code Execution → File Management → Database
2. **Core Skills**: API Integration → RAG → Monitoring
3. **Advanced**: Image/Vision → Advanced AI → Enterprise
4. **Production**: Observability → Workflow → Interface SDK

## Support

For issues or questions:
- Documentation: `/docs`
- GitHub Issues: https://github.com/yourusername/Open-Sable/issues
- Examples: This directory contains 100+ working examples

## Contributing

To add new examples:
1. Create a new file in this directory
2. Follow the existing example structure
3. Add comprehensive demonstrations
4. Update this README
5. Submit a pull request

## Integration with Agents

All skills can be used by agents for autonomous tasks:

```python
from core.agent import SableAgent
from skills.code_executor import CodeExecutor
from skills.file_manager import FileManager
from skills.database_skill import DatabaseManager
from skills.api_client import APIClient
from skills.rag_skill import RAGSystem
from skills.image_skill import ImageGenerator
from core.advanced_ai import PromptLibrary, ChainOfThought

# Agent can execute code
agent.add_skill("execute_code", code_executor.execute)

# Agent can manage files
agent.add_skill("manage_files", file_manager.upload_file)

# Agent can query databases
agent.add_skill("query_database", db_manager.execute)

# Agent can call APIs
agent.add_skill("call_api", api_client.get)

# Agent can search knowledge base
agent.add_skill("search_knowledge", rag.search)

# Agent can generate images
agent.add_skill("generate_image", image_generator.generate)

# Agent can use advanced reasoning
agent.add_skill("reason", chain_of_thought.reason)
```

## Next Steps

1. **Customize Examples**: Modify examples for your use case
2. **Create Workflows**: Combine skills for complex tasks
3. **Add Skills**: Extend with custom skills
4. **Monitor Performance**: Use metrics and tracing
5. **Scale Up**: Deploy with Docker/Kubernetes

## Documentation

See individual example files for detailed documentation and code comments. Each file contains 10+ comprehensive examples demonstrating the feature in different scenarios.
