<div align="center">
  <img src="cover.png" alt="CatLang Cover" width="800"/>
</div>

# CatLang - n8n to LangGraph Converter with MCP Server

Framework for converting n8n workflows into LangGraph implementations. Available as an **MCP Server** for integration with Claude Desktop and other MCP clients, as well as a standalone tool.

## Quick Start

**Prerequisites:** Python 3.10+, Claude Desktop (for MCP), one of the LLM API keys (e.g., OpenAI, OpenRouter, or xAI)

**Installation:**

```bash
git clone <repository-url>
cd catlang
pip install -r requirements.txt
cp env_template.txt .env
# Edit .env and add at least one LLM API key (OPENAI_API_KEY, OPENROUTER_API_KEY, or XAI_API_KEY)
# Set LLM_PROVIDER to your preferred provider (openai, openrouter, or xai)
```

Configure Claude Desktop to use the MCP server, then start using the tools in Claude Desktop!

## MCP Server Tools

1. **`analyze_n8n_workflow`** - Analyze n8n workflow JSON and generate production requirements
2. **`extract_custom_logic`** - Extract custom logic from code nodes (Python/JavaScript)
3. **`generate_langgraph_implementation`** - Generate LangGraph code from requirements
4. **`validate_implementation`** - Validate LangGraph code for syntax, compliance, and best practices
5. **`list_guides`** / **`query_guide`** - Access implementation guides

**Resources:** Access guides via URI scheme `guide://docs/{category}/{name}`

## Conversion Process (3 Phases)

**Phase 1: Production Requirements** - Analyze n8n workflow JSON into technical specifications  
**Phase 2: Custom Logic Extraction** - Extract and analyze custom node code  
**Phase 3: LangGraph Implementation** - Generate complete LangGraph implementation

See `OrchestrationPrompts/` for details on each phase.

## Key Concepts

**Paradigm Selection:**
- **Functional API (`@entrypoint`)**: For simple to medium workflows with sequential processing
- **Graph API (`StateGraph`)**: For complex workflows with state management and parallel execution

See `guides/paradigm-selection.md` for details.

## Project Structure

```
catlang/
├── config/              # MCP server Configuration & LLM providers
├── guides/              # Implementation guides (MCP resources)
├── src/                 # Source code (MCP server, tools, resources)
├── tests/               # Test suite
├── requirements.txt
└── run_mcp_server.py    # Entry point MCP server
```

## License

MIT License - Copyright (c) 2026 Catlang, Muhammad Fajar Agus Saputra
MIT License - Copyright (c) 2025 MADAILAB, Rohit Aggarwal , Hitesh Balegar
