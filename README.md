# CatLang - n8n to LangGraph Converter with MCP Server

A comprehensive framework for converting n8n workflows to LangGraph implementations. Available as both an **MCP (Model Context Protocol) Server** for integration with Claude Desktop and other MCP clients, and as a standalone workflow conversion tool.

## Overview

CatLang provides two ways to convert n8n workflows to LangGraph:

1. **MCP Server (Recommended)**: Integrate with Claude Desktop or any MCP-compatible client for interactive workflow conversion
2. **Standalone Tool**: Use the three-phase orchestration process directly via Python

The conversion process analyzes n8n workflow specifications, extracts custom logic, and generates production-ready LangGraph code following best practices.

## Quick Start

### As MCP Server (Recommended)

The MCP server provides interactive tools for workflow conversion through Claude Desktop or other MCP clients.

**Prerequisites:**
- Python 3.10+
- Claude Desktop (or another MCP client)
- OpenAI API key (or other LLM provider)

**Installation:**

1. Clone the repository:
```bash
git clone <repository-url>
cd catlang
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp env_template.txt .env
# Edit .env and add your OPENAI_API_KEY
```

4. Configure Claude Desktop (see [Claude Desktop Setup Guide](docs/CLAUDE_DESKTOP_SETUP.md))

5. Start using the tools in Claude Desktop!

For detailed setup instructions, see the [MCP Server Setup Guide](docs/MCP_SERVER_SETUP.md).

### As Standalone Workflow Converter

Use the three-phase orchestration process directly:

## MCP Server Usage

The MCP server exposes the following tools:

### Available Tools

1. **`analyze_n8n_workflow`** - Analyzes n8n workflow JSON and generates production requirements
2. **`extract_custom_logic`** - Extracts custom logic from code nodes (Python/JavaScript)
3. **`generate_langgraph_implementation`** - Generates LangGraph code from requirements
4. **`validate_implementation`** - Validates generated LangGraph code for syntax, compliance, and best practices
5. **`list_guides`** - Lists available implementation guides
6. **`query_guide`** - Retrieves specific implementation guide content

### Resources

The server also provides access to implementation guides as MCP resources:

- **URI Scheme**: `guide://docs/{category}/{name}`
- **Categories**: `implementation`, `paradigm`, `setup`, `testing`, `integration`, `structure`, `requirements`, `general`
- Access guides via MCP resource protocol or use the `list_guides` and `query_guide` tools

For detailed usage examples, see [Usage Examples](docs/USAGE_EXAMPLES.md).

---

## Standalone Workflow Conversion

The standalone conversion process follows a three-phase orchestration approach:

### Phase 1: Production Requirements Analysis
**Prompt:** `OrchestrationPrompts/Step1-ProductionRequirements.md`

Analyzes the n8n workflow JSON and creates detailed technical specifications:

- **Global Workflow Summary**: Objective, triggers, execution rules, security, and error handling
- **Per Node Specification**: Functionality, built-in parameters, workflow-specific configurations, data mapping, and execution paths
- **Custom Node Identification**: Identifies custom nodes that require separate analysis
- **Express.js Requirements**: Additional requirements for implementation (middleware, dependencies, rate limiting, etc.)

**Input**: n8n workflow JSON  
**Output**: Technical specification document

### Phase 2: Custom Logic Extraction
**Prompt:** `OrchestrationPrompts/Step2-CustomLogic.md`

Analyzes custom node code (Python functions, Lambda functions, etc.) and creates technical requirements:

- **Purpose**: What the code accomplishes
- **Inputs**: Parameters, data types, validation rules
- **Processing Logic**: Step-by-step transformation description
- **Outputs**: Return data structure and format
- **Dependencies**: External libraries and their purposes
- **Error Handling**: Exception types and handling strategies

**Input**: Custom node code (Python/Node.js)  
**Output**: Custom node requirements specification

### Phase 3: LangGraph Implementation
**Prompt:** `OrchestrationPrompts/Step3-MainOrchestorPrompt.md`

Converts the workflow specification into a LangGraph implementation:

1. **Guide Review**: Reads all implementation guides in `guides/` directory
2. **Workflow Analysis**: Analyzes n8n JSON and custom node requirements
3. **Implementation Planning**: Selects paradigm (Functional API vs Graph API) and execution pattern
4. **Implementation**: Creates LangGraph code with proper decorators and patterns
5. **Final Review**: Cross-references against all guides for compliance

**Input**: Production requirements + custom logic specifications  
**Output**: Complete LangGraph implementation

### Prerequisites

- Python 3.10 or higher
- n8n workflow JSON export
- Understanding of LangGraph concepts (Functional API and Graph API)
- LLM provider API key (OpenAI recommended for MVP)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd catlang
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp env_template.txt .env
# Edit .env and add your API keys
```

4. Review implementation guides in `guides/`:
   - `paradigm-selection.md` - Choosing between Functional API and Graph API
   - `functional-api-implementation.md` - Functional API patterns
   - `graph-api-implementation.md` - Graph API patterns
   - `authentication-setup.md` - Authentication patterns
   - `api-integration.md` - API integration best practices
   - `project-structure.md` - Project organization standards
   - `testing-and-troubleshooting.md` - Testing strategies
   - `output-requirements.md` - Output format requirements

## Usage

### Step 1: Analyze Your n8n Workflow

1. Export your n8n workflow as JSON
2. Use `Step1-ProductionRequirements.md` prompt with your n8n JSON
3. Review the generated technical specification
4. Identify any custom nodes that need separate analysis

### Step 2: Extract Custom Logic (If Applicable)

1. For each custom node identified in Step 1:
   - Extract the custom code (Python/Node.js)
   - Use `Step2-CustomLogic.md` prompt with the custom code
   - Save the requirements specification to `/req-for-custom-nodes/<node-name>.md`

### Step 3: Generate LangGraph Implementation

1. Use `Step3-MainOrchestorPrompt.md` with:
   - Production requirements from Step 1
   - Custom node requirements from Step 2 (if any)
2. Review the generated LangGraph implementation
3. Verify compliance with implementation guides
4. Test and iterate as needed

## Example Implementation

This repository includes a complete example conversion: **Automated Outbound Sales Email Campaign**

### Workflow Process
1. Reads company URLs from Google Sheets
2. Fetches and extracts website content
3. Generates company summaries using OpenAI (gpt-4o-mini)
4. Finds contact emails via Hunter.io
5. Generates personalized cold emails
6. Creates Gmail drafts
7. Logs successes and failures back to Google Sheets

### Implementation Details

**Paradigm**: Functional API with `@entrypoint` (Synchronous pattern)

**Justification**:
- Sequential workflow with simple if/else conditional branching
- No need for complex state management or parallel execution
- Standard Python control flow is adequate
- Matches "Simple to Moderate" complexity profile

### Running the Example

1. Set up environment variables (see `env_template.txt`)
2. Configure Google Cloud credentials (see `guides/authentication-setup.md`)
3. Run the workflow:
```bash
python src/workflow.py
```

For detailed setup instructions, see the example workflow's documentation in `src/workflow.py`.

## Project Structure

```
catlang/
├── config/                        # Configuration files
│   ├── server.yaml                # MCP server configuration
│   └── providers.yaml            # LLM provider configuration
├── docs/                          # Documentation
│   ├── MCP_SERVER_SETUP.md       # MCP server setup guide
│   ├── CLAUDE_DESKTOP_SETUP.md   # Claude Desktop integration guide
│   ├── USAGE_EXAMPLES.md         # Usage examples
│   └── TESTING.md                 # Testing documentation
├── guides/                        # Implementation guides (MCP resources)
│   ├── api-integration.md
│   ├── authentication-setup.md
│   ├── functional-api-implementation.md
│   ├── graph-api-implementation.md
│   ├── output-requirements.md
│   ├── paradigm-selection.md
│   ├── project-structure.md
│   └── testing-and-troubleshooting.md
├── src/                           # Source code
│   ├── mcp_server/                # MCP server implementation
│   │   ├── server.py              # Main server
│   │   ├── tools/                 # Tool implementations
│   │   └── resources/            # Resource handlers
│   ├── llm_provider/              # LLM provider abstraction
│   ├── workflow_engine/           # Workflow engine (future)
│   └── workflow.py               # Example: Sales email campaign
├── tests/                         # Test suite
│   ├── test_fase10_mcp_integration.py
│   └── ...                       # Other test files
├── scripts/                       # Utility scripts
│   └── verify_setup.py           # Setup verification
├── requirements.txt               # Python dependencies
├── run_mcp_server.py             # MCP server entry point
├── env_template.txt               # Environment variables template
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Key Concepts

### Paradigm Selection

Choose between two LangGraph paradigms based on workflow complexity:

- **Functional API (`@entrypoint`)**: For simple to moderate complexity workflows with sequential processing and simple conditionals
- **Graph API (`StateGraph`)**: For complex workflows requiring state management, parallel execution, or advanced control flow

See `guides/paradigm-selection.md` for detailed decision criteria.

### Execution Patterns

- **Synchronous**: Default for most workflows, simpler implementation
- **Asynchronous**: Use when concurrent API calls or parallel processing is needed

### Custom Nodes

When custom nodes are identified:
1. Extract custom code to separate files
2. Use Step 2 prompt to analyze
3. Reference requirements in Step 3 prompt
4. Implement as Python functions following LangGraph patterns

## Troubleshooting

### Common Issues

**Authentication Problems**
- Ensure credentials are properly configured
- Check OAuth scopes and permissions
- See `guides/authentication-setup.md` for detailed guidance

**API Integration Issues**
- Verify API keys and rate limits
- Check network connectivity and timeouts
- Review `guides/api-integration.md` for patterns

**Implementation Errors**
- Verify paradigm selection matches workflow complexity
- Ensure sync/async consistency throughout
- Cross-reference with implementation guides

For more detailed troubleshooting, see `guides/testing-and-troubleshooting.md`.

## Security Notes

- Never commit `.env`, `credentials.json`, or `token.json` to version control
- These files are automatically excluded via `.gitignore`
- Rotate API keys regularly
- Use minimal OAuth scopes required
- Review security best practices in implementation guides

## Documentation

- [MCP Server Setup Guide](docs/MCP_SERVER_SETUP.md) - Complete setup instructions for MCP server
- [Claude Desktop Integration](docs/CLAUDE_DESKTOP_SETUP.md) - Configure Claude Desktop to use CatLang
- [Usage Examples](docs/USAGE_EXAMPLES.md) - Examples of using all tools
- [Testing Guide](docs/TESTING.md) - Running tests and test coverage

## External Documentation References

- [LangGraph Functional API Overview](https://blog.langchain.com/introducing-the-langgraph-functional-api/)
- [LangGraph Functional API Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/functional_api.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [n8n Documentation](https://docs.n8n.io/)

## Contributing

When contributing to this framework:
1. Follow the orchestration process for any new conversions
2. Update guides if new patterns are discovered
3. Maintain consistency with existing implementation patterns
4. Document any deviations or alternative approaches

## License

MIT License - See [LICENSE](LICENSE) file for details.

Copyright (c) 2025 MADAILAB, Rohit Aggarwal, Hitesh Balegar
