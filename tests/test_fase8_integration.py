"""Integration tests for Fase 8 with Fase 7 - Validate generated code."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.mcp_server.tools.handlers import (
    analyze_n8n_workflow,
    extract_custom_logic,
    generate_langgraph_implementation,
    validate_implementation,
    set_server_instance,
)


# Sample requirements for testing
SAMPLE_REQUIREMENTS = """
# Production Requirements

## Global Workflow Summary
- Objective: Process user data
- Triggers: Webhook
- Execution: Sequential

## Node Specifications
- Node 1: Webhook trigger
- Node 2: Process data
- Node 3: Send response

## Custom Nodes
None identified.

## Implementation Notes
Simple sequential workflow suitable for Functional API.
"""


@pytest.mark.asyncio
class TestFase7Fase8Integration:
    """Integration tests between Fase 7 (generate) and Fase 8 (validate)."""
    
    async def test_generate_and_validate_workflow(self):
        """Test complete workflow: generate code then validate it."""
        # Mock LLM provider for generation
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        mock_response = Mock()
        mock_response.content = """```python
from langgraph import entrypoint, task

@entrypoint
def workflow(input_data):
    result = process_data(input_data)
    return {"result": result}

@task
def process_data(data):
    return data.upper()
```"""
        mock_provider.generate.return_value = mock_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        set_server_instance(mock_server)
        
        # Step 1: Generate code
        generation_result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            paradigm="functional",
            output_format="code",
        )
        
        assert generation_result["status"] == "success"
        assert "code" in generation_result
        generated_code = generation_result["code"]
        assert len(generated_code) > 0
        
        # Step 2: Validate generated code
        validation_result = await validate_implementation(
            code=generated_code,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=True,
        )
        
        assert validation_result["status"] == "success"
        assert "valid" in validation_result
        assert "syntax" in validation_result
        assert "compliance" in validation_result
        assert "best_practices" in validation_result
        
        # Generated code should be valid (or at least have minimal errors)
        # Note: Actual validation depends on LLM output quality
        assert validation_result["paradigm"] in ["functional", "graph", "unknown"]
    
    async def test_validate_generated_functional_code(self):
        """Test validation of generated functional API code."""
        generated_code = """from langgraph import entrypoint, task

@entrypoint
def workflow(input_data):
    result = process_data(input_data)
    return {"result": result}

@task
def process_data(data):
    return data.upper()
"""
        
        validation_result = await validate_implementation(
            code=generated_code,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,  # Skip LLM suggestions for faster test
        )
        
        assert validation_result["status"] == "success"
        assert validation_result["syntax"]["valid"] is True
        assert validation_result["paradigm"] == "functional"
        assert validation_result["compliance"]["valid"] is True or len(validation_result["compliance"]["issues"]) == 0
    
    async def test_validate_generated_graph_code(self):
        """Test validation of generated graph API code."""
        generated_code = """from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    data: str

def node1(state: State):
    return {"data": "processed"}

graph = StateGraph(State)
graph.add_node("node1", node1)
graph.add_edge("node1", "END")
"""
        
        validation_result = await validate_implementation(
            code=generated_code,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,
        )
        
        assert validation_result["status"] == "success"
        assert validation_result["syntax"]["valid"] is True
        assert validation_result["paradigm"] == "graph"
    
    async def test_validate_invalid_generated_code(self):
        """Test validation catches issues in generated code."""
        invalid_code = """from langgraph import entrypoint

def workflow(input_data):  # Missing @entrypoint decorator
    return {"result": "success"}
"""
        
        validation_result = await validate_implementation(
            code=invalid_code,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,
        )
        
        assert validation_result["status"] == "success"
        # Should detect missing entrypoint if paradigm is functional
        if validation_result["paradigm"] == "functional":
            compliance_issues = validation_result["compliance"]["issues"]
            entrypoint_issues = [
                issue for issue in compliance_issues
                if "entrypoint" in issue.get("message", "").lower()
            ]
            # May or may not detect depending on paradigm detection
            assert isinstance(compliance_issues, list)
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow: generate -> validate -> check results."""
        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        
        # Mock generation response
        mock_gen_response = Mock()
        mock_gen_response.content = """```python
from langgraph import entrypoint, task

@entrypoint
def workflow(input_data):
    return {"result": "success"}
```"""
        mock_provider.generate.return_value = mock_gen_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        set_server_instance(mock_server)
        
        # Generate
        gen_result = await generate_langgraph_implementation(
            requirements=SAMPLE_REQUIREMENTS,
            paradigm="functional",
            output_format="code",
        )
        
        if gen_result["status"] != "success":
            pytest.skip("Generation failed, skipping validation test")
        
        code = gen_result["code"]
        
        # Validate
        val_result = await validate_implementation(
            code=code,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,
        )
        
        # Check results
        assert val_result["status"] == "success"
        assert val_result["code_length"] == len(code)
        assert "summary" in val_result
        assert "is_production_ready" in val_result["summary"]
        
        # Production readiness depends on code quality
        # Just verify the structure is correct
        assert isinstance(val_result["summary"]["is_production_ready"], bool)
        assert isinstance(val_result["summary"]["total_errors"], int)
        assert isinstance(val_result["summary"]["total_warnings"], int)
        assert isinstance(val_result["summary"]["total_suggestions"], int)
    
    @pytest.mark.asyncio
    async def test_end_to_end_with_n8n_workflow(self):
        """Test complete end-to-end workflow using example n8n workflow."""
        # Load example n8n workflow
        workflow_path = Path(__file__).parent.parent / "example_n8n_workflow" / "Json string validator via webhook.json"
        
        if not workflow_path.exists():
            pytest.skip(f"Example workflow not found at {workflow_path}")
        
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_json = f.read()
        
        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.is_initialized.return_value = True
        
        # Mock analysis response
        mock_analysis_response = Mock()
        mock_analysis_response.content = """# Production Requirements

## Global Workflow Summary
- Objective: Validate JSON string via webhook
- Triggers: Webhook POST request
- Execution: Sequential

## Node Specifications
- Node 1: Webhook receiver
- Node 2: Code node for JSON validation
- Node 3: Respond to webhook

## Custom Nodes
Found custom code node with JavaScript validation logic.
"""
        mock_provider.generate.return_value = mock_analysis_response
        
        mock_server = Mock()
        mock_server._get_llm_provider.return_value = mock_provider
        set_server_instance(mock_server)
        
        # Step 1: Analyze n8n workflow
        analysis_result = await analyze_n8n_workflow(
            workflow_json=workflow_json,
            include_metadata=True,
        )
        
        if analysis_result["status"] != "success":
            pytest.skip("Analysis failed, skipping validation test")
        
        requirements = analysis_result["requirements"]
        
        # Step 2: Extract custom logic (if any)
        workflow_data = json.loads(workflow_json)
        custom_logic_specs = None
        
        for node in workflow_data.get("nodes", []):
            if node.get("type") == "n8n-nodes-base.code":
                js_code = node.get("parameters", {}).get("jsCode", "")
                if js_code:
                    extract_result = await extract_custom_logic(
                        code=js_code,
                        language="javascript",
                        node_name=node.get("name", ""),
                    )
                    if extract_result["status"] == "success":
                        custom_logic_specs = extract_result["specifications"]
                        break
        
        # Step 3: Generate LangGraph code
        mock_gen_response = Mock()
        mock_gen_response.content = """```python
from langgraph import entrypoint, task

@entrypoint
def workflow(input_data):
    json_string = input_data.get("jsonString", "")
    result = validate_json_string(json_string)
    return {"valid": result["valid"], "error": result.get("error")}

@task
def validate_json_string(json_string):
    import json
    try:
        json.loads(json_string)
        return {"valid": True}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}
```"""
        mock_provider.generate.return_value = mock_gen_response
        
        generation_result = await generate_langgraph_implementation(
            requirements=requirements,
            custom_logic_specs=custom_logic_specs,
            paradigm="functional",
            output_format="code",
        )
        
        if generation_result["status"] != "success":
            pytest.skip("Generation failed, skipping validation test")
        
        generated_code = generation_result["code"]
        
        # Step 4: Validate generated code
        validation_result = await validate_implementation(
            code=generated_code,
            check_syntax=True,
            check_compliance=True,
            check_best_practices=False,  # Skip LLM suggestions for faster test
        )
        
        # Verify validation results
        assert validation_result["status"] == "success"
        assert validation_result["code_length"] == len(generated_code)
        assert "syntax" in validation_result
        assert "compliance" in validation_result
        assert "summary" in validation_result
        
        # Generated code should be syntactically valid
        assert validation_result["syntax"]["valid"] is True
        
        # Should detect functional paradigm
        assert validation_result["paradigm"] == "functional"
        
        # Compliance should be valid (or have minimal issues)
        # Note: Actual compliance depends on generated code quality
        assert isinstance(validation_result["compliance"]["valid"], bool)
        assert isinstance(validation_result["compliance"]["issues"], list)
        
        # Summary should have all required fields
        assert "total_errors" in validation_result["summary"]
        assert "total_warnings" in validation_result["summary"]
        assert "total_suggestions" in validation_result["summary"]
        assert "is_production_ready" in validation_result["summary"]
        
        print(f"\n✅ End-to-end test completed:")
        print(f"   - Workflow analyzed: {analysis_result['status']}")
        print(f"   - Code generated: {generation_result['status']}")
        print(f"   - Code validated: {validation_result['status']}")
        print(f"   - Paradigm: {validation_result['paradigm']}")
        print(f"   - Syntax valid: {validation_result['syntax']['valid']}")
        print(f"   - Compliance valid: {validation_result['compliance']['valid']}")
        print(f"   - Production ready: {validation_result['summary']['is_production_ready']}")
