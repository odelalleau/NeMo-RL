"""Test script for tool use environment."""

import json
import tempfile
import time

from nemo_rl.environments.tool_use_environment import ToolUseEnv, ToolUseLogic


def test_tool_use_environment():
    """Test the tool use environment functionality."""
    print("Testing Tool Use Environment")
    print("=" * 50)
    
    # Test configuration
    config = {
        "max_turns": 5,
        "available_tools": ["web_search", "code_execution"]
    }
    
    # Generate environment
    env_state = ToolUseLogic.generate(config)
    print(f"Generated environment: {env_state}")
    
    # Initialize environment
    init_message = ToolUseLogic.init(env_state)
    print(f"\nInitialization message:\n{init_message}")
    
    # Test cases with JSON format
    test_cases = [
        # Code execution test
        '<tool_call>{"name": "code_execution", "arguments": {"code": "print(2 + 3 * 4)"}}</tool_call>',
        # Code execution with math
        '<tool_call>{"name": "code_execution", "arguments": {"code": "import math; print(math.sqrt(16) + math.sin(math.pi/2))"}}</tool_call>',
        # Web search test
        '<tool_call>{"name": "web_search", "arguments": {"query": "artificial intelligence"}}</tool_call>',
        # Web search test 2
        '<tool_call>{"name": "web_search", "arguments": {"query": "python programming"}}</tool_call>',
        # Invalid tool test
        '<tool_call>{"name": "invalid_tool", "arguments": {}}</tool_call>',
        # Invalid format test
        'This is not a valid tool call',
    ]
    
    print(f"\nTesting {len(test_cases)} tool calls:")
    print("-" * 40)
    
    for i, action in enumerate(test_cases, 1):
        print(f"\nTest {i}: {action}")
        
        response, reward, terminated, new_env_state = ToolUseLogic.step(action, env_state)
        
        print(f"Response: {response}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Turn count: {new_env_state['turn_count']}/{new_env_state['max_turns']}")
        
        env_state = new_env_state
        
        if terminated:
            print("Environment terminated!")
            break
    
    print(f"\nTest completed!")


def test_ray_environment():
    """Test the Ray-based environment."""
    print("\n" + "=" * 50)
    print("Testing Ray Environment")
    print("=" * 50)
    
    try:
        import ray
        
        if not ray.is_initialized():
            ray.init(local_mode=True)
        
        # Create Ray environment
        env_config = {
            "max_turns": 3,
            "available_tools": ["web_search", "code_execution"]
        }
        
        env = ToolUseEnv.remote(env_config)
        
        # Simulate a conversation
        message_logs = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Calculate 5 + 7 * 2"},
                {"role": "assistant", "content": 'Let me calculate that for you. <tool_call>{"name": "code_execution", "arguments": {"code": "print(5 + 7 * 2)"}}</tool_call>'}
            ]
        ]
        
        metadata_batch = [
            {
                "workspace": {},
                "turn_count": 0,
                "max_turns": 3,
                "available_tools": ["web_search", "code_execution"],
                "task_completed": False
            }
        ]
        
        # Step the environment
        result = ray.get(env.step.remote(message_logs, metadata_batch))
        
        print(f"Environment result: {result}")
        print(f"Observations: {result.observations}")
        print(f"Rewards: {result.rewards}")
        print(f"Terminated: {result.terminateds}")
        
        # Clean up
        ray.get(env.shutdown.remote())
        
        print("Ray environment test completed successfully!")
        
    except ImportError:
        print("Ray not available, skipping Ray environment test")
    except Exception as e:
        print(f"Ray environment test failed: {e}")


def test_json_parsing():
    """Test JSON parsing functionality."""
    print("\n" + "=" * 50)
    print("Testing JSON Parsing")
    print("=" * 50)
    
    test_cases = [
        # Valid cases
        '<tool_call>{"name": "code_execution", "arguments": {"code": "print(2+2)"}}</tool_call>',
        '<tool_call>\n{\n  "name": "web_search",\n  "arguments": {\n    "query": "test"\n  }\n}\n</tool_call>',
        '<tool_call>{"name": "code_execution", "arguments": {}}</tool_call>',
        
        # Invalid cases
        '<tool_call>code_execution("print(2+2)")</tool_call>',  # Old format
        '<tool_call>{"name": "exec"}</tool_call>',  # Missing arguments
        '<tool_call>invalid json</tool_call>',  # Invalid JSON
        'No tool call here',  # No tool call tags
        '<tool_call>{"name": 123}</tool_call>',  # Invalid name type
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case[:50]}...")
        result = ToolUseLogic._parse_tool_call(test_case)
        print(f"Result: {result}")


def test_duckduckgo_search():
    """Test DuckDuckGo search functionality."""
    print("\n" + "=" * 50)
    print("Testing DuckDuckGo Search")
    print("=" * 50)
    
    from nemo_rl.environments.tool_use_environment import ToolRegistry
    
    # Test search
    query = "python programming"
    print(f"Searching for: {query}")
    
    result, success = ToolRegistry.web_search(query)
    print(f"Success: {success}")
    print(f"Result:\n{result}")


if __name__ == "__main__":
    test_json_parsing()
    test_duckduckgo_search()
    test_tool_use_environment()
    test_ray_environment() 