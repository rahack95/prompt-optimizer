#!/usr/bin/env python3
"""
Test client for the MCP Prompt Optimizer Server
Use this to test your MCP server before integrating with Claude Desktop
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_prompt_optimizer():
    """Test the prompt optimizer MCP server"""
    
    # Path to your server.py file
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None
    )
    
    print("🚀 Starting MCP Prompt Optimizer test client...\n")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            print("✅ Connected to MCP server\n")
            
            # List available tools
            tools = await session.list_tools()
            print("📋 Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()
            
            # Test 1: Optimize a single prompt
            print("=" * 60)
            print("TEST 1: Optimize Single Prompt")
            print("=" * 60)
            
            test_prompt = "Fix the errors in the code"
            print(f"Input: {test_prompt}\n")
            
            result = await session.call_tool(
                "optimize_prompt",
                arguments={"prompt": test_prompt}
            )
            
            response = json.loads(result.content[0].text)
            print("📤 Optimized Prompt:")
            print("-" * 60)
            print(response["optimized_prompt"])
            print("-" * 60)
            print(f"\n📊 Metrics:")
            print(f"  Original Length: {response['metrics']['original_length']}")
            print(f"  Optimized Length: {response['metrics']['optimized_length']}")
            print(f"  Improvement: {response['metrics']['improvement_percentage']}%")
            print()
            
            # Test 2: Batch optimize prompts
            print("\n" + "=" * 60)
            print("TEST 2: Batch Optimize Prompts")
            print("=" * 60)
            
            test_prompts = [
                "Explain machine learning",
                "Write a REST API",
                "Debug my code"
            ]
            
            print("Input prompts:")
            for i, p in enumerate(test_prompts, 1):
                print(f"  {i}. {p}")
            print()
            
            batch_result = await session.call_tool(
                "batch_optimize_prompts",
                arguments={"prompts": test_prompts}
            )
            
            batch_response = json.loads(batch_result.content[0].text)
            print("📤 Batch Results:")
            for i, result in enumerate(batch_response["results"], 1):
                print(f"\n  {i}. Original: {result['original']}")
                print(f"     Status: {result['status']}")
                if result['status'] == 'success':
                    print(f"     Optimized: {result['optimized'][:100]}...")
            print()
            
            # Test 3: Get optimization tips
            print("\n" + "=" * 60)
            print("TEST 3: Get Optimization Tips")
            print("=" * 60)
            
            tips_result = await session.call_tool(
                "get_optimization_tips",
                arguments={}
            )
            
            tips_response = json.loads(tips_result.content[0].text)
            print("\n💡 Prompt Optimization Tips:")
            for tip in tips_response["prompt_optimization_tips"]:
                print(f"  • {tip}")
            
            print("\n✨ Common Improvements:")
            for improvement in tips_response["common_improvements"]:
                print(f"  • {improvement}")
            
            print("\n" + "=" * 60)
            print("✅ All tests completed successfully!")
            print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(test_prompt_optimizer())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()