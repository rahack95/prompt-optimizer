#!/usr/bin/env python3
"""
MCP Server for DSPy Prompt Optimization
Provides prompt optimization capabilities through the Model Context Protocol
"""

import asyncio
import json
from typing import Any
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import DSPy
try:
    import dspy
except ImportError:
    print("Error: dspy-ai not installed. Run: pip install dspy-ai")
    exit(1)

# Initialize DSPy
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables")
    exit(1)

lm = dspy.LM("gemini/gemini-2.0-flash-exp", api_key=api_key)
dspy.configure(lm=lm)

# Define DSPy Components
class PromptOptimizerSignature(dspy.Signature):
    """Improve a raw user prompt to get higher-quality LLM responses."""
    
    raw_prompt: str = dspy.InputField(
        desc="Original user prompt from the user"
    )
    optimized_prompt: str = dspy.OutputField(
        desc="Clear, structured, and effective prompt with role definition, clear instructions, and expected output format"
    )

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PromptOptimizerSignature)

    def forward(self, raw_prompt: str):
        return self.predict(raw_prompt=raw_prompt)

def prompt_quality_metric(example, prediction, trace=None):
    """Metric to evaluate prompt quality"""
    text = prediction.optimized_prompt.lower()
    score = 0

    if "you are" in text or "act as" in text:
        score += 1
    if "step" in text or "bullet" in text or "format" in text:
        score += 1
    if "output" in text or "provide" in text:
        score += 1
    if len(text) > len(example.raw_prompt) * 1.5:
        score += 1

    return score

# Training set
trainset = [
    dspy.Example(
        raw_prompt="Explain LLMs"
    ).with_inputs("raw_prompt"),

    dspy.Example(
        raw_prompt="Write python code for an API"
    ).with_inputs("raw_prompt"),

    dspy.Example(
        raw_prompt="Tell me about transformers"
    ).with_inputs("raw_prompt"),
    
    dspy.Example(
        raw_prompt="Fix the errors in the code"
    ).with_inputs("raw_prompt"),
]

# Compile optimized program
print("Initializing DSPy optimizer...")
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=prompt_quality_metric,
    max_bootstrapped_demos=3
)

optimized_program = optimizer.compile(
    PromptOptimizer(),
    trainset=trainset
)
print("DSPy optimizer ready!")

# Create MCP server instance
server = Server("prompt-optimizer")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="optimize_prompt",
            description="Optimize a raw user prompt to get higher-quality LLM responses. Transforms basic prompts into structured, effective prompts with clear role definitions, step-by-step instructions, and output format specifications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The raw prompt to optimize"
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="batch_optimize_prompts",
            description="Optimize multiple prompts at once. Useful for processing several prompts efficiently.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of raw prompts to optimize"
                    }
                },
                "required": ["prompts"]
            }
        ),
        types.Tool(
            name="get_optimization_tips",
            description="Get tips and best practices for writing effective prompts",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name == "optimize_prompt":
        if not arguments or "prompt" not in arguments:
            raise ValueError("Missing required argument: prompt")
        
        raw_prompt = arguments["prompt"]
        
        try:
            result = optimized_program(raw_prompt=raw_prompt)
            optimized = result.optimized_prompt
            
            # Calculate metrics
            original_length = len(raw_prompt)
            optimized_length = len(optimized)
            improvement = ((optimized_length - original_length) / original_length * 100)
            
            response = {
                "original_prompt": raw_prompt,
                "optimized_prompt": optimized,
                "metrics": {
                    "original_length": original_length,
                    "optimized_length": optimized_length,
                    "improvement_percentage": round(improvement, 2)
                }
            }
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(response, indent=2)
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error optimizing prompt: {str(e)}"
                )
            ]
    
    elif name == "batch_optimize_prompts":
        if not arguments or "prompts" not in arguments:
            raise ValueError("Missing required argument: prompts")
        
        prompts = arguments["prompts"]
        results = []
        
        for raw_prompt in prompts:
            try:
                result = optimized_program(raw_prompt=raw_prompt)
                optimized = result.optimized_prompt
                results.append({
                    "original": raw_prompt,
                    "optimized": optimized,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "original": raw_prompt,
                    "optimized": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps({"results": results}, indent=2)
            )
        ]
    
    elif name == "get_optimization_tips":
        tips = {
            "prompt_optimization_tips": [
                "Define a clear role for the AI (e.g., 'You are an expert...')",
                "Break down complex tasks into step-by-step instructions",
                "Specify the expected output format (JSON, markdown, bullet points, etc.)",
                "Include relevant context and constraints",
                "Use examples to illustrate what you want",
                "Be specific about tone and style preferences",
                "Ask for reasoning or explanation when needed",
                "Use structured formats for better consistency"
            ],
            "common_improvements": [
                "Adding role definitions",
                "Breaking instructions into numbered steps",
                "Specifying output structure",
                "Including error handling requirements",
                "Adding context and background information"
            ]
        }
        
        return [
            types.TextContent(
                type="text",
                text=json.dumps(tips, indent=2)
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main entry point for the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="prompt-optimizer",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())