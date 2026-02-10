# Agent Framework

Aratta includes a provider-agnostic agent framework built on the ReAct
(Reason-Act-Observe) pattern. Agents invoke any configured provider — local
or cloud — through SCRI. The agent doesn't know or care which provider handles
the request. Switch the model alias and the same agent uses a different backend.

## Quick Start

```python
from aratta.agents import Agent, AgentConfig, AgentContext

# Create a context with your AI client
ctx = AgentContext.create(aratta_client)

# Create and run an agent
agent = Agent(
    config=AgentConfig(model="local", max_iterations=10),
    context=ctx,
)
result = await agent.run("What's 2 + 2?")
print(result["content"])
```

## Components

### Agent

The main entry point. Holds configuration, message history, tools, and metrics.

```python
agent = Agent(
    agent_id="my-agent",           # optional, auto-generated if omitted
    config=AgentConfig(...),
    context=ctx,
    tools=[...],                   # optional, pulled from context if omitted
    system_prompt="...",           # optional, auto-generated with tool list
)
```

### AgentConfig

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `"local"` | Model alias or provider:model string |
| `temperature` | `0.7` | Sampling temperature |
| `max_iterations` | `10` | Max ReAct loop cycles |
| `timeout_seconds` | `300` | Total execution timeout |
| `enable_thinking` | `False` | Enable extended thinking |
| `allowed_tools` | `None` | Whitelist (None = all) |
| `blocked_tools` | `None` | Blacklist |
| `require_confirmation` | `[]` | Tools that pause for human approval |

### AgentContext

Scoped service access following least-privilege:

```python
# Full context with tool execution
ctx = AgentContext.create(client)

# Minimal context (no executor)
ctx = AgentContext.minimal(client)
```

### ReAct Loop

The loop cycles through:

1. **Reason** — send conversation to LLM, get response
2. **Act** — if LLM returns tool calls, execute them
3. **Observe** — feed tool results back as messages
4. **Repeat** until final answer or limits hit

Exit conditions: final answer, max iterations, timeout, error, or tool
requiring confirmation.

## Tool Execution

Tools run through the `ToolExecutor` with:

- **Permission checks** — restricted/standard/elevated/system levels
- **Timeout enforcement** — per-tool execution limits
- **Audit trail** — all executions logged

### Built-in Tools

| Tool | Permission | Description |
|------|-----------|-------------|
| `python_sandbox` | Elevated | AST-validated code execution |
| `execute_code` | Elevated | Alias for python_sandbox |
| `get_time` | Restricted | Current UTC time |
| `list_tools` | Restricted | List available tools |

### Code Sandbox

Python code execution with:

- AST validation before execution (blocked imports, calls, attributes)
- Subprocess isolation with restricted environment
- Output size limits (100KB)
- Configurable timeout
- Allowlisted imports only (math, json, re, collections, etc.)

### Custom Tools

```python
async def my_tool(args: dict):
    return {"result": args["query"]}

executor = ctx.tool_executor
executor.register_tool("my_tool", my_tool)
```

## Callbacks

```python
agent.on_complete(lambda result: print(f"Done: {result}"))
agent.on_tool_call(lambda name, args: print(f"Calling {name}"))
```

## Serialization

Agents can be serialized and restored:

```python
data = agent.to_dict()
restored = Agent.from_dict(data, context=ctx)
```
