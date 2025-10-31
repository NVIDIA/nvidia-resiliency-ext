# NVRX Attribution MCP Integration - Architecture Guide

## Overview

This document describes the architecture and design decisions behind the MCP integration for NVIDIA Resiliency Extension (NVRX) Attribution modules.

## Design Goals

1. **Modularity**: Each attribution module operates independently
2. **Composability**: Modules can be chained into pipelines
3. **Scalability**: Support distributed execution across multiple servers
4. **Flexibility**: Work with both programmatic and AI assistant interfaces
5. **Compatibility**: Maintain backward compatibility with existing NVRX API

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  • Python Scripts  • AI Assistants  • Monitoring Systems    │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol
┌────────────────────────┴────────────────────────────────────┐
│                    MCP Integration Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ MCP Server   │  │ MCP Client   │  │  Registry    │      │
│  │  • Tools     │  │  • Calls     │  │  • Modules   │      │
│  │  • Resources │  │  • Pipeline  │  │  • Metadata  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ Base Attribution API
┌────────────────────────┴────────────────────────────────────┐
│              NVRX Attribution Base Layer                     │
│  ┌──────────────────────────────────────────────────┐       │
│  │           NVRxAttribution Base Class             │       │
│  │  • preprocess_input • attribution • output       │       │
│  │  • Async execution  • Pipeline support           │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │ Implementation
┌────────────────────────┴────────────────────────────────────┐
│              Attribution Module Implementations              │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐    │
│  │ FR Analyzer   │  │ Application   │  │ Component    │    │
│  │ (Collective)  │  │ Log Analyzer  │  │ Health       │    │
│  │               │  │ (possible)    │  │ Analyzer     │    │
│  │               │  │               │  │ (possible)   │    │
│  └───────────────┘  └───────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Registry (`registry.py`)

**Purpose**: Central registry for attribution modules with metadata and dependency tracking.

**Key Features**:
- Module registration with input/output schemas
- Dependency graph management
- Topological sorting for execution order
- Result caching
- Serialization/deserialization

**Design Pattern**: Singleton pattern with global registry instance

```python
# Register a module
global_registry.register(
    name="log_analyzer",
    module_class=NVRxLogAnalyzer,
    description="...",
    input_schema={...},
    output_schema={...},
    dependencies=[]
)

# Get execution order
order = global_registry.get_execution_order(["log_analyzer", "fr_analyzer", "combined"])
# Returns: ["log_analyzer", "fr_analyzer", "combined"] (respecting dependencies)
```

### 2. MCP Server (`mcp_server.py`)

**Purpose**: Expose attribution modules as MCP tools and resources.

**Key Features**:
- Auto-generate MCP tools from registered modules
- Pipeline execution support
- Result caching with resource URIs
- Status and introspection endpoints

**Communication**:
- Input: MCP protocol over stdio
- Output: JSON-serialized results

**Tool Types**:
1. **Module tools**: One per registered module (`log_analyzer`, `fr_analyzer`, etc.)
2. **Pipeline tool**: `run_pipeline` for multi-module execution
3. **Utility tools**: `status`, `get_result`

**Resource Pattern**:
```
attribution://<module_name>/<result_id>
Example: attribution://log_analyzer/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

### 3. MCP Client (`mcp_client.py`)

**Purpose**: Programmatic interface for calling MCP servers.

**Two Client Types**:

#### NVRxMCPClient
- Connects to a single MCP server
- Async context manager pattern
- Methods: `run_module()`, `run_pipeline()`, `get_result()`

#### MultiServerClient
- Manages multiple MCP servers
- Automatic module routing
- Cross-server pipeline orchestration

**Usage Pattern**:
```python
async with NVRxMCPClient(server_command) as client:
    result = await client.run_module("log_analyzer", input_data=...)
```

### 4. Module Definitions (`module_definitions.py`)

**Purpose**: Register all built-in NVRX attribution modules.

**Module Metadata Includes**:
- Input/output JSON schemas (for MCP)
- LLM requirements
- Dependencies on other modules
- Default configuration values

## Data Flow

### Single Module Execution

```
1. Client calls tool "log_analyzer" with input_data
       ↓
2. MCP Server receives call
       ↓
3. Server looks up module in registry
       ↓
4. Server creates/retrieves module instance
       ↓
5. Module executes: preprocess → attribution → output
       ↓
6. Server caches result with UUID
       ↓
7. Server returns: {result, result_id, resource_uri, state}
       ↓
8. Client receives response
```

## State Management

### AttributionState Enum

```python
class AttributionState(Enum):
    CONTINUE = auto()  # Continue to next module in pipeline
    STOP = auto()      # Stop pipeline execution
```

**Usage**:
- Output handlers return `(result, state)`
- Pipeline checks state after each module
- Allows modules to signal critical conditions

### Result Caching

**Cache Key Format**: `module_name:result_id`

**Cache Locations**:
1. **In-memory**: Registry maintains cache during server lifetime
2. **Resources**: Accessible via MCP resource protocol
3. **Optional persistent**: Can be extended with disk/database backend

## Serialization Strategy

### Challenge
Attribution results are arbitrary Python objects (dataclasses, custom types, etc.)

### Solution: Multi-strategy serialization

```python
def serialize_result(result):
    if is_dataclass(result):
        return json.dumps(asdict(result))
    elif hasattr(result, 'to_dict'):
        return json.dumps(result.to_dict())
    elif hasattr(result, '__dict__'):
        return json.dumps(result.__dict__)
    else:
        return json.dumps(str(result))
```

## Extension Points

```python
class MyAnalyzer(NVRxAttribution):
    def __init__(self, args):
        super().__init__(
            preprocess_input=self.preprocess,
            attribution=self.analyze,
            output_handler=self.output
        )
    
    # Implement your methods...

# Register it
global_registry.register(
    name="my_analyzer",
    module_class=MyAnalyzer,
    description="Custom analyzer",
    input_schema={...},
    output_schema={...}
)
```