# agent_mode_cli
Answering the question: What happens when you give the LLM bash?

Inspired by [You Should Write an Agent](https://fly.io/blog/everyone-write-an-agent/) by Thomas H. Ptacek, @tqbf.


## Overview

This repository contains command-line scripts (located in `bin/`) for interacting with AI models:

- **`ai`** - Bash wrapper for prompting local Ollama models on the command line
- **`am`** - Python agent-mode CLI that uses OpenAI API and can call limited local tools (ping, sed, grep, git, curl) on your machine
- **`om`** - Python agent-mode CLI modeled after `am`, but backed by local Ollama models (no OpenAI API key required)

## Installation

1. Clone this repository or download the scripts from the `bin/` directory
2. Copy the scripts to your `~/bin` directory:
   ```bash
   cp bin/ai ~/bin/
   cp bin/am ~/bin/
   chmod +x ~/bin/ai ~/bin/am
   ```
3. Ensure `~/bin` is in your PATH

## Usage

### `ai` - Ollama Wrapper

Sends prompts to a local Ollama installation.

**Prerequisites:**
- Ollama must be installed
- glow must be installed
- assumes tput and tee are available in your environment
- Default model: `gpt-oss` (can be modified in the script)

`ai` automatically detects NVIDIA GPUs (including RTX 4090) when `nvidia-smi` is available, sets `OLLAMA_USE_GPU=1`/`OLLAMA_GPU_TYPE=cuda` by default, and ensures a local `ollama serve` is running before streaming your prompt.

**Usage:**
```bash
ai What is the capital of France?
ai "Write a haiku about coding"
```

### `am` - OpenAI API Agent Mode Wrapper

Sends prompts to OpenAI's API, with the ability to execute bash commands on the **local machine**.  

**Danger:** This script can execute arbitrary bash commands on your local machine. It's best to use this in a walled off container or VM to avoid security risks.

**Prerequisites:**
- OpenAI API key (set as environment variable)
- Python package: `openai` (install with: `pip install -U openai`)

**Setup:**
```bash
export OPENAI_API_KEY='your-api-key-here'
# Optional:
export AM_MODEL='gpt-4o-mini'    # default used by `am` if not set
export AM_DEBUG=1                # enable verbose debug logs (use 0 to disable)
```

**Usage:**
```bash
am What is the capital of France?
am "Write a haiku about coding"
```



### `om` - Ollama Agent Mode Wrapper

Provides the same agent-mode experience as `am`, but routes all model calls through your local Ollama runtime. Tool calls (`bash`, `set_model`, `set_debug`) work the same way, except they are handled entirely on your machine.

**Prerequisites:**
- [Ollama](https://github.com/ollama/ollama) installed and running locally (`ollama serve`)
- Python package: `ollama` (installed automatically via `pip install -r requirements.txt`)
- Optional environment variables:
  - `OM_MODEL` – overrides the default model (`gpt-oss`)
  - `OM_DEBUG` – enable verbose logs (same semantics as `AM_DEBUG`)
  - `OM_PROMPT_FILE` – path to an additional system prompt file (defaults to `~/.om_prompt`)
  - `OLLAMA_HOST` – point to a remote Ollama instance if desired

`om` now shares the same detection logic: if a local NVIDIA GPU (especially an RTX 4090) is present it enables Ollama's CUDA backend, and it will launch `ollama serve` automatically when a local daemon isn't already running.

**Usage:**
```bash
om What is the capital of France?
om "List files in the current directory"
```

The agent will invoke local tools as needed. When it wants to run `bash`, you'll be asked to confirm unless you enable approve-all mode for that prompt.
