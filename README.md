# agent_mode_cli
Answering the question: What happens when you give the LLM bash?

Inspired by [You Should Write an Agent](https://fly.io/blog/everyone-write-an-agent/) by Thomas H. Ptacek, @tqbf.


## Overview

This repository contains command-line scripts (located in `bin/`) for interacting with AI models:

- **`ai`** - Bash wrapper for prompting local Ollama models on the command line
- **`am`** - Python agent-mode CLI that uses OpenAI API and can call limited local tools (ping, sed, grep, git, curl) on your machine

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

**Usage:**
```bash
ai What is the capital of France?
ai "Write a haiku about coding"
```

### `am` - OpenAI API Wrapper

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


