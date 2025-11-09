# agent_mode_cli
Answering the question: What happens when you give the LLM bash?

## Overview

This repository contains Python command-line scripts for interacting with AI models:

- **`ai`** - A simple wrapper for local Ollama models
- **`am`** - A wrapper for OpenAI API

## Installation

1. Clone this repository or download the scripts
2. Copy the scripts to your `~/bin` directory:
   ```bash
   cp ai am ~/bin/
   chmod +x ~/bin/ai ~/bin/am
   ```
3. Ensure `~/bin` is in your PATH

## Usage

### `ai` - Ollama Wrapper

Sends prompts to a local Ollama installation.

**Prerequisites:**
- Ollama must be installed and running locally on port 11434
- Default model: `llama2` (can be modified in the script)

**Usage:**
```bash
ai What is the capital of France?
ai "Write a haiku about coding"
```

### `am` - OpenAI API Wrapper

Sends prompts to OpenAI's API.

**Prerequisites:**
- OpenAI API key (set as environment variable)

**Setup:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Usage:**
```bash
am What is the capital of France?
am "Write a haiku about coding"
```

## Configuration

Both scripts use default models:
- `ai`: Uses `llama2` model (Ollama)
- `am`: Uses `gpt-3.5-turbo` model (OpenAI)

To change the model, edit the respective script and modify the `model` parameter.

## Requirements

- Python 3.x (uses only standard library modules)
- For `ai`: Ollama installed and running
- For `am`: OpenAI API key
