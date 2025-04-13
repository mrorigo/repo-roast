# Repo Roast 🔍

A powerful tool that analyzes your Git repository and provides intelligent feedback using Large Language Models. Get comprehensive reports on maintainability, production readiness, test coverage, and more.

## Overview

Repo Roast combines the repository scanning capabilities of [repomix](https://github.com/natemoo-re/repomix) with the analytical power of Large Language Models to deliver insightful code analysis reports. It can help you identify:

- Code smells and refactoring opportunities
- Production readiness gaps
- Test coverage needs
- And more, based on custom prompts

## Features

- 🔍 **Deep Repository Analysis** - Scans your entire repository structure
- 🤖 **LLM-Powered Insights** - Leverages AI to provide meaningful feedback
- 🔧 **Multiple Report Types** - Choose from various analysis templates
- ⚙️ **Customizable** - Works with different LLM providers and models
- 💰 **Token Optimization** - Options to reduce token usage and costs

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm (for repomix)
- [UV](https://github.com/astral-sh/uv) - A fast, reliable Python package installer and resolver (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/repo-roast.git
cd repo-roast
```

2. We recommend using [UV](https://github.com/astral-sh/uv), a fast Python package installer and resolver:

```bash
# Install UV if you don't have it already
pip install uv

# Create a virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

3. Create a `.env` file from the example:

```bash
cp .env.example .env
```

4. Edit the `.env` file with your LLM API credentials:

```
OPENAI_API_BASE=https://api.openai.com/v1  # or your alternative API endpoint
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```bash
# Activate your virtual environment first, or use uv run:
uv run repo_roast.py --repo /path/to/repository --type maintainability-refactoring-plan --output report.md
```

### Available Roast Types

- `maintainability-refactoring-plan` - Identifies code smells and suggests refactoring opportunities
- `production-readyness-report` - Assesses whether the codebase is ready for production
- `test-coverage-plan` - Creates a plan to achieve desired test coverage

### Command Line Options

```
usage: repo_roast.py [-h] [--repo REPO] --type {maintainability-refactoring-plan,production-readyness-report,test-coverage-plan} --output OUTPUT [--model MODEL] [--optimize {compact,aggressive}] [--yes]

Roast a Git repository using repomix and an LLM.

options:
  -h, --help            show this help message and exit
  --repo REPO           Path to the target Git repository (default: current directory).
  --type {maintainability-refactoring-plan,production-readyness-report,test-coverage-plan}
                        The type of roast to perform.
  --output OUTPUT       Path to the output file for the LLM response.
  --model MODEL         The LLM model to use (default: gemma3:4b).
  --optimize {compact,aggressive}
                        Apply a preset of repomix arguments for token optimization.
  --yes                 Bypass confirmation prompt and proceed automatically.

Arguments after '--' are passed directly to the npx repomix command.
```

### Examples

#### Analyze Maintainability of Current Directory

```bash
uv run repo_roast.py --type maintainability-refactoring-plan --output maintainability-report.md
```

#### Analyze Production Readiness with Token Optimization

```bash
uv run repo_roast.py --repo /path/to/my-app --type production-readyness-report --output production-ready.md --optimize compact
```

#### Create a Test Coverage Plan Using a Specific Model

```bash
uv run repo_roast.py --type test-coverage-plan --output test-plan.md --model gpt-4
```

#### Pass Additional Arguments to repomix

```bash
uv run repo_roast.py --type maintainability-refactoring-plan --output report.md -- --exclude "node_modules/**" --exclude "build/**"
```

## Token Optimization

To reduce token usage and potential costs:

- `--optimize compact` applies minimal optimizations (removes comments and empty lines)
- `--optimize aggressive` applies maximum optimizations (compact plus compression and simplified structure)

## Environment Variables

- `OPENAI_API_KEY` - Your API key for OpenAI or compatible service
- `OPENAI_API_BASE` - The base URL for the API (defaults to OpenAI's API endpoint if not specified)

## Custom Prompts

You can create custom roast types by adding new markdown files to the `prompts/` directory. The filename (without extension) will become the available roast type.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
