import argparse
import os
import sys
import subprocess
import openai # Import the library
from openai import OpenAI # Explicitly import the client
from dotenv import load_dotenv
import tiktoken # Import tiktoken

PROMPTS_DIR = "prompts"
REPOMIX_OUTPUT_FILE = "repomix-output.md" # Default output file from repomix
DEFAULT_MODEL = "gemma3:4b"
TOKEN_WARNING_THRESHOLD = 10000

# Define repomix optimization presets
REPOMIX_OPTIMIZE_PRESETS = {
    "compact": ["--remove-comments", "--remove-empty-lines"],
    "aggressive": [
        "--remove-comments",
        "--remove-empty-lines",
        "--compress",
        "--no-directory-structure",
        "--no-file-summary"
    ],
}

def estimate_token_count(text: str, encoding_name: str) -> int:
    """Estimates the number of tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        print(f"Estimated token count: {num_tokens}")  # Always print the token count
        return num_tokens
    except ValueError:
        print(f"Warning: Could not load tokenizer for encoding '{encoding_name}'. Token count estimation skipped.", file=sys.stderr)
        return -1 # Indicate failure to estimate

def discover_roast_types(prompts_dir: str) -> list[str]:
    """Scans the prompts directory for available roast types (.md files)."""
    if not os.path.isdir(prompts_dir):
        print(f"Error: Prompts directory '{prompts_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    types = []
    try:
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".md"):
                types.append(os.path.splitext(filename)[0])
    except OSError as e:
        print(f"Error scanning prompts directory '{prompts_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    if not types:
        print(f"Error: No prompt files (.md) found in '{prompts_dir}'.", file=sys.stderr)
        sys.exit(1)

    return sorted(types)

def run_repomix(repo_path: str, passthrough_args: list[str], optimize_preset: str | None = None, instruction_file_path: str | None = None):
    """Runs the npx repomix command with optional optimization presets and instruction file."""
    base_command = ["npx", "repomix", "--style", "markdown"]

    preset_args = []
    if optimize_preset:
        preset_args = REPOMIX_OPTIMIZE_PRESETS.get(optimize_preset, [])
        if not preset_args and optimize_preset in REPOMIX_OPTIMIZE_PRESETS: # Check if preset exists but is empty
             print(f"Warning: Optimization preset '{optimize_preset}' is defined but has no arguments.", file=sys.stderr)
        elif optimize_preset not in REPOMIX_OPTIMIZE_PRESETS:
             # This shouldn't happen if argparse choices are set correctly, but as a safeguard:
             print(f"Warning: Unknown optimization preset '{optimize_preset}'. Ignoring.", file=sys.stderr)
             optimize_preset = None # Ensure we don't report it later
    instruction_args = []
    if instruction_file_path:
        # Basic check if file exists before passing to repomix
        if not os.path.exists(instruction_file_path):
             print(f"Warning: Instruction file '{instruction_file_path}' not found. Proceeding without it.", file=sys.stderr)
        else:
             print(f"Using instruction file: {instruction_file_path}")
             instruction_args = ["--instruction-file-path", instruction_file_path]

    # Order: base command, preset args, instruction args, repo path, passthrough args
    command = base_command + preset_args + instruction_args + [repo_path] + passthrough_args

    print(f"Running repomix for repository: {repo_path}" + (f" (Optimize preset: {optimize_preset})" if optimize_preset else ""))
    print(f"Command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')

        if result.returncode != 0:
            print(f"Error: repomix failed with exit code {result.returncode}", file=sys.stderr)
            print("--- repomix stdout ---", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print("--- repomix stderr ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        else:
            print("repomix completed successfully.")

        if not os.path.exists(REPOMIX_OUTPUT_FILE):
             print(f"Error: repomix ran but the expected output file '{REPOMIX_OUTPUT_FILE}' was not found.", file=sys.stderr)
             print("This might indicate an issue with repomix or file permissions.", file=sys.stderr)
             print("--- repomix stdout ---", file=sys.stderr)
             print(result.stdout, file=sys.stderr)
             print("--- repomix stderr ---", file=sys.stderr)
             print(result.stderr, file=sys.stderr)
             sys.exit(1)

    except FileNotFoundError:
        print("Error: 'npx' command not found. Make sure Node.js and npm/npx are installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running repomix: {e}", file=sys.stderr)
        sys.exit(1)

def read_file(file_path: str) -> str:
    """Reads the content of a file."""
    print(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Successfully read {len(content)} characters from {file_path}")
        return content
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def call_llm(api_key: str, api_base: str | None, prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Calls the LLM API with the given prompt."""
    print(f"Calling LLM API (Model: {model})...")
    try:
        client = OpenAI(api_key=api_key, base_url=api_base) # base_url can be None

        response = client.chat.completions.create(
            model=model,
            messages=[
                # Optional: Add a system prompt if desired later
                # {"role": "system", "content": "You are a helpful code analysis assistant."},
                {"role": "user", "content": prompt}
            ]
            # Optional: Add temperature, max_tokens etc. if needed
            # temperature=0.7,
            # max_tokens=4096
        )

        # Basic check for response structure
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
             print("Error: LLM API response structure is unexpected.", file=sys.stderr)
             print(f"Full response: {response}", file=sys.stderr)
             sys.exit(1)

        result_text = response.choices[0].message.content
        print("LLM API call successful.")
        return result_text

    except openai.AuthenticationError:
        print("Error: LLM API Authentication failed. Check your API key.", file=sys.stderr)
        sys.exit(1)
    except openai.RateLimitError:
         print("Error: LLM API rate limit exceeded. Please wait and try again.", file=sys.stderr)
         sys.exit(1)
    except openai.APIConnectionError as e:
        print(f"Error: Could not connect to LLM API: {e}", file=sys.stderr)
        sys.exit(1)
    except openai.APIError as e:
        print(f"Error: An LLM API error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during the LLM API call: {e}", file=sys.stderr)
        sys.exit(1)

def write_output(file_path: str, content: str):
    """Writes the content to the specified file."""
    print(f"Writing output to file: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Output file written successfully.")
    except IOError as e:
        print(f"Error writing output file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # Load environment variables from .env file if it exists
    load_dotenv()

    available_roast_types = discover_roast_types(PROMPTS_DIR)

    parser = argparse.ArgumentParser(
        description="Roast a Git repository using repomix and an LLM.",
        epilog="Arguments after '--' are passed directly to the npx repomix command."
    )
    parser.add_argument("--repo", default=".", help="Path to the target Git repository (default: current directory).")
    parser.add_argument("--type", required=True, choices=available_roast_types, help="The type of roast to perform.")
    parser.add_argument("--output", required=True, help="Path to the output file for the LLM response.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"The LLM model to use (default: {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--optimize",
        choices=list(REPOMIX_OPTIMIZE_PRESETS.keys()),
        default=None, # No preset by default
        help="Apply a preset of repomix arguments for token optimization."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass confirmation prompt and proceed automatically."
    )

    args, repomix_passthrough_args = parser.parse_known_args()

    if repomix_passthrough_args and repomix_passthrough_args[0] != '--':
         parser.error("Arguments intended for repomix must be preceded by '--'. "
                      f"Found: {repomix_passthrough_args[0]}")
    elif repomix_passthrough_args:
        repomix_passthrough_args = repomix_passthrough_args[1:]

    print("Repo Roast Configuration:")
    print(f"  Repository Path: {args.repo}")
    print(f"  Roast Type: {args.type}")
    print(f"  Output File: {args.output}")

    # --- Steps ---
    # 1. Check API Credentials
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE") # Can be None

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        print("Please set it directly or create a .env file.", file=sys.stderr)
        sys.exit(1)

    if api_base:
        print(f"  Using API Base URL: {api_base}")
    else:
        print("  Using default OpenAI API Base URL.")

    # Determine instruction file path
    prompt_file_path = os.path.join(PROMPTS_DIR, f"{args.type}.md")
    # 2. Run repomix
    run_repomix(
        repo_path=args.repo,
        passthrough_args=repomix_passthrough_args,
        optimize_preset=args.optimize,
        instruction_file_path=prompt_file_path
    )

    # 3. Read repomix output
    repomix_content = read_file(REPOMIX_OUTPUT_FILE)

    # 4. Estimate token count and warn if necessary
    token_count = estimate_token_count(repomix_content, "cl100k_base")  # Or o200k_base, depending on model

    if token_count > TOKEN_WARNING_THRESHOLD and not args.yes:
        print(f"Warning: The input to the LLM has an estimated token count of {token_count}, which may be expensive.")
        confirmation = input("Do you want to continue? (yes/no): ").lower()
        if confirmation != "yes":
            print("Operation cancelled.")
            sys.exit(0)  # Exit gracefully

    # 6. Call LLM API
    llm_response = call_llm(api_key, api_base, repomix_content, model=args.model)

    # 7. Write output file
    write_output(args.output, llm_response)

    # 8. Optional: Clean up repomix output file? Or keep it for inspection?
    # For now, let's keep it. Add cleanup later if needed.
    # os.remove(REPOMIX_OUTPUT_FILE)
    print("Repo Roast finished successfully!")


if __name__ == "__main__":
    main()
