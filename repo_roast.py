#!/usr/bin/env python3
"""
Repo Roast: A tool for analyzing and reviewing Git repositories using LLMs.

This tool combines repository analysis with LLM-powered code reviews to provide
insightful, entertaining, and educational feedback on codebases.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


class OptimizePreset(Enum):
    """Optimization presets for repository analysis."""
    COMPACT = "compact"
    AGGRESSIVE = "aggressive"


class ExitCode(Enum):
    """Exit codes for the application."""
    SUCCESS = 0
    CONFIG_ERROR = 1
    REPOMIX_ERROR = 2
    API_ERROR = 3
    FILE_ERROR = 4
    UNEXPECTED_ERROR = 5


@dataclass
class Config:
    """Application configuration."""
    repo_path: str
    roast_type: str
    output_file: str
    model: str
    optimize_preset: Optional[OptimizePreset] = None
    skip_confirmation: bool = False
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    repomix_args: List[str] = None


class TokenCounter:
    """Utility for counting tokens in text."""

    DEFAULT_ENCODING = "cl100k_base"
    WARNING_THRESHOLD = 10000

    @staticmethod
    def estimate_token_count(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
        """
        Estimate the number of tokens in the given text.

        Args:
            text: The text to analyze
            encoding_name: The encoding to use for tokenization

        Returns:
            The estimated token count, or -1 if estimation fails
        """
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except ValueError as e:
            logging.warning(f"Could not load tokenizer for encoding '{encoding_name}': {e}")
            return -1


class PromptManager:
    """Manages prompt templates and discovery."""

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the PromptManager.

        Args:
            prompts_dir: Directory containing prompt template files
        """
        self.prompts_dir = Path(prompts_dir)

    def discover_roast_types(self) -> List[str]:
        """
        Discover available roast types from prompt templates.

        Returns:
            List of available roast type names (without extension)

        Raises:
            FileNotFoundError: If the prompts directory doesn't exist
            ValueError: If no prompt files are found
        """
        if not self.prompts_dir.is_dir():
            raise FileNotFoundError(f"Prompts directory '{self.prompts_dir}' not found")

        types = []
        for file_path in self.prompts_dir.glob("*.md"):
            types.append(file_path.stem)

        if not types:
            raise ValueError(f"No prompt files (.md) found in '{self.prompts_dir}'")

        return sorted(types)

    def get_prompt_path(self, roast_type: str) -> Path:
        """
        Get the file path for the specified roast type.

        Args:
            roast_type: The roast type to get the prompt for

        Returns:
            Path to the prompt file
        """
        return self.prompts_dir / f"{roast_type}.md"


class RepomixRunner:
    """Handles running the repomix tool and processing its output."""

    # Default output file from repomix
    DEFAULT_OUTPUT_FILE = "repomix-output.md"

    # Define repomix optimization presets
    OPTIMIZE_PRESETS = {
        OptimizePreset.COMPACT: ["--remove-comments", "--remove-empty-lines"],
        OptimizePreset.AGGRESSIVE: [
            "--remove-comments",
            "--remove-empty-lines",
            "--compress",
            "--no-directory-structure",
            "--no-file-summary"
        ],
    }

    def __init__(self, output_file: str = DEFAULT_OUTPUT_FILE):
        """
        Initialize the RepomixRunner.

        Args:
            output_file: Path where repomix will write its output
        """
        self.output_file = output_file

    def run(self, repo_path: str, args: List[str] = None,
            optimize_preset: Optional[OptimizePreset] = None,
            instruction_file_path: Optional[str] = None) -> str:
        """
        Run repomix on the specified repository.

        Args:
            repo_path: Path to the target Git repository
            args: Additional arguments to pass to repomix
            optimize_preset: Optimization preset to apply
            instruction_file_path: Path to instruction file for repomix

        Returns:
            Path to the output file containing repomix results

        Raises:
            FileNotFoundError: If npx command is not available
            subprocess.CalledProcessError: If repomix fails
            FileNotFoundError: If output file is not found after running repomix
        """
        import subprocess
        from subprocess import CalledProcessError

        base_command = ["npx", "repomix", "--style", "markdown"]

        # Add optimization preset arguments if specified
        preset_args = []
        if optimize_preset:
            preset_args = self.OPTIMIZE_PRESETS.get(optimize_preset, [])

        # Add instruction file arguments if specified
        instruction_args = []
        if instruction_file_path:
            if not Path(instruction_file_path).exists():
                logging.warning(f"Instruction file '{instruction_file_path}' not found. "
                              f"Proceeding without it.")
            else:
                logging.info(f"Using instruction file: {instruction_file_path}")
                instruction_args = ["--instruction-file-path", instruction_file_path]

        # Build the full command
        command = base_command + preset_args + instruction_args + [repo_path] + (args or [])

        logging.info(f"Running repomix for repository: {repo_path}" +
                   (f" (Optimize preset: {optimize_preset.value})" if optimize_preset else ""))
        logging.debug(f"Command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            logging.info("repomix completed successfully.")

            if not Path(self.output_file).exists():
                raise FileNotFoundError(
                    f"repomix ran successfully but the expected output file "
                    f"'{self.output_file}' was not found."
                )

            return self.output_file

        except FileNotFoundError:
            logging.error("'npx' command not found. Make sure Node.js and npm/npx are installed.")
            raise
        except CalledProcessError as e:
            logging.error(f"repomix failed with exit code {e.returncode}")
            logging.error("--- repomix stdout ---")
            logging.error(e.stdout)
            logging.error("--- repomix stderr ---")
            logging.error(e.stderr)
            raise


class LLMClient:
    """Client for interacting with LLM APIs."""

    DEFAULT_MODEL = "gemma3:4b"

    def __init__(self, api_key: str, api_base: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (optional)
        """
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def generate_response(self, prompt: str, model: str = DEFAULT_MODEL) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: The model to use

        Returns:
            The generated text response

        Raises:
            openai.AuthenticationError: If authentication fails
            openai.RateLimitError: If rate limit is exceeded
            openai.APIConnectionError: If connection fails
            openai.APIError: For other API errors
        """
        logging.info(f"Calling LLM API (Model: {model})...")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
                # Additional parameters can be added as needed:
                # temperature=0.7,
                # max_tokens=4096
            )

            if not response.choices or not response.choices[0].message:
                raise ValueError("Unexpected API response structure")

            return response.choices[0].message.content

        except (openai.AuthenticationError, openai.RateLimitError,
                openai.APIConnectionError, openai.APIError) as e:
            logging.error(f"LLM API error: {e}")
            raise


class FileManager:
    """Handles file operations."""

    @staticmethod
    def read_file(file_path: str) -> str:
        """
        Read content from a file.

        Args:
            file_path: Path to the file to read

        Returns:
            The file content as a string

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: For other file reading errors
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logging.debug(f"Read {len(content)} characters from {file_path}")
            return content
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Error reading file {file_path}: {e}")
            raise

    @staticmethod
    def write_file(file_path: str, content: str) -> None:
        """
        Write content to a file.

        Args:
            file_path: Path to write to
            content: Content to write

        Raises:
            IOError: If writing fails
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Successfully wrote output to {file_path}")
        except IOError as e:
            logging.error(f"Error writing to file {file_path}: {e}")
            raise


class RepoRoaster:
    """Main application class for repo-roast."""

    def __init__(self):
        """Initialize the repo roaster."""
        self.prompt_manager = PromptManager()
        self.repomix_runner = RepomixRunner()
        self.token_counter = TokenCounter()
        self.file_manager = FileManager()
        self.llm_client = None  # Will be initialized during run

    def configure_logging(self, verbose: bool = False) -> None:
        """
        Configure the logging system.

        Args:
            verbose: Whether to enable verbose logging
        """
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(levelname)s: %(message)s'
        )

    def parse_args(self) -> Config:
        """
        Parse command line arguments.

        Returns:
            Config object with parsed arguments
        """
        available_roast_types = self.prompt_manager.discover_roast_types()

        parser = argparse.ArgumentParser(
            description="Roast a Git repository using repomix and an LLM.",
            epilog="Arguments after '--' are passed directly to the npx repomix command."
        )
        parser.add_argument("--repo", default=".",
                          help="Path to the target Git repository (default: current directory).")
        parser.add_argument("--type", required=True, choices=available_roast_types,
                          help="The type of roast to perform.")
        parser.add_argument("--output", required=True,
                          help="Path to the output file for the LLM response.")
        parser.add_argument("--model", default=LLMClient.DEFAULT_MODEL,
                          help=f"The LLM model to use (default: {LLMClient.DEFAULT_MODEL}).")
        parser.add_argument("--optimize", choices=[p.value for p in OptimizePreset],
                          default=None, help="Apply a preset of repomix arguments for token optimization.")
        parser.add_argument("--yes", action="store_true",
                          help="Bypass confirmation prompt and proceed automatically.")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose logging.")

        args, repomix_passthrough_args = parser.parse_known_args()

        # Configure logging early based on verbose flag
        self.configure_logging(args.verbose)

        # Process passthrough args
        cleaned_passthrough_args = []
        if repomix_passthrough_args:
            if repomix_passthrough_args[0] != '--':
                parser.error("Arguments intended for repomix must be preceded by '--'. "
                           f"Found: {repomix_passthrough_args[0]}")
            else:
                cleaned_passthrough_args = repomix_passthrough_args[1:]

        # Convert string optimize preset to enum
        optimize_preset = None
        if args.optimize:
            try:
                optimize_preset = OptimizePreset(args.optimize)
            except ValueError:
                parser.error(f"Invalid optimization preset: {args.optimize}")

        # Load environment variables from .env file
        load_dotenv()

        return Config(
            repo_path=args.repo,
            roast_type=args.type,
            output_file=args.output,
            model=args.model,
            optimize_preset=optimize_preset,
            skip_confirmation=args.yes,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            repomix_args=cleaned_passthrough_args
        )

    def validate_config(self, config: Config) -> None:
        """
        Validate the configuration.

        Args:
            config: The configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not config.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. "
                           "Please set it directly or create a .env file.")

    def run(self) -> int:
        """
        Run the repo-roast process.

        Returns:
            Exit code
        """
        try:
            # Parse arguments and validate configuration
            config = self.parse_args()
            self.validate_config(config)

            # Log configuration
            logging.info("Repo Roast Configuration:")
            logging.info(f"  Repository Path: {config.repo_path}")
            logging.info(f"  Roast Type: {config.roast_type}")
            logging.info(f"  Output File: {config.output_file}")
            logging.info(f"  Model: {config.model}")
            if config.optimize_preset:
                logging.info(f"  Optimization Preset: {config.optimize_preset.value}")

            # Initialize LLM client
            self.llm_client = LLMClient(config.api_key, config.api_base)
            if config.api_base:
                logging.info(f"  Using API Base URL: {config.api_base}")
            else:
                logging.info("  Using default OpenAI API Base URL.")

            # Run repository analysis
            prompt_file_path = str(self.prompt_manager.get_prompt_path(config.roast_type))
            repomix_output_file = self.repomix_runner.run(
                repo_path=config.repo_path,
                args=config.repomix_args,
                optimize_preset=config.optimize_preset,
                instruction_file_path=prompt_file_path
            )

            # Read analysis output
            repo_analysis = self.file_manager.read_file(repomix_output_file)

            # Check token count and confirm if necessary
            token_count = self.token_counter.estimate_token_count(repo_analysis)
            logging.info(f"Estimated token count: {token_count}")

            if token_count > self.token_counter.WARNING_THRESHOLD and not config.skip_confirmation:
                logging.warning(f"The input to the LLM has an estimated token count of {token_count}, "
                              f"which may be expensive.")
                confirmation = input("Do you want to continue? (yes/no): ").lower()
                if confirmation != "yes":
                    logging.info("Operation cancelled.")
                    return ExitCode.SUCCESS.value

            # Generate LLM response
            llm_response = self.llm_client.generate_response(repo_analysis, model=config.model)

            # Write output
            self.file_manager.write_file(config.output_file, llm_response)

            logging.info("Repo Roast finished successfully!")
            return ExitCode.SUCCESS.value

        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            return ExitCode.CONFIG_ERROR.value
        except FileNotFoundError as e:
            logging.error(f"File error: {e}")
            return ExitCode.FILE_ERROR.value
        except subprocess.SubprocessError as e:
            logging.error(f"Repomix error: {e}")
            return ExitCode.REPOMIX_ERROR.value
        except openai.OpenAIError as e:
            logging.error(f"API error: {e}")
            return ExitCode.API_ERROR.value
        except Exception as e:
            logging.exception(f"Unexpected error: {e}")
            return ExitCode.UNEXPECTED_ERROR.value


def main():
    """Program entry point."""
    roaster = RepoRoaster()
    sys.exit(roaster.run())


if __name__ == "__main__":
    main()
