#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from pathlib import Path
import mkdocs_gen_files
import os
from typing import Optional, List, Set


def process_directory(
    root: Path,
    src: Path,
    verbose: bool = True,
    exclude_patterns: Set[str] = None
) -> None:
    """
    Recursively process directories including symlinks to generate documentation.

    Args:
        root: Root project directory
        src: Source code directory to process
        verbose: Whether to print detailed processing information
        exclude_patterns: Set of patterns to exclude (e.g., 'venv', 'tests')
    """
    if exclude_patterns is None:
        exclude_patterns = {'venv', 'env', '.env', 'tests', 'test'}

    def log(msg: str, level: Optional[str] = None) -> None:
        if verbose:
            prefix = f"[{level}] " if level else ""
            print(f"{prefix}{msg}")

    def should_process_directory(path: str) -> bool:
        """Check if directory should be processed based on exclude patterns."""
        path_parts = Path(path).parts
        return not any(
            exclude in path_parts
            for exclude in exclude_patterns
        )

    log(f"Starting processing from root directory: {root}")
    log(f"Source directory: {src}")
    log(f"Exclusion patterns: {exclude_patterns}")

    stats = {
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "excluded_files": 0,
        "errors": 0
    }

    # Use os.walk with followlinks=True to properly handle symlinks
    for dirpath, dirnames, filenames in os.walk(src, followlinks=True):
        # Skip excluded directories
        if not should_process_directory(dirpath):
            log(f"\nSkipping excluded directory: {dirpath}", "EXCLUDE")
            stats["excluded_files"] += len([f for f in filenames if f.endswith('.py')])
            continue

        current_dir = Path(dirpath)
        is_symlink = current_dir.is_symlink()
        symlink_target = current_dir.resolve() if is_symlink else None

        log(f"\nProcessing directory: {current_dir}")
        if is_symlink:
            log(f"  Symlink detected -> points to: {symlink_target}")

        # Process all Python files in current directory
        python_files = sorted([
            f for f in filenames
            if f.endswith('.py')
            and not any(test in f.lower() for test in ['test', 'conftest'])
        ])
        stats["total_files"] += len(python_files)

        if not python_files:
            log("  No Python files found in directory", "INFO")
            continue

        log(f"  Found {len(python_files)} Python files: {python_files}")

        for filename in python_files:
            path = current_dir / filename
            log(f"\n  Processing file: {filename}")

            try:
                # Get the relative path from src to maintain proper structure
                module_path = path.relative_to(src).with_suffix("")
                doc_path = path.relative_to(src).with_suffix(".md")
                full_doc_path = Path("reference", doc_path)

                parts = tuple(module_path.parts)

                # Skip __init__.py and main.py files
                if parts[-1] in ["__init__", "main"]:
                    log(f"    Skipping special file: {path}", "SKIP")
                    stats["skipped_files"] += 1
                    continue

                # Create documentation file
                identifier = ".".join(parts)
                log(f"    Creating documentation for module: {identifier}")
                log(f"    Output path: {full_doc_path}")

                with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                    print("::: " + identifier, file=fd)

                # Set edit path relative to root
                relative_path = path.relative_to(root)
                mkdocs_gen_files.set_edit_path(full_doc_path, relative_path)
                log(f"    Set edit path: {relative_path}")

                stats["processed_files"] += 1
                log("    Successfully processed file", "SUCCESS")

            except (ValueError, RuntimeError) as e:
                stats["errors"] += 1
                log(f"    Error processing {path}: {e}", "ERROR")

    # Print final statistics
    log("\nProcessing completed! Summary:", "STATS")
    log(f"  Total files found: {stats['total_files']}")
    log(f"  Successfully processed: {stats['processed_files']}")
    log(f"  Skipped files: {stats['skipped_files']}")
    log(f"  Excluded files: {stats['excluded_files']}")
    log(f"  Errors encountered: {stats['errors']}")


class DirectoryNotFoundError(Exception):
    """Raised when the source directory is not found."""
    pass

# Define exclusion patterns
exclude_patterns = {
        'venv',
        'env',
        '.env',
        'tests',
        'test',
        '__pycache__',
        '.pytest_cache',
        'build',
        'dist',
        'egg-info'
    }

root = Path(__file__).parent.parent

#src = root / "src/tabpfn/src"
#process_directory(root, src, verbose=True, exclude_patterns=exclude_patterns)

#src = root / "src/tabpfn-extensions/src"
#process_directory(root, src, verbose=True, exclude_patterns=exclude_patterns)

src = root / "src/tabpfn-client/"
process_directory(root, src, verbose=True, exclude_patterns=exclude_patterns)
