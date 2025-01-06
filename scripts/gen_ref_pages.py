from pathlib import Path
import mkdocs_gen_files
import os
from typing import Optional, List, Set, Dict, NamedTuple
import ast


class FileContent(NamedTuple):
    """Statistics about Python file content."""
    has_functions: bool
    has_classes: bool
    has_variables: bool
    has_constants: bool
    has_type_defs: bool
    has_dataclasses: bool
    has_enums: bool
    has_protocols: bool
    has_decorators: bool
    line_count: int
    doc_strings: int
    comment_lines: int
    imports_count: int
    complexity_score: int  # Rough measure of code complexity


def calculate_complexity(node: ast.AST) -> int:
    """
    Calculate a rough complexity score for an AST node.

    Args:
        node: AST node to analyze

    Returns:
        int: Complexity score
    """
    score = 0

    # Count branching structures
    if isinstance(node, (ast.If, ast.While, ast.For)):
        score += 1

    # Count exception handling
    elif isinstance(node, ast.Try):
        score += len(node.handlers) + len(node.finalbody)

    # Count logical operators
    elif isinstance(node, (ast.BoolOp, ast.BinOp)):
        score += 1

    # Recursively process child nodes
    for child in ast.iter_child_nodes(node):
        score += calculate_complexity(child)

    return score


def get_method_signature(node: ast.FunctionDef) -> str:
    """
    Generate a method signature from an AST node.

    Args:
        node: AST node of the function/method

    Returns:
        str: Method signature
    """
    args = []

    # Add self/cls parameter for methods
    if node.args.args and node.args.args[0].arg in ('self', 'cls'):
        args.append(node.args.args[0].arg)

    # Add positional arguments
    args.extend(arg.arg for arg in node.args.args[1:])

    # Add keyword arguments with defaults
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if default is None:
            args.append(f"{arg.arg}")
        else:
            args.append(f"{arg.arg}={ast.unparse(default)}")

    # Add *args if present
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")

    # Add **kwargs if present
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    return f"def {node.name}({', '.join(args)})"


def analyze_file_content(file_path: Path) -> Optional[FileContent]:
    """
    Analyze Python file content for different types of definitions.

    Args:
        file_path: Path to the Python file

    Returns:
        Optional[FileContent]: Content statistics if parsing successful, None if errors
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip files that are just whitespace
        if not content.strip():
            return None

        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return None

        # Initialize counters
        has_functions = False
        has_classes = False
        has_variables = False
        has_constants = False
        has_type_defs = False
        has_dataclasses = False
        has_enums = False
        has_protocols = False
        has_decorators = False
        doc_strings = 0
        comment_lines = 0
        imports_count = 0
        complexity_score = 0

        for node in ast.walk(tree):
            # Check for functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_functions = True
                if ast.get_docstring(node):
                    doc_strings += 1

            # Check for classes
            elif isinstance(node, ast.ClassDef):
                has_classes = True
                if ast.get_docstring(node):
                    doc_strings += 1

            # Check for variables
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                # Check for constants (all caps names)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id.isupper():
                                has_constants = True
                            else:
                                has_variables = True
                else:
                    has_variables = True

            # Check for type definitions
            elif isinstance(node, ast.AnnAssign):
                has_type_defs = True

            # Check for dataclasses
            elif isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                        has_dataclasses = True
                    has_decorators = True

            # Check for enums
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'Enum':
                        has_enums = True

            # Check for protocols
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'Protocol':
                        has_protocols = True

            # Count imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports_count += 1

            # Calculate complexity
            complexity_score += calculate_complexity(node)

            # Count comment lines
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                comment_lines += len(node.value.s.split('\n'))

            # Module level docstring
            elif isinstance(node, ast.Module) and ast.get_docstring(node):
                doc_strings += 1

        # Count non-empty lines
        line_count = len([line for line in content.splitlines() if line.strip()])

        return FileContent(
            has_functions=has_functions,
            has_classes=has_classes,
            has_variables=has_variables,
            has_constants=has_constants,
            has_type_defs=has_type_defs,
            has_dataclasses=has_dataclasses,
            has_enums=has_enums,
            has_protocols=has_protocols,
            has_decorators=has_decorators,
            line_count=line_count,
            doc_strings=doc_strings,
            comment_lines=comment_lines,
            imports_count=imports_count,
            complexity_score=complexity_score
        )

    except (UnicodeDecodeError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return None


def is_valid_source_dir(path: Path) -> bool:
    """
    Check if the source directory exists and is accessible.

    Args:
        path: Path to check

    Returns:
        bool: True if directory exists and is accessible
    """
    try:
        return path.exists() and path.is_dir() and os.access(path, os.R_OK)
    except (OSError, IOError):
        return False


def resolve_symlink_safely(path: Path) -> Optional[Path]:
    """
    Safely resolve a symlink with cycle detection.

    Args:
        path: Path to resolve

    Returns:
        Optional[Path]: Resolved path or None if error/cycle detected
    """
    try:
        seen = set()
        current = path
        while current.is_symlink():
            if current in seen:
                print(f"Symlink cycle detected at {current}")
                return None
            seen.add(current)
            current = current.resolve()
        return current
    except (OSError, RuntimeError) as e:
        print(f"Error resolving symlink {path}: {e}")
        return None


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
        exclude_patterns: Set of patterns to exclude
    """
    # Check source directory
    if not is_valid_source_dir(src):
        raise DirectoryNotFoundError(f"Source directory not found or not accessible: {src}")

    if exclude_patterns is None:
        exclude_patterns = {
            'venv', 'env', '.env', '.venv', 'virtualenv',  # Virtual environments
            'tests', 'test', 'testing',  # Test directories
            '__pycache__', '.pytest_cache',  # Cache directories
            'build', 'dist', '*egg-info',  # Build artifacts
            '.git', '.svn', '.hg',  # Version control
            'docs', 'examples', 'notebooks',  # Documentation
            'scripts', 'tools', 'utils',  # Utility directories
            'migrations', 'alembic',  # Database migrations
            'node_modules', 'static', 'media',  # Web assets
            'tmp', 'temp', 'cache',  # Temporary files
            '.*'  # Hidden directories
        }

    def log(msg: str, level: Optional[str] = None) -> None:
        if verbose:
            prefix = f"[{level}] " if level else ""
            print(f"{prefix}{msg}")

    def should_process_directory(path: str) -> bool:
        """Check if directory should be processed based on exclude patterns."""
        path_parts = Path(path).parts
        return not any(
            any(part.lower().startswith(exclude.lower()) for exclude in exclude_patterns)
            for part in path_parts
        )

    log(f"Starting processing from root directory: {root}")
    log(f"Source directory: {src}")
    log(f"Exclusion patterns: {exclude_patterns}")

    stats = {
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "excluded_files": 0,
        "empty_files": 0,
        "syntax_errors": 0,
        "symlink_errors": 0,
        "other_errors": 0,
        "errors": 0,  # Total error count
        "content_stats": {
            "with_functions": 0,
            "with_classes": 0,
            "with_variables": 0,
            "with_constants": 0,
            "with_types": 0,
            "with_docstrings": 0,
            "with_dataclasses": 0,
            "with_enums": 0,
            "with_protocols": 0,
            "with_decorators": 0,
            "total_complexity": 0,
            "avg_complexity": 0
        }
    }

    for dirpath, dirnames, filenames in os.walk(src, followlinks=True):
        if not should_process_directory(dirpath):
            stats["excluded_files"] += len([f for f in filenames if f.endswith('.py')])
            continue

        current_dir = Path(dirpath)

        # Handle symlinks
        if current_dir.is_symlink():
            resolved_dir = resolve_symlink_safely(current_dir)
            if not resolved_dir:
                stats["symlink_errors"] += 1
                continue
            current_dir = resolved_dir

        python_files = sorted([
            f for f in filenames
            if f.endswith('.py')
               and not any(test in f.lower() for test in ['test', 'conftest'])
        ])
        stats["total_files"] += len(python_files)

        if not python_files:
            continue

        for filename in python_files:
            path = current_dir / filename
            log(f"\n  Processing file: {filename}")

            try:
                # Analyze file content
                content_stats = analyze_file_content(path)
                if not content_stats:
                    stats["empty_files"] += 1
                    log(f"    Skipping empty/invalid file: {path}", "EMPTY")
                    continue

                # Update content statistics
                if content_stats.has_functions:
                    stats["content_stats"]["with_functions"] += 1
                if content_stats.has_classes:
                    stats["content_stats"]["with_classes"] += 1
                if content_stats.has_variables:
                    stats["content_stats"]["with_variables"] += 1
                if content_stats.has_constants:
                    stats["content_stats"]["with_constants"] += 1
                if content_stats.has_type_defs:
                    stats["content_stats"]["with_types"] += 1
                if content_stats.doc_strings > 0:
                    stats["content_stats"]["with_docstrings"] += 1
                if content_stats.has_dataclasses:
                    stats["content_stats"]["with_dataclasses"] += 1
                if content_stats.has_enums:
                    stats["content_stats"]["with_enums"] += 1
                if content_stats.has_protocols:
                    stats["content_stats"]["with_protocols"] += 1
                if content_stats.has_decorators:
                    stats["content_stats"]["with_decorators"] += 1

                # Update complexity metrics
                stats["content_stats"]["total_complexity"] += content_stats.complexity_score
                if stats["processed_files"] > 0:
                    stats["content_stats"]["avg_complexity"] = (
                            stats["content_stats"]["total_complexity"] / stats["processed_files"]
                    )

                # Skip if no meaningful content
                if not any([
                    content_stats.has_functions,
                    content_stats.has_classes,
                    content_stats.has_variables,
                    content_stats.has_constants,
                    content_stats.has_type_defs
                ]):
                    log(f"    Skipping file with no meaningful content: {path}", "SKIP")
                    stats["skipped_files"] += 1
                    continue

                module_path = path.relative_to(src).with_suffix("")
                doc_path = path.relative_to(src).with_suffix(".md")
                full_doc_path = Path("reference", doc_path)

                parts = tuple(module_path.parts)

                if parts[-1] in ["__init__", "main"]:
                    stats["skipped_files"] += 1
                    continue

                identifier = ".".join(parts)

                with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                    # Generate standard module documentation
                    print("::: " + identifier, file=fd)

                    # Generate class-specific documentation if present
                    if content_stats.has_classes:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                class_name = node.name
                                print(f"\n## {class_name}", file=fd)

                                # Get class docstring
                                class_doc = ast.get_docstring(node)
                                if class_doc:
                                    print(f"\n{class_doc}\n", file=fd)

                                # List class inheritance
                                if node.bases:
                                    bases = [ast.unparse(base) for base in node.bases]
                                    print(f"\nInherits from: {', '.join(bases)}\n", file=fd)

                                print(f"::: {identifier}.{class_name}", file=fd)

                                # Document methods grouped by type
                                methods = {
                                    'Special Methods': [],
                                    'Public Methods': [],
                                    'Protected Methods': [],
                                    'Private Methods': []
                                }

                                for method in [n for n in node.body if
                                               isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                                    method_name = method.name
                                    if method_name.startswith('__') and method_name.endswith('__'):
                                        methods['Special Methods'].append(method)
                                    elif method_name.startswith('__'):
                                        methods['Private Methods'].append(method)
                                    elif method_name.startswith('_'):
                                        methods['Protected Methods'].append(method)
                                    else:
                                        methods['Public Methods'].append(method)

                                for category, category_methods in methods.items():
                                    if category_methods:
                                        print(f"\n### {category}", file=fd)
                                        for method in category_methods:
                                            method_name = method.name
                                            print(f"\n#### {method_name}", file=fd)

                                            # Add method signature
                                            signature = get_method_signature(method)
                                            print(f"```python\n{signature}\n```\n", file=fd)

                                            # Add method docstring
                                            docstring = ast.get_docstring(method)
                                            if docstring:
                                                print(f"{docstring}\n", file=fd)

                                            print(f"::: {identifier}.{class_name}.{method_name}", file=fd)

                relative_path = path.relative_to(root)
                mkdocs_gen_files.set_edit_path(full_doc_path, relative_path)

                stats["processed_files"] += 1

            except SyntaxError:
                stats["syntax_errors"] += 1
                stats["errors"] += 1
                log(f"Syntax error in {path}", "ERROR")
            except Exception as e:
                stats["other_errors"] += 1
                stats["errors"] += 1
                log(f"Error processing {path}: {e}", "ERROR")

    # Print final statistics
    log("\nProcessing Summary:", "STATS")
    log(f"  Processed: {stats['processed_files']} files", "STATS")
    if stats['errors'] > 0:
        log(f"  Errors: {stats['errors']} ({stats['syntax_errors']} syntax, {stats['other_errors']} other)", "STATS")


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