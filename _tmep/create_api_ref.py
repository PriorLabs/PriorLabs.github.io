#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import os
from pathlib import Path
import ast

def parse_python_file(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())
    
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and not child.name.startswith('_'):
                    methods.append(child.name)
            classes[node.name] = methods
    return classes

def create_nav_list(classes_by_file):
    with open("nav.yml", "w") as f:
        f.write("- TabPFN-Local:\n")
        for file_path, classes in classes_by_file.items():
            module_name = Path(file_path).stem
            f.write(f"    - {module_name}:\n")
            for class_name in classes:
                doc_path = f"api/{file_path}/{class_name}.md".replace('.py/', '/')
                f.write(f"      - {class_name}: {doc_path}\n")

def create_doc_files(repo_path, output_dir):
    classes_by_file = {}
    
    for py_file in Path(repo_path).rglob("*.py"):
        rel_path = py_file.relative_to(repo_path)
        module_path = str(rel_path)
        
        classes = parse_python_file(py_file)
        if classes:
            classes_by_file[module_path] = classes.keys()
        
        module_dir = Path(output_dir) / rel_path.parent
        os.makedirs(module_dir, exist_ok=True)
        
        for class_name, methods in classes.items():
            doc_path = module_dir / f"{class_name}.md"
            with open(doc_path, 'w') as f:
                f.write(f"::: {str(rel_path).replace('/', '.')[:-3]}.{class_name}\n")
                f.write("    handler: python\n")
                f.write("    options:\n")
                f.write("      members:\n")
                f.write("        - __init__\n")
                f.write("        - forward\n")
                for method in methods:
                    f.write(f"        - {method}\n")
    
    create_nav_list(classes_by_file)
                        
if __name__ == "__main__":
    # Example usage
    repo_path = "tabpfn-client"
    output_dir = "docs"
    create_doc_files(repo_path, output_dir)