import ast

class SemanticAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.methods = []

    def visit_FunctionDef(self, node):
        """ Extracts semantic categories from function definitions """
        method_info = {
            "category": "method declaration",
            "function_name": node.name,
            "parameters": [
                {"category": "parameter", "token": arg.arg} for arg in node.args.args
            ],
            "body": self.extract_body_elements(node.body),  
        }
        self.methods.append(method_info)
        self.generic_visit(node)

    def extract_body_elements(self, body):
        """ Recursively extract semantic categories from function body with token details """
        elements = []
        for stmt in body:
            elements.extend(self.process_node(stmt))
        return elements

    def process_node(self, node):
        """ Categorizes an individual AST node and returns a list of its tokens and categories """
        elements = []

        def add_element(category, node):
            """ Helper to add categorized tokens """
            elements.append({
                "category": category,
                "token": self.get_source(node)
            })

        # Assignment
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            add_element("assignment", node)

        # Literals, Variables, and Operators
        elif isinstance(node, ast.Constant):
            add_element("literal", node)

        elif isinstance(node, ast.Name):
            add_element("variable name", node)

        elif isinstance(node, ast.UnaryOp):
            add_element("operator", node)

        # Attribute Access (External Variable/Method)
        elif isinstance(node, ast.Attribute):
            add_element("external variable/method", node)

        # Loops
        elif isinstance(node, (ast.For, ast.While)):
            add_element("loop body", node)
            for stmt in node.body:
                elements.extend(self.process_node(stmt))

        # Conditional Statements
        elif isinstance(node, ast.If):
            add_element("conditional statement", node.test)

            add_element("conditional body", node)
            elements.extend(self.extract_nested_calls(node.test))

            
            # Extract variable names inside the condition
            for sub_node in ast.walk(node.test):
                if isinstance(sub_node, ast.Name):
                    add_element("variable name", sub_node)
                elif isinstance(sub_node, ast.Attribute):
                    add_element("variable name", sub_node.value)
                    add_element("external variable/method", sub_node)

            

        elif isinstance(node, ast.Compare):
            add_element("conditional statement", node)
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Name):
                    add_element("variable name", sub_node)
                elif isinstance(sub_node, ast.Attribute):
                    add_element("variable name", sub_node.value)
                    add_element("external variable/method", sub_node)

        # Function Calls
        elif isinstance(node, ast.Call):
            add_element("method call", node)

        # Exception Handling
        elif isinstance(node, ast.Raise):
            add_element("exception handling", node)

        # Return Statement
        elif isinstance(node, ast.Return):
            add_element("return", node)

        # Indexing
        elif isinstance(node, ast.Subscript):
            add_element("index operation", node)

        return elements

    def extract_nested_calls(self, node):
        """ Extracts method calls within expressions like loop headers, conditions, and returns """
        calls = []
        if isinstance(node, ast.Call):
            calls.append({"category": "method call", "token": self.get_source(node)})
        elif isinstance(node, (ast.BinOp, ast.BoolOp, ast.UnaryOp)):
            calls.extend(self.extract_nested_calls(node.left) if hasattr(node, 'left') else [])
            calls.extend(self.extract_nested_calls(node.right) if hasattr(node, 'right') else [])
            calls.extend(self.extract_nested_calls(node.operand) if hasattr(node, 'operand') else [])
        elif isinstance(node, (ast.IfExp, ast.Compare)):
            calls.extend(self.extract_nested_calls(node.left))
            for comparator in node.comparators:
                calls.extend(self.extract_nested_calls(comparator))
        return calls

    def get_source(self, node):
        """ Get the source code of the given AST node (only works if input was a string) """
        return ast.unparse(node) if hasattr(ast, 'unparse') else None

def extract_semantic_categories(source_code):
    tree = ast.parse(source_code)
    analyzer = SemanticAnalyzer()
    analyzer.visit(tree)
    return analyzer.methods

# Example Code
source_code = """
def is_error(node): 
    for child_node in node.get_children(): 
        if isinstance(child_node, astroid.Raise): 
            return True 
    return False 
"""

import json
# print(json.dumps(extract_semantic_categories(source_code), indent=2))


# Read the source code from final_methods.py
with open("final_methods.py", "r", encoding="utf-8") as f:
    source_code = f.read()

# Parse the AST
tree = ast.parse(source_code)

# Extract individual functions
methods = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

# Process each method separately
results = {}
for method in methods:
    method_name = method.name
    method_code = ast.unparse(method)  # Convert AST node back to code (Python 3.9+ required)
    results[method_name] = extract_semantic_categories(method_code)

# Save the results to a file
output_file = "semantic_categories.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Semantic categories saved to {output_file}")