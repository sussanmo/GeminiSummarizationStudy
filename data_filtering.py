import os
import json
import gzip  # Make sure to import gzip
import random 
# from radon.complexity import cc_visit
import ast
import re



def jsonIteration(directory):
    methods = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl.gz'):
            file_path = os.path.join(directory, filename)
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Access the fields you need
                    func_name = data.get('func_name')
                    original_string = data.get('original_string')
                    
                    # Extract only the code and docstring separately
                    code_lines = original_string.splitlines()
                    docstring = ''
                    code = ''
                    is_docstring = False
                    
                    for line in code_lines:
                        if line.strip().startswith('"""'):
                            is_docstring = not is_docstring  # Toggle docstring status
                            if is_docstring and not docstring:  # Start of docstring
                                docstring += line.strip().strip('"""') + '\n'
                            continue  # Skip adding the docstring line to code
                        if is_docstring:
                            docstring += line.strip() + '\n'  # Add lines to docstring
                        else:
                            code += line + '\n'  # Add to code if not in docstring
                    
                    # Only consider methods with a docstring
                    if docstring:
                        methods.append({
                            'func_name': func_name,
                            'original_string': original_string,
                            'code': code.strip(),  # Remove trailing newline
                            'docstring': docstring.strip()  # Remove trailing newline
                        })
    return methods

def calculate_line_count(code):
    """Calculate the number of lines in the method."""
    return code.count('\n') + 1  # Add 1 for the last line

def analyze_code_complexity(code: str):
    """Analyze the cyclomatic complexity of the given code."""
    count = 0
    try:
        complexities = cc_visit(code)
        return {complexity.name: complexity.complexity for complexity in complexities}
    except SyntaxError as e:
        print(f"Skipping function due to SyntaxError: {e}")
        return None  # Skip or return a specific indicator for invalid functions
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    


def check_line_wrapping(code, max_length=80):
    """Check if any line exceeds the max_length."""
    lines = code.splitlines()
    return any(len(line) > max_length for line in lines)

def filter_methods(methods, sample_percentage=0.05):
    """Randomly select a percentage of methods and calculate additional metrics."""
    selected_methods = []
    sample_size = max(1, int(len(methods) * sample_percentage))  # Ensure at least one sample
    random_sample = random.sample(methods, min(sample_size, len(methods)))  # Avoid exceeding length

    for method in random_sample:
        # Check for 'code' or 'original_string'
        if 'original_string' not in method:
            print(f"Warning: Method {method.get('func_name', 'unknown')} does not have an 'original_string' key.")
            continue  # Skip if 'original_string' is missing

        original_string = method['original_string']
        line_count = calculate_line_count(original_string)

        try:
            cyclomatic_complexity = analyze_code_complexity(original_string)
        except SyntaxError as e:
            print(f"Skipping method '{method['func_name']}' due to SyntaxError: {e}")
            cyclomatic_complexity = None  # Handle errors gracefully

        has_line_wrapping = check_line_wrapping(original_string)

        selected_methods.append({
            'func_name': method['func_name'],
            'original_string': original_string,  # Keep the original code string
            'docstring': method['docstring'],
            'line_count': line_count,
            'cyclomatic_complexity': cyclomatic_complexity,
            'has_line_wrapping': has_line_wrapping
        })
    
    #print("Selected methods:", selected_methods)  # Debug print
    print(len(selected_methods))
    return selected_methods

                   
def random_sample(methods, sample_percentage=0.05):
    """Randomly sample a percentage of methods."""
    sample_size = int(len(methods) * sample_percentage)
    return random.sample(methods, sample_size)

def save_to_json(data, filename):
    """Save the filtered methods to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def filter_methods(filtered_file, final_methods_file, output_file):
    def extract_method_name(method_string):
        """Extract the method name from a method string."""
        match = re.match(r"def\s+(\w+)\s*\(", method_string)
        return match.group(1) if match else None

    # Load the final methods
    with open(final_methods_file, 'r', encoding='utf-8') as f:
        final_methods = json.load(f)

    # Extract method names
    method_names = {
        extract_method_name(method['method'])
        for method in final_methods if 'method' in method
    }

    matching_entries = []

    # Detect if the file is GZip or plain text
    open_func = gzip.open if filtered_file.endswith('.gz') else open
    with open_func(filtered_file, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                continue
            try:
                entry = json.loads(line)  # Parse JSON
                if entry['func_name'] in method_names:
                    matching_entries.append(entry)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue

    # Save the matching entries to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matching_entries, f, ensure_ascii=False, indent=4)

def normalize_method_signature(signature):
    """Normalize the method signature for consistent matching."""
    # Strip spaces, newlines, and colons
    return " ".join(signature.strip().split())
def parse_decldesc(decldesc_file):
    """Parse decldesc file to extract method signatures and their descriptions."""
    decldesc_entries = []
    with open(decldesc_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()
    
    for line in lines:
        if "DCNL" in line:
            # Split into method and description
            method, description = line.split("DCNL", 1)
            method = normalize_method_signature(method)
            description = description.strip().replace("DCNL", " ").strip()
            decldesc_entries.append((method, description))
    
    return decldesc_entries

def add_descriptions_to_methods(methods_file, decldesc_file, output_file):
    # Load the JSON data from final_methods.json
    with open(methods_file, 'r', encoding='utf-8') as f:
        methods_data = json.load(f)
    
    # Parse the decldesc file
    decldesc_entries = parse_decldesc(decldesc_file)

    # Debug: Log decldesc method signatures
    print("Parsed decldesc methods:")
    for method, description in decldesc_entries:
        print(f"- {method}")

    # Update methods_data with descriptions
    unmatched_methods = []
    for entry in methods_data:
        raw_signature = entry['method'].split("\n")[0].strip()  # Extract the first line of the method
        normalized_signature = normalize_method_signature(raw_signature)
        
        # Direct comparison
        matched_description = None
        for method, description in decldesc_entries:
            if normalized_signature == method:
                matched_description = description
                break

        if matched_description:
            entry['description'] = matched_description
        else:
            unmatched_methods.append(raw_signature)
    
    # Write the updated data back to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(methods_data, f, indent=4)
    
    # Log unmatched methods for debugging
    print(f"Unmatched methods ({len(unmatched_methods)}):")
    for method in unmatched_methods:
        print(f"- {method}")

if __name__ == '__main__':
    directory = 'CodeSearchNet/python/python/final/jsonl/train'
    # all_methods = jsonIteration(directory)
    #sampled_methods = random_sample(all_methods, 0.05)
    # sampled_methods = filter_methods(all_methods)

    # Save the sampled methods to a JSON file
    # save_to_json(sampled_methods, 'filtered_dataset.jsonl.gz')

    filtered_file = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/filtered_dataset.jsonl.gz'
    final_methods_file = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/final_methods.json'
    output_file = 'final_methods_with_summary.json'

    # jsonIteration2(filtered_file, final_methods_file)

    # Path to the gzipped file
    # gzipped_file = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/code-docstring-corpus/parallel-corpus/data_ps.all.train.gz'

    # Print the first two entries
    # print_first_two_entries(gzipped_file)

    # gzipped_file = '/path/to/data_ps.all.train.gz'
    # final_methods_file = '/path/to/final_methods.json'
    decldesc_file = '/Users/suadhm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/code-docstring-corpus/parallel-corpus/data_ps.decldesc.train'
    output_file = 'final_methods_with_descriptions.json'

    add_descriptions_to_methods(final_methods_file, decldesc_file, output_file)
