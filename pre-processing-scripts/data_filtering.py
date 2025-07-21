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


def create_tasks(easy_methods, medium_methods, hard_methods, num_tasks=11):
    tasks = []
    
    # Shuffle the methods lists to ensure random order
    random.shuffle(easy_methods)
    random.shuffle(medium_methods)
    random.shuffle(hard_methods)
    
    for i in range(1, num_tasks + 1):
        # Select one method from each category and pop to avoid repetition
        easy_method = easy_methods.pop()
        medium_method = medium_methods.pop()
        hard_method = hard_methods.pop()

        # Randomize the order of easy, medium, and hard methods
        methods_in_task = [easy_method, medium_method, hard_method]
        random.shuffle(methods_in_task)

        # Create a task and add it to the list
        task = {
            f"Task {i}": methods_in_task
        }
        tasks.append(task)
    
    # Save the tasks to a new JSON file
    with open('task_map.json', 'w') as f:
        json.dump(tasks, f, indent=4)

    print(f"Generated {num_tasks} tasks in 'task_map.json'.")

if __name__ == '__main__':
    # directory = 'CodeSearchNet/python/python/final/jsonl/train'
    # all_methods = jsonIteration(directory)
    # print(len(all_methods))
    # sampled_methods = random_sample(all_methods, 0.05)
    # print (len(sampled_methods))
    # # sampled_methods = filter_methods(all_methods)

    # # Save the sampled methods to a JSON file
    # # save_to_json(sampled_methods, 'filtered_dataset.jsonl.gz')

    # filtered_file = '/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/filtered_dataset.jsonl.gz'
    final_methods_file = '/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/final_methods.json'
    # output_file = 'final_methods_with_summary.json'

    # # jsonIteration2(filtered_file, final_methods_file)

    # # Path to the gzipped file
    # # gzipped_file = '/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/code-docstring-corpus/parallel-corpus/data_ps.all.train.gz'

    # # Print the first two entries
    # # print_first_two_entries(gzipped_file)

    # # gzipped_file = '/path/to/data_ps.all.train.gz'
    # # final_methods_file = '/path/to/final_methods.json'
    # # decldesc_file = '/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/code-docstring-corpus/parallel-corpus/data_ps.decldesc.train'
    # # output_file = 'final_methods_with_descriptions.json'

    # # add_descriptions_to_methods(final_methods_file, decldesc_file, output_file)

    #     # Load the final_methods_with_descriptions.json file
    # with open('/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/final_methods_with_descriptions.json', 'r') as f:
    #     methods = json.load(f)

    # # Organize methods by cyclomatic complexity
    # easy_methods = [method for method in methods if method['cyclomatic_complexity'] == 3]
    # medium_methods = [method for method in methods if method['cyclomatic_complexity'] == 5]
    # hard_methods = [method for method in methods if method['cyclomatic_complexity'] in [7, 8]]
    
    # # Generate tasks
    # # tasks = create_tasks(easy_methods, medium_methods, hard_methods)

    
    import json

    def calculate_averages(json_data):
        # Ensure json_data is a list of dictionaries
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Parse JSON string into a Python object
        
        total_line_length = 0
        total_method_length = 0
        total_methods = len(json_data)

        for method_data in json_data:
            method_code = method_data["code"]  # Use 'code' instead of 'method'
            lines = method_code.split('\n')
            method_length = len(lines)
            total_method_length += method_length
            
            for line in lines:
                total_line_length += len(line.strip())
        
        average_line_length = total_line_length / total_methods
        average_method_length = total_method_length / total_methods
        
        return average_line_length, average_method_length

    # Ensure final_methods_file is properly formatted
    # print(type(final_methods_file))  # Debugging output
    # if isinstance(final_methods_file, str):
    #     final_methods_file = json.loads(final_methods_file)  # Convert to list of dictionaries

    print(final_methods_file)  # Check if indexing works correctly
    # avg_line_length, avg_method_length = calculate_averages(final_methods_file)

    # print(f"The final selected methods had an average line length of {avg_line_length:.2f} and an average method length of {avg_method_length:.2f}.")


    # import json

    # json_file_path = "/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/final_methods.json"
    # # json_file_path = '/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/filtered_methods.json'
    # try:
    #     with open(json_file_path, "r", encoding="utf-8") as f:
    #         data = json.load(f)  # Load JSON content into a Python object

    #     # Ensure the data is a list
    #     if not isinstance(data, list):
    #         raise ValueError("JSON data is not a list of methods.")

    #     total_line_count = 0
    #     total_methods = len(data)
    #     total_line_lengths = 0
    #     total_lines = 0

    #     for i, method_data in enumerate(data):
    #         # Get line count
    #         line_count = method_data.get("line_count", 0)  
    #         total_line_count += line_count  

    #         # Compute total characters per method
    #         method_code = method_data.get("method", "").strip()  
    #         lines = method_code.split("\n")  
    #         total_chars = sum(len(line.strip()) for line in lines)  

    #         total_line_lengths += total_chars  
    #         total_lines += len(lines)  

    #     # Compute averages
    #     average_line_count = total_line_count / total_methods if total_methods > 0 else 0
    #     average_line_length = total_line_lengths / total_lines if total_lines > 0 else 0

    #     print(f"\nFinal Average Method Length: {average_line_count:.2f} lines")
    #     print(f"Final Average Line Length: {average_line_length:.2f} characters per line")

    # except FileNotFoundError:
    #     print("Error: The file was not found.")
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding JSON: {e}")
    # except Exception as e:
    #     print(f"Unexpected error: {e}")

    #--------------
    import json
    import math

    # json_file_path = "/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/final_methods.json"
    json_file_path = "/Users/-hm/Desktop/Research/LLM_Summarization/Gemini_Summarization/GeminiSummarizationStudy/filtered_methods.json"
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(len(data))

        if not isinstance(data, list):
            raise ValueError("JSON data is not a list of methods.")

        line_counts = []
        line_lengths = []
        complexities = []

        for method_data in data:
            # Method lines
            line_count = method_data.get("line_count", 0)
            line_counts.append(line_count)

            # Line length per line of code
            method_code = method_data.get("method", "").strip()
            lines = method_code.split("\n")
            line_lengths.extend([len(line.strip()) for line in lines])

            # Cyclomatic complexity (grab first value from dict)
            cc = method_data.get("cyclomatic_complexity")
            if isinstance(cc, int) or isinstance(cc, float):
                complexities.append(cc)

        def compute_stats(values):
            if not values:
                return (0, 0, 0, 0)
            mean = sum(values) / len(values)
            std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            return (mean, std, min(values), max(values))

        avg_len, std_len, min_len, max_len = compute_stats(line_counts)
        avg_line, std_line, min_line, max_line = compute_stats(line_lengths)
        avg_cc, std_cc, min_cc, max_cc = compute_stats(complexities)

        print("\n--- Method Length (in lines) ---")
        print(f"Mean: {avg_len:.2f}, Std: {std_len:.2f}, Min: {min_len}, Max: {max_len}")

        print("\n--- Line Length (in characters) ---")
        print(f"Mean: {avg_line:.2f}, Std: {std_line:.2f}, Min: {min_line}, Max: {max_line}")

        print("\n--- Cyclomatic Complexity ---")
        print(f"Mean: {avg_cc:.2f}, Std: {std_cc:.2f}, Min: {min_cc}, Max: {max_cc}")

    except FileNotFoundError:
        print("Error: The file was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # total_line_count = 0
    # total_methods = len(data)

    # for i, method_data in enumerate(data):
    #     line_count = method_data.get("line_count", 0)
    #     print(f"Method {i + 1} Line Count: {line_count}")  # Debugging line
    #     total_line_count += line_count  

    # average_line_count = total_line_count / total_methods if total_methods > 0 else 0
    # print(f"\nFinal Average Line Count: {average_line_count:.2f} lines")
