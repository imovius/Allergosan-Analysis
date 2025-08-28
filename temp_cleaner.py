import re

with open('customer_segmentation_workflow.py', 'r') as f:
    lines = f.readlines()

# Very simple heuristic: if a line doesn't start with a common Python keyword,
# a variable name from the script, a comment, or indentation, it's likely markdown.
# A more robust solution would be a proper parser, but this should fix the current issue.
code_keywords = ['import', 'from', 'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'print', 'return', 'assert']
known_vars = ['df', 'X_famd', 'np', 'pd', 'plt', 'sns', 'famd', 'kmeans', 'profile', 'segment']

cleaned_lines = []
for line in lines:
    stripped_line = line.strip()
    if not stripped_line: # keep empty lines
        cleaned_lines.append(line)
        continue

    # Check if it looks like code
    is_code = False
    if stripped_line.startswith('#'):
        is_code = True
    else:
        for keyword in code_keywords:
            if stripped_line.startswith(keyword):
                is_code = True
                break
        if not is_code:
            for var in known_vars:
                if stripped_line.startswith(var):
                    is_code = True
                    break
    
    # If it's not identified as code and not a continuation of a multiline statement
    if not is_code and not line.startswith(' '):
         # More complex check for markdown-like lines
        if re.match(r'^[A-Za-z\s\-*`#\[\]\d\.]+$', stripped_line) and not re.search(r'[=+\-*/%()]', stripped_line):
             # It's likely markdown if it contains patterns like '##' or '- ' and no operators
            if stripped_line.startswith(('###', '-', '*', '1.', '2.', '3.', '4.')) or '`' in stripped_line:
                 cleaned_lines.append(f"# {line}")
            else:
                 cleaned_lines.append(line) # Assume it's a multiline string or similar
        else:
            cleaned_lines.append(line)
    else:
        cleaned_lines.append(line)

with open('customer_segmentation_workflow_cleaned.py', 'w') as f:
    f.writelines(cleaned_lines)

print("Cleaned script created.")