import pandas as pd
import re

# ==== CONFIG ====
INPUT_FILE = 'unique_aois_from_columns_3.csv'  # Your raw AOI labels
OUTPUT_FILE = 'aoi_to_category_mapping_3.csv'  # Output to review/edit
# ================

# Load AOI labels
df = pd.read_csv(INPUT_FILE)
aoi_labels = df['AOI_Label'].dropna().astype(str).tolist()

def categorize_aoi(aoi):
    name = aoi.lower()

    # Clean out "unnamed" and blanks
    if 'unnamed' in name or name.strip() == '':
        return 'Other'
    
    if 'gem' in name:
        return 'Gemini'
    
    if 'sum' in name or 'ummary' in name:
        return 'Summary'

    # Argument
    if 'arg' in name or 'aq' in name or 'a4r' in name or 'arr' in name or 'art' in name:
        return 'Argument'
    
    #Parameter
    if 'param' in name:
        return 'Parameter'

    # Assignment
    if re.search(r'assign|asign|asgn|asssign|assing|assignment|ass', name):
        return 'Assignment'
    
    # Conditional Body
    if 'bod' in name and 'co' in name:
        return 'ConditionalBody'

    # Conditional Statement
    if re.search(r'cond|condition|condit|if_stmt|conmd|codn|cons|coond', name):
        return 'ConditionalStatement'

    # Loop Body
    if 'bod' in name and 'loo' in name or 'loopbd' in name:
        return 'LoopBody'
    
    # Loop Body
    if re.search(r'loop|loopsta|loopstae|loopbody|looop', name):
        return 'LoopStatement'

    # Method Declaration
    if re.search(r'method\s?dec|meth[do]dec|declaration|methdodec|methdoec|method_dec|methodddec', name):
        return 'MethodDeclaration'

    # Method Call
    if re.search(r'methodcall|method cal|methocall|methdocall|methocall|methodca|math|meth|med|mehod', name):
        return 'MethodCall'

    # Return
    if 'return' in name or 'retrun' in name:
        return 'Return'

    # Exception Handling
    if 'except' in name or 'exception' in name or 'excpet' in name or 'exept' in name:
        return 'ExceptionHandling'

    # Variable Name
    if 'var' in name or 'vai' in name or 'vaar' in name:
        return 'VariableName'

    # External Variable or Method
    if re.search(r'external|exter|externav|externalvar|externalmethod|ext|xtern', name):
        return 'ExternalVarOrMethod'

    # External Class
    if 'external class' in name or name.strip() == 'external':
        return 'ExternalClass'

    # Literals
    if 'lit' in name:
        return 'Literal'

    # Operator
    if 'operator' in name or 'oporator' in name:
        return 'Operator'

    # Index Operation
    if 'index' in name or 'inex' in name:
        return 'IndexOperation'

    # Declaration (standalone, if not already captured above)
    if name.strip() == 'declaration':
        return 'Declaration'

    # Code (catch-all for general code region)
    if name == 'code' or 'code variable' in name:
        return 'Code'

    return 'Other'

# Apply categorization
df['SemanticCategory'] = df['AOI_Label'].apply(categorize_aoi)

# Save result
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved categorized AOIs to {OUTPUT_FILE}")
