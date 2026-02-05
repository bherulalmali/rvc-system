import re

file_path = "temp_utils.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

pattern1 = r'(\s+\w+)\s*:\s*([\w\.]+)\s*=\s*([\w\.]+)\(\s*\)'
matches1 = list(re.finditer(pattern1, content))

print(f"--- Found {len(matches1)} Matches ---")
for i, m in enumerate(matches1):
    print(f"Match {i+1}: '{m.group(0)}'")
    print(f"  G1 (Var): '{m.group(1)}'")
    print(f"  G2 (Type): '{m.group(2)}'")
    print(f"  G3 (Class): '{m.group(3)}'")
