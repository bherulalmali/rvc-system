import re

content_config = """
    # some comments
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
"""

# Pattern 1 from Notebook
# r'([ \t]+\w+)[ \t]*:[ \t]*([\\w\\.]+)[ \t]*=[ \t]*([\\w\\.]+)\\([ \t]*\\)'
# Converted to Python string (removing double backslashes for \w which become just \w)
# But wait, in the notebook string literal logic:
# Notebook JSON: "pattern1 = r'([ \\t]+\\w+)[ \\t]*:[ \\t]*([\\w\\.]+)[ \\t]*=[ \\t]*([\\w\\.]+)\\([ \\t]*\\)'"
# Python runtime sees:
pattern1 = r'([ \t]+\w+)[ \t]*:[ \t]*([\w\.]+)[ \t]*=[ \t]*([\w\.]+)\([ \t]*\)'

print(f"Pattern 1: {pattern1}")
matches1 = re.findall(pattern1, content_config)
print(f"Matches for Pattern 1: {matches1}")

# Group Ref Error Test
content_heal = "some code field(default_factory=EncDecBaseConfig)()"
heal_pattern = r'field\(default_factory=([\w\.]+)\)\(\)'
print(f"\nHeal Pattern: {heal_pattern}")

try:
    # Reproducing: content = re.sub(heal_pattern, r'field(default_factory=\1)', content)
    # Note: \1 in raw string
    fixed = re.sub(heal_pattern, r'field(default_factory=\1)', content_heal)
    print(f"Healed: {fixed}")
except Exception as e:
    print(f"Heal Error: {e}")
