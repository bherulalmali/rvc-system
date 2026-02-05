import re

content = """
    # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise: QuantNoiseConfig = field(default=QuantNoiseConfig())
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
"""

# Simulate the two patterns
pattern1 = r'([ \t]+\w+)[ \t]*:[ \t]*([\w\.]+)[ \t]*=[ \t]*([\w\.]+)\([ \t]*\)'
pattern2 = r'field\(default=([\\w\\.]+)\([ \t]*\)'

print("--- Testing Pattern 2 (field(default=...)) ---")
matches2 = re.findall(pattern2, content)
print(f"Matches: {matches2}")

def repl2(m):
    return f"field(default_factory={m.group(1)}"

new_content = re.sub(pattern2, repl2, content)
print(f"\nNew Content:\n{new_content}")
