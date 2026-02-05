import re

content = """
@dataclass
class AdaptiveLossConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    ddp_backend: DDP_BACKEND_CHOICES = II("distributed_training.ddp_backend")
"""

pattern1 = r'([ \t]+\w+)[ \t]*:[ \t]*([\w\.]+)[ \t]*=[ \t]*([\w\.]+)\([ \t]*\)'
matches = re.findall(pattern1, content)
print(f"Matches: {matches}")
