import pandas as pd
import json
import re

# Extract the *last* JSON block only
def extract_last_json(text):
    matches = re.findall(r"<json>(.*?)</json>", text, re.DOTALL)
    if not matches:
        return None
    last_block = matches[-1]  
    try:
        return json.loads(last_block)
    except json.JSONDecodeError:
        return None