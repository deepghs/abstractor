import json
import os


def parse_json_from_llm_output(text: str):
    lines = []
    has_prefix = False
    for line in text.splitlines(keepends=False):
        line = line.strip()
        if not lines and line.startswith('```'):
            has_prefix = True
        else:
            lines.append(line)

    if has_prefix and lines[-1].startswith('```'):
        lines = lines[:-1]
    return json.loads(os.linesep.join(lines))
