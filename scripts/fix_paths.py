"""
scripts/fix_paths.py
Adds robust sys.path configuration to all Streamlit page scripts.
"""

import glob
import os
import re

PREFIX = """import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""

if __name__ == "__main__":
    count = 0
    for path in glob.glob("pages/*.py"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if "ROOT_DIR = os.path.dirname" not in content:
            m = re.match(r'^("""[\s\S]*?""")\s*', content)
            if m:
                doc = m.group(1)
                rest = content[len(m.group(0)):].lstrip()
                new_content = doc + "\n\n" + PREFIX + "\n" + rest
            else:
                new_content = PREFIX + "\n" + content
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            count += 1
    print(f"[SUCCESS] Processed pages, added ROOT_DIR to {count} files.")
