import re
import py_compile

with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Fix Boss's Orders
text = text.replace("Boss\\'\\s*Orders", "Boss's Orders")
text = text.replace("Boss\\'s Orders", "Boss's Orders")
text = text.replace("Boss'\\s*Orders", "Boss's Orders")

# Fix quad quotes
text = text.replace('""""', '"""')
text = text.replace('"""""', '"""')

# Fix any weird single/double quotes in docstrings
lines = text.split('\n')
clean_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('"""') and stripped.count('"""') == 1 and not stripped.endswith('"""'):
        pass
    elif stripped.startswith('""') and not stripped.startswith('"""'):
        line = line.replace('""', '"""')
    elif stripped.endswith('""""'):
        line = line.replace('""""', '"""')
    clean_lines.append(line)

text = '\n'.join(clean_lines)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(text)

try:
    py_compile.compile('main.py', doraise=True)
    print('SUCCESS: main.py compiles cleanly with 0 syntax errors!')
except py_compile.PyCompileError as e:
    print(f'Compilation error: {e}')
