#!/usr/bin/env python3
"""Fix corrupted emojis in app.py routing section"""

# Read the file
with open(r'c:\Users\manas\OneDrive\Desktop\model-monitoring-dashboard\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the corrupted emoji routing
# Replace the corrupted emoji characters with proper ones
content = content.replace('"� Dataset Upload"', '"📊 Dataset Upload"')
content = content.replace('"�📋 Schema Definition"', '"📋 Schema Definition"')

# Write back the fixed content
with open(r'c:\Users\manas\OneDrive\Desktop\model-monitoring-dashboard\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed emoji routing issues in app.py')
