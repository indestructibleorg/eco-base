# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: fix_event_comparison
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""Fix Event comparison issue in event_driven_system.py"""
import re
# Read the file
with open('event_driven_system.py', 'r', encoding='utf-8') as f:
    content = f.read()
# Find the Event class and add __lt__ method
# We'll look for the __init__ method and add __lt__ after it
pattern = r'(self\.event_id = hashlib\.md5\(f"[^"]+:\{self\.timestamp\.isoformat\(\)\}"\.encode\(\)\)\.hexdigest\(\)\[:8\])'
replacement = r'''\1
    def __lt__(self, other):
        """支持事件比较，用于优先级队列"""
        if not isinstance(other, Event):
            return NotImplemented
        return self.priority < other.priority'''
new_content = re.sub(pattern, replacement, content)
# Write back
with open('event_driven_system.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("✅ Event comparison fix applied successfully!")
print("The __lt__ method has been added to the Event class.")