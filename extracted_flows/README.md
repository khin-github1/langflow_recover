# Langflow Recovery — Results

Recovered from `langflow_psql_vol.tar.gz` (Postgres 18 data directory) by booting a
throwaway `postgres:18.0` container over the backup in a local Docker volume and
exporting directly from the `flow` table. **Nothing connected to the old/compromised
server or the network.** The Alembic boot error was irrelevant here — that comes from
the Langflow *app*; we only ran raw Postgres and read the tables.

## What was recovered

| Item | Count | Location |
|------|-------|----------|
| Flows (full graph JSON) | **175** | `flows/` |
| Unique component code blobs | **312** | `components/` |
| — likely custom (your code) | **267** | `components/custom/` |
| — stock Langflow components | **45** | `components/builtin/` |
| Standalone components (`is_component=true`) | 0 | — |

All custom component source was embedded in the flows at
`node.data.node.template.code.value` (exactly as expected). Code was deduplicated by
content hash — where you evolved a component across flows (e.g.
`OllamaVerificationFromAnalyzer` exists at two sizes), **each distinct version is kept**.

## Folder layout

- `flows/` — one `flow_<name>_<id8>.json` per flow. This is the full record (metadata + graph).
- `components/custom/` — your thesis components, one `.py` each. Each file has a header
  comment listing the component type, class name, and which flows use it.
- `components/builtin/` — stock Langflow components (kept for completeness/reference).
- `standalone_components/` — flows saved as reusable components (none in this DB).
- `manifest.csv` — index of every code blob: file, type, class, likely_custom, num_flows, size.
- `../_dbdump/flows.ndjson` — the raw durable dump (all 175 flows, 66 MB). Source of truth
  if you ever want to re-extract differently.

## Verification

- 264/267 custom `.py` files compile under Python 3.8. The 3 that don't
  (`LLMRouterComponent`, `LLMSelectorComponent`, `parser`) are **not truncated** — they
  use `match` statements / modern `aiohttp` (Python 3.10+, which Langflow runs). They are
  complete.

## ⚠️ Forensic notes (server was compromised)

- **`_h` table** in the `langflow` DB is an attacker artifact — a single text column
  holding the output of the Unix `id` command (`uid=999(postgres)…`). This is a
  proof-of-concept marker showing the attacker gained **command execution as the
  `postgres` OS user** through the database (e.g. `COPY … FROM PROGRAM`). Not your data.
- **`langflow/app/src/.ak_token.py`** (2781 bytes) — a hidden attacker script, **do not
  run it**. It is a lateral-movement / persistence tool that:
    - targets the AIT Authentik SSO server `https://authen.brain.cs.ait.ac.th` (SSL
      verification disabled);
    - logs in with hardcoded credentials — username `bci`, password `bc1lab@AIT`;
    - mints a **non-expiring API token** named `vscode-sync-bci` (intent=api) and reads
      back its key (`view_key`) to exfiltrate it.
  This means the attacker used the Langflow/Postgres RCE to pivot to your institution's
  identity provider and establish persistent API access.

### Immediate actions (identity compromise — do these first)
1. **Change the `bci` account password now** (`bc1lab@AIT` is burned).
2. In Authentik, **revoke/delete the API token `vscode-sync-bci`** (Directory → Tokens),
   and audit for any other tokens you don't recognize.
3. Check Authentik **event logs** for that token's creation time and every request that
   used it, to scope what was accessed.
4. **Notify AIT IT/security** — this targets shared university SSO infrastructure, not
   just your project.
5. Rotate any other credentials/tokens that lived on the compromised server.
