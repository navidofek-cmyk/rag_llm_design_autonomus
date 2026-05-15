#!/usr/bin/env bash
# PreToolUse hook — bezpečnostní sandbox
# Blokuje vše co by mohlo poškodit systém nebo jít mimo projekt

set -euo pipefail

CMD="${CLAUDE_TOOL_INPUT_COMMAND:-}"
PROJECT_DIR="$(pwd)"

# ── 1. DESTRUKTIVNÍ OPERACE ──────────────────────────────────────────────────
# Povoleno: rm -rf /tmp/* (cleanup po git clone)
# Blokováno: rm mimo /tmp
if echo "$CMD" | grep -qE 'rm\s+.*-rf|rm\s+--force|shred|truncate|dd\s+if=' && \
   ! echo "$CMD" | grep -qE 'rm\s+-rf\s+/tmp/'; then
  echo "BLOCKED [safety]: destruktivní operace mimo /tmp: $CMD"
  exit 2
fi

# ── 2. PRIVILEGE ESCALATION ─────────────────────────────────────────────────
if echo "$CMD" | grep -qE '^\s*(sudo|su\s|pkexec|doas)'; then
  echo "BLOCKED [safety]: privilege escalation: $CMD"
  exit 2
fi

# ── 3. PRÁCE MIMO PROJEKT ────────────────────────────────────────────────────
if echo "$CMD" | grep -qE 'cd\s+(\/(?!tmp)(?!home/ivand/projects/learning_python)|~\s*$|\.\./)'; then
  echo "BLOCKED [safety]: cd mimo projekt: $CMD"
  exit 2
fi

# ── 4. GIT DESTRUKTIVNÍ ──────────────────────────────────────────────────────
if echo "$CMD" | grep -qE 'git\s+(push|reset\s+--hard|clean\s+-f|rebase|merge\b)'; then
  echo "BLOCKED [safety]: destruktivní git operace: $CMD"
  exit 2
fi

# ── 5. SÍŤOVÝ PŘÍSTUP ────────────────────────────────────────────────────────
# Povoleno: localhost, github.com pro git clone, ollama.com
if echo "$CMD" | grep -qE 'curl|wget|nc\s|netcat'; then
  if ! echo "$CMD" | grep -qE 'localhost|127\.0\.0\.1|github\.com|ollama\.com'; then
    echo "BLOCKED [safety]: nepovolený síťový přístup: $CMD"
    exit 2
  fi
fi

# ── 6. ZÁPIS MIMO PROJEKT ────────────────────────────────────────────────────
if echo "$CMD" | grep -qE '>\s*\/(?!tmp)(?!home/ivand/projects/learning_python)'; then
  echo "BLOCKED [safety]: zápis mimo projekt: $CMD"
  exit 2
fi

# ── 7. GLOBÁLNÍ INSTALACE ────────────────────────────────────────────────────
if echo "$CMD" | grep -qE 'pip\s+install\s+--user|uv\s+add\s+--global|npm\s+install\s+-g'; then
  echo "BLOCKED [safety]: globální instalace: $CMD"
  exit 2
fi

# ── 8. NEBEZPEČNÉ SYSTÉMOVÉ PŘÍKAZY ─────────────────────────────────────────
if echo "$CMD" | grep -qE 'mkfs|fdisk|parted|crontab|systemctl|service\s'; then
  echo "BLOCKED [safety]: systémový příkaz: $CMD"
  exit 2
fi

# Vše OK
exit 0
