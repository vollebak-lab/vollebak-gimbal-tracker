"""
Predator Project - Conversation & Troubleshooting History Exporter
Extracts full transcript history into a clean, searchable Markdown archive.
"""

import json
import os
import sys

def export_transcript():
    # Force UTF-8 stdout
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    conv_id = "2aee8499-8b3a-4340-9cdc-9fe31ae9e225"
    app_data = os.path.expanduser(r"~\.gemini\antigravity")
    log_dir = os.path.join(app_data, "brain", conv_id, ".system_generated", "logs")
    log_path = os.path.join(log_dir, "transcript_full.jsonl")

    if not os.path.exists(log_path):
        log_path = os.path.join(log_dir, "transcript.jsonl")
        if not os.path.exists(log_path):
            print(f"[ERROR] Could not find transcript at {log_dir}")
            return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(repo_root, "docs", "conversation_history_full_archive.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    user_turns = 0
    assistant_turns = 0

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f, open(out_path, 'w', encoding='utf-8') as out:
        out.write('# Predator Neuromorphic Drone Detection — Complete Engineering & Troubleshooting Archive\n\n')
        out.write('**Generated**: 2026-10-02\n')
        out.write('**Source**: Antigravity Full Session Transcript\n')
        out.write(f'**Conversation ID**: `{conv_id}`\n\n')
        out.write('---\n\n')

        turn_idx = 1
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue

            stype = entry.get('type', '')
            source = entry.get('source', '')
            created_at = entry.get('created_at', '')
            content = entry.get('content', '')

            if stype == 'USER_INPUT' or source == 'USER_EXPLICIT':
                if content and not content.startswith('<CONTEXT_SUMMARY>') and not content.startswith('<ADDITIONAL_METADATA>'):
                    clean_content = content.replace('<USER_REQUEST>', '').replace('</USER_REQUEST>', '').strip()
                    out.write(f'## Turn {turn_idx}: User Request\n')
                    out.write(f'*Timestamp: {created_at}*\n\n')
                    out.write(f'{clean_content}\n\n')
                    out.write('---\n\n')
                    turn_idx += 1
                    user_turns += 1

            elif stype == 'PLANNER_RESPONSE' and source == 'MODEL':
                if content and content.strip():
                    out.write(f'### Assistant Analysis & Implementation Response\n')
                    out.write(f'*Timestamp: {created_at}*\n\n')
                    out.write(f'{content.strip()}\n\n')
                    out.write('---\n\n')
                    assistant_turns += 1

    print(f'[SUCCESS] Exported {user_turns} user requests and {assistant_turns} responses.')
    print(f'[SAVED] {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)')

if __name__ == '__main__':
    export_transcript()
