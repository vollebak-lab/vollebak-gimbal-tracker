import json
import os
import sys

transcript_path = r"C:\Users\snowd\.gemini\antigravity\brain\a20f25c8-5ec7-42f5-8306-4730fa85739a\.system_generated\logs\transcript_full.jsonl"
output_md_path = r"C:\Users\snowd\OneDrive\Documents\Vollebak\predator\docs\Full_Conversation_Transcript.md"

if not os.path.exists(transcript_path):
    print(f"Error: Transcript file not found at {transcript_path}")
    sys.exit(1)

print(f"Reading transcript from: {transcript_path}")

md_lines = [
    "# Predator Neuromorphic Drone Propeller Detection — Complete Conversation & Troubleshooting Transcript\n",
    f"**Export Date**: {sys.version.split()[0]}  \n",
    "**Conversation ID**: `a20f25c8-5ec7-42f5-8306-4730fa85739a`  \n",
    "**Hardware**: NVIDIA Jetson Orin Nano Developer Kit (8GB RAM, IP `10.0.0.34`)  \n",
    "**Sensor**: IDS UE-39B0XCP-E (Sony IMX636 1280x720 Neuromorphic Camera)  \n",
    "**Lens**: Edmund Optics 8mm $f/8$ BLUE Series M12 (#27052)  \n\n",
    "---\n\n"
]

turn_num = 1
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception as e:
            continue
        
        step_type = entry.get("type", "")
        source = entry.get("source", "")
        content = entry.get("content", "")
        created_at = entry.get("created_at", "")
        
        if step_type == "USER_INPUT":
            md_lines.append(f"## 👤 Turn {turn_num} — User Request ({created_at})\n\n")
            md_lines.append(f"{content}\n\n")
            md_lines.append("---\n\n")
            turn_num += 1
        elif step_type == "PLANNER_RESPONSE" and content:
            # Check if there's actual text content to the user
            clean_content = content.strip()
            if clean_content:
                md_lines.append(f"### 🤖 Antigravity Response\n\n")
                md_lines.append(f"{clean_content}\n\n")
                md_lines.append("---\n\n")

with open(output_md_path, "w", encoding="utf-8") as out:
    out.writelines(md_lines)

print(f"Successfully generated full conversation transcript: {output_md_path} ({len(md_lines)} sections)")
