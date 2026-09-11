ID: ISSUE-GH-1480
Title: agent-integration cannot execute agent-preflight.sh on native Windows
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: UNKNOWN
GitHub: 1480
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> The unchanged-head integration gate uv run scripts/agent-integration.py --base origin/main fails before repository checks on native Windows. scripts/agent-ready.py:145 passes scripts/agent-preflight.sh directly to subprocess, which raises OSError: [WinError 193] %1 is not a valid Win32 application.\n\nThis is a protocol/checker portability defect, not a DFlash2 model defect. Owner: ENG-RELEASE-WINDOWS. A repair needs its own spec or an existing Windows-release spec ## Owed, a red-before subprocess test, green-after evidence, and fresh review. The checker must invoke an available shell explicitly or provide a Windows-native entry point without weakening any preflight assertion.

## Resolution

-
