#!/usr/bin/env python3
"""PathGraph AI - Interactive Terminal Demo Launcher.

Powers the PathGraph AI system in an interactive terminal environment.
Demonstrates ESCO occupation search, skill resolution, gap analysis,
semantic candidate surfacing, and Kahn's topological learning path generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path so it works out of the box
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pathgraph_ai.cli.demo import main

if __name__ == "__main__":
    main()
