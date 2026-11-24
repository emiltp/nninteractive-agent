"""Text-prompted agentic segmentation with nnInteractive.

The package is split so the core never imports Qt or napari:

* ``core``: image I/O, the nnInteractive session, prompt rasterization, metrics and
  off-screen rendering, tied together by ``AgentWorkspace``.
* ``tools``: the agent-facing tool set (names, JSON schemas, handlers) on top of a workspace.
* ``mcp_server``: serves the tools over stdio MCP for Claude Code.
"""

import os
from importlib.metadata import PackageNotFoundError, version

# Apple GPUs: PyTorch has no adaptive_avg_pool3d on MPS (nnInteractive's AutoZoom uses it), so let it run
# on the CPU. PyTorch reads this when it is imported, so it has to be set here, before that.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    __version__ = version("nninteractive-agent")
except PackageNotFoundError:
    __version__ = "unknown"
