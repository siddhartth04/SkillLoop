"""SkillLoop — test-gated skill learning for any agent."""
from .core import SkillLoop
from .record import Session, record_tool
from .schema import Trace, Step, ToolCall, Signal, normalize
from .gate import Policy

__version__ = "0.3.0"
__all__ = ["SkillLoop", "Session", "record_tool", "Trace", "Step", "ToolCall", "Signal", "normalize", "Policy"]
