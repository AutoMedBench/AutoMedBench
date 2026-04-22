"""medbench_vqa — Python helpers available inside the agent sandbox.

Agents ``from medbench_vqa import inspect_image, public_medical_search, submit_answer``.

These helpers cover the three auxiliary capabilities that previously existed as
separate LLM function-call tools in eval_vqa V1.  In eval_vqa_v2 the LLM only
sees ``execute_code``; these helpers are invoked from inside agent-authored
Python.  Every call is logged to ``$WORKSPACE_DIR/tool_calls.jsonl`` so the
runner's artefact scorer can attribute tool usage to S1-S5 phases.
"""

from .image_inspection import inspect_image
from .medical_search import public_medical_search
from .submit import submit_answer

__all__ = ["inspect_image", "public_medical_search", "submit_answer"]
