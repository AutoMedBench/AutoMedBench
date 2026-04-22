"""BUG-043 / BUG-040: recover XML-wrapped tool_calls from assistant content."""

from __future__ import annotations

from tool_call_recovery import recover_tool_calls


# --- minimax dialect (BUG-043) ---------------------------------------------

MINIMAX_SAMPLE = """
Here is my next action.

<minimax:tool_call>
<invoke name="execute_code">
<parameter name="language">python</parameter>
<parameter name="code">import re
def postprocess(raw: str) -> str:
    return raw.strip().lower()
</parameter>
</invoke>
</minimax:tool_call>
"""


def test_recover_minimax_invoke():
    recovered = recover_tool_calls(MINIMAX_SAMPLE)
    assert len(recovered) == 1
    tc = recovered[0]
    assert tc["name"] == "execute_code"
    assert tc["arguments"]["language"] == "python"
    assert "def postprocess" in tc["arguments"]["code"]
    assert tc["id"].startswith("recovered_")


def test_recover_minimax_multiple_invokes():
    content = MINIMAX_SAMPLE + "\n" + MINIMAX_SAMPLE
    recovered = recover_tool_calls(content)
    assert len(recovered) == 2


# --- qwen dialect (BUG-040 degraded format) --------------------------------

QWEN_SAMPLE = """
<tool_call>
<function=execute_code>
<parameter=language>
python
</parameter>
<parameter=code>
import os
os.listdir('.')
</parameter>
</function>
</tool_call>
"""


def test_recover_qwen_function_tag():
    recovered = recover_tool_calls(QWEN_SAMPLE)
    assert len(recovered) == 1
    tc = recovered[0]
    assert tc["name"] == "execute_code"
    assert tc["arguments"]["language"] == "python"
    assert "os.listdir" in tc["arguments"]["code"]


# --- safety: should not false-trigger on normal content --------------------

def test_empty_content_returns_empty():
    assert recover_tool_calls("") == []
    assert recover_tool_calls(None) == []  # type: ignore[arg-type]


def test_plain_text_returns_empty():
    assert recover_tool_calls("I will now run some code.") == []


def test_claude_thinking_content_not_false_trigger():
    # Claude sometimes writes prose about tools; nothing should match.
    content = (
        "I'll call execute_code with language=python to verify the postprocess "
        "function by running a test on a sample input."
    )
    assert recover_tool_calls(content) == []


def test_truncated_invoke_recovers_tail():
    # BUG-043: max_tokens cut off before </parameter></invoke>; tail still usable.
    content = (
        "<minimax:tool_call>\n<invoke name=\"execute_code\">\n"
        "<parameter name=\"language\">python</parameter>\n"
        "<parameter name=\"code\">import os\nprint('hi')\n"
    )
    recovered = recover_tool_calls(content)
    assert len(recovered) == 1
    assert recovered[0]["name"] == "execute_code"
    assert recovered[0]["arguments"]["language"] == "python"
    assert "import os" in recovered[0]["arguments"]["code"]


def test_truncated_function_tag_recovers_tail():
    # BUG-040: qwen degraded and hit max_tokens mid-code
    content = (
        "<tool_call>\n<function=execute_code>\n"
        "<parameter=language>python</parameter>\n"
        "<parameter=code>import sys\nsys.exit(0)\n"
    )
    recovered = recover_tool_calls(content)
    assert len(recovered) == 1
    assert recovered[0]["name"] == "execute_code"
    assert "sys.exit" in recovered[0]["arguments"]["code"]


# --- mixed content tolerance ----------------------------------------------

def test_prose_mentioning_invoke_tag_is_rejected():
    """Prose describing a tool call inline must not be recovered as one."""
    content = (
        "Please call <invoke name=\"execute_code\"> with "
        "<parameter name=\"code\">x = 1</parameter> inside the wrapper."
    )
    assert recover_tool_calls(content) == []


def test_prose_mentioning_function_tag_is_rejected():
    content = (
        "The agent might do <function=execute_code> then pass "
        "<parameter=code>print('hi')</parameter> as usual."
    )
    assert recover_tool_calls(content) == []


def test_bare_invoke_at_line_start_is_accepted():
    """No wrapper but tag starts its own line — legitimate truncated emission."""
    content = (
        "<invoke name=\"execute_code\">\n"
        "<parameter name=\"code\">print('ok')</parameter>\n"
        "</invoke>\n"
    )
    recovered = recover_tool_calls(content)
    assert len(recovered) == 1
    assert recovered[0]["name"] == "execute_code"


def test_content_with_surrounding_prose():
    content = (
        "Let me retry with a fresh approach.\n"
        + MINIMAX_SAMPLE
        + "\nThat should do it."
    )
    recovered = recover_tool_calls(content)
    assert len(recovered) == 1
    assert recovered[0]["name"] == "execute_code"
