import dataclasses


CodeAgentCFG = r"""
    start: THOUGHT START_CODE CODE* END_CODE
    
    THOUGHT: "Thought:" /.+/ _NL
    START_CODE: "Code:\n```" /(?:py|python)?/ _NL 
    CODE: /[^`]+/ _NL
    END_CODE: "```<end_code>"
    
    %import common.NEWLINE -> _NL
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

CodeAgentRegex = r'Thought: .+\.\nCode:\n```(?:py|python)?\n([^`].|\n)*\n```<end_code>'

ToolCallingAgentRegex = r'Action:\n\{\n  "name": "[^"\n]+",\n  "arguments": \{.*\}\n\}'


@dataclasses.dataclass()
class _Grammars:
    CodeAgentCFG=CodeAgentCFG
    CodeAgentRegex=CodeAgentRegex
    ToolCallingAgentRegex=ToolCallingAgentRegex
