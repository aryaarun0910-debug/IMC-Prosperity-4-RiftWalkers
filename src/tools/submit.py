"""
Submit Script — Strip trader.py for IMC submission.

Pipeline:
  1) Tokenize with Python's tokenize module (Python 3.12 f-string aware)
  2) Drop comments and module/function/class docstrings
  3) Re-emit tokens with collapsed spacing (single space only when alphanumerics
     would otherwise concatenate into a different identifier/keyword)
  4) Track INDENT/DEDENT depth and emit 1-space per level

Validates: parses, contains Trader class, runs short import smoke-test.

Usage:
    py -3.12 tools/submit.py
    py -3.12 tools/submit.py --check
    py -3.12 tools/submit.py --output my_sub.py
"""

import sys
import os
import ast
import tokenize
import io

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TOOLS_DIR)
sys.path.insert(0, _ROOT_DIR)

TRADER_PATH = os.path.join(_ROOT_DIR, "trader.py")
DEFAULT_OUTPUT = os.path.join(_ROOT_DIR, "trader_submit.py")
MAX_SIZE_KB = 100

# Tokens we deliberately drop / handle specially
_DROP = {
    tokenize.COMMENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.NL,           # blank-line / continuation NL
}

# Keywords after which we MUST preserve a trailing space, even if the next
# token starts with a non-alphanumeric character. Without this, e.g.
# "from .module import x" would collapse to "from.module import x" which
# is a SyntaxError.
_SPACE_AFTER = {"from", "import", "as", "in", "is", "not", "and", "or",
                "if", "elif", "else", "while", "for", "return", "yield",
                "raise", "del", "global", "nonlocal", "assert", "lambda",
                "with", "except", "class", "def", "await", "async", "pass"}


def _is_word_char(ch):
    return ch.isalnum() or ch == "_"


def strip_trader(source_path):
    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    # First pass: identify docstring tokens to drop.
    # A STRING token is a docstring iff it's the first statement of a module,
    # function, or class body. We approximate: any STRING token that appears
    # immediately after INDENT or NEWLINE/NL at the start of a logical line.
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))

    is_docstring = [False] * len(tokens)
    # Track when we just entered a new block (after `:` then NEWLINE then INDENT)
    # OR when we are at module top (start of file).
    awaiting_first_stmt = True
    for i, tok in enumerate(tokens):
        ttype = tok.type
        if ttype == tokenize.INDENT:
            awaiting_first_stmt = True
            continue
        if ttype == tokenize.NEWLINE:
            # After NEWLINE, the next non-trivial token is start of next stmt.
            # Only a NEW block awaits a docstring; ordinary stmts don't.
            continue
        if ttype in (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING):
            continue
        # First real token after INDENT (or at module top): if it's a STRING
        # followed by NEWLINE, it's a docstring.
        if awaiting_first_stmt and ttype == tokenize.STRING:
            # Confirm it's a standalone expr statement: next non-trivial
            # token is NEWLINE or another STRING (implicit concat). Either
            # way, mark this token a docstring (replace with `0`).
            is_docstring[i] = True
        awaiting_first_stmt = False
        # A new function/class body opens at a `:` at end of a line.
        if ttype == tokenize.OP and tok.string == ":":
            # Could be slice / dict / annotation — but harmless: we just
            # rearm awaiting_first_stmt; the next INDENT confirms a real
            # block, and only INDENT actually triggers docstring detection.
            awaiting_first_stmt = False

    out_parts = []
    indent_level = 0
    line_started = True       # True when next emit is at start of a new line
    prev_emitted_text = ""    # last non-whitespace text emitted (for spacing rule)
    paren_depth = 0           # ()/[]/{} depth — inside, NL is meaningless

    def emit(text):
        nonlocal prev_emitted_text, line_started
        if not text:
            return
        out_parts.append(text)
        prev_emitted_text = text
        line_started = False

    for i, tok in enumerate(tokens):
        ttype = tok.type
        tstr = tok.string

        if ttype in _DROP:
            continue
        if ttype == tokenize.COMMENT:
            continue
        if ttype == tokenize.INDENT:
            indent_level += 1
            continue
        if ttype == tokenize.DEDENT:
            indent_level = max(0, indent_level - 1)
            continue
        if ttype == tokenize.NEWLINE:
            if not line_started:
                out_parts.append("\n")
                line_started = True
                prev_emitted_text = ""
            continue
        if ttype == tokenize.NL:
            # NL is non-logical (inside parens, blank lines). Drop entirely;
            # we already collapse blank lines. Inside parens this is safe
            # because the tokenizer will not require a NEWLINE there.
            continue
        if not tstr:
            continue

        # Replace docstrings with `0` (a no-op constant statement).
        if is_docstring[i]:
            tstr = "0"

        # Beginning of a new line — inject indentation (1 space per level).
        if line_started:
            if indent_level > 0:
                out_parts.append(" " * indent_level)
            line_started = False
            prev_emitted_text = ""

        # Spacing rule: between two non-trivial tokens, insert one space if
        # collapsing them would form a different identifier/number/keyword,
        # i.e. when last char of prev and first char of curr are BOTH
        # word-chars. Also force a trailing space after listed keywords.
        need_space = False
        if prev_emitted_text:
            last = prev_emitted_text[-1]
            first = tstr[0]
            if _is_word_char(last) and _is_word_char(first):
                need_space = True
            elif prev_emitted_text in _SPACE_AFTER:
                need_space = True

        if need_space:
            out_parts.append(" ")
        emit(tstr)

        # Track paren depth (informational only; NL handling above already safe)
        if ttype == tokenize.OP:
            if tstr in ("(", "[", "{"):
                paren_depth += 1
            elif tstr in (")", "]", "}"):
                paren_depth = max(0, paren_depth - 1)

    result = "".join(out_parts)
    # Ensure final newline.
    if not result.endswith("\n"):
        result += "\n"
    # Compact: drop runs of pure-blank lines (defensive — should not occur)
    out_lines = []
    for line in result.split("\n"):
        if line.strip() or out_lines and out_lines[-1] != "":
            out_lines.append(line)
    result = "\n".join(out_lines)
    if not result.endswith("\n"):
        result += "\n"
    return result


def validate_stripped(stripped_source):
    try:
        compile(stripped_source, "<stripped>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    tree = ast.parse(stripped_source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)

    if "Trader" not in names:
        return False, "Missing 'Trader' class"
    return True, "OK"


def smoke_import(stripped_source, output_path):
    """Write to a temp module, import it, instantiate Trader."""
    import importlib.util
    spec = importlib.util.spec_from_loader("trader_stripped_smoke", loader=None)
    mod = importlib.util.module_from_spec(spec)
    try:
        exec(compile(stripped_source, output_path, "exec"), mod.__dict__)
        T = getattr(mod, "Trader", None)
        if T is None:
            return False, "Trader class missing after exec"
        T()  # ensure constructor works
        return True, "exec+instantiate OK"
    except Exception as e:
        return False, f"exec error: {type(e).__name__}: {e}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Strip trader.py for submission")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Check size only")
    parser.add_argument("--source", type=str, default=TRADER_PATH)
    args = parser.parse_args()

    source_path = args.source
    if not os.path.exists(source_path):
        print(f"ERROR: {source_path} not found")
        sys.exit(1)

    orig_size = os.path.getsize(source_path)
    print(f"Original: {source_path}")
    print(f"  Size: {orig_size:,} bytes ({orig_size/1024:.1f} KB)")

    stripped = strip_trader(source_path)
    stripped_size = len(stripped.encode("utf-8"))
    reduction = (1 - stripped_size / orig_size) * 100

    print(f"\nStripped:")
    print(f"  Size: {stripped_size:,} bytes ({stripped_size/1024:.1f} KB)")
    print(f"  Reduction: {reduction:.1f}%")

    if stripped_size > MAX_SIZE_KB * 1024:
        over = stripped_size - MAX_SIZE_KB * 1024
        print(f"\n  WARNING: Exceeds {MAX_SIZE_KB}KB limit!")
        print(f"  Over by: {over:,} bytes ({over/1024:.1f} KB)")
    else:
        headroom = MAX_SIZE_KB * 1024 - stripped_size
        print(f"  Under limit with {headroom:,} bytes ({headroom/1024:.1f} KB) headroom")

    valid, msg = validate_stripped(stripped)
    print(f"\nValidation: {'PASS' if valid else 'FAIL'} -- {msg}")
    if not valid:
        sys.exit(1)

    smoke_ok, smoke_msg = smoke_import(stripped, args.output)
    print(f"Smoke:      {'PASS' if smoke_ok else 'FAIL'} -- {smoke_msg}")
    if not smoke_ok:
        sys.exit(1)

    if args.check:
        return

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(stripped)
    print(f"\nWritten to: {args.output}")
    line_count = len(stripped.split("\n"))
    print(f"Lines: {line_count}")


if __name__ == "__main__":
    main()
