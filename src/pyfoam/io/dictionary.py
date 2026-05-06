"""
OpenFOAM dictionary parser.

Handles the C++-like dictionary syntax used by OpenFOAM for configuration files
(controlDict, fvSchemes, fvSolution, boundary, etc.).

Syntax features:
- Key-value pairs: ``key  value;``
- Nested dictionaries: ``subdict { ... };``
- Lists: ``key  (val1 val2 val3);``
- Strings: ``"quoted string"``
- Comments: ``// line comment`` and ``/* block comment */``
- Macros: ``$variable``, ``$subdict/variable``
- Includes: ``#include "file"``
- Calculations: ``#calc ...``

Usage::

    from pyfoam.io.dictionary import parse_dict, FoamDict

    with open("controlDict") as f:
        text = f.read()
    d = parse_dict(text)
    print(d["application"])
    print(d["subdict"]["key"])
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Union

__all__ = [
    "FoamDict",
    "FoamList",
    "parse_dict",
    "parse_dict_file",
    "expand_macros",
    "Token",
    "Tokenizer",
]

# ---------------------------------------------------------------------------
# FoamDict / FoamList — rich dict/list with macro support
# ---------------------------------------------------------------------------


class FoamDict(dict):
    """An OpenFOAM dictionary with macro expansion and path lookup.

    Supports:
    - Nested dict/list access via ``d["a/b/c"]`` path syntax
    - Macro expansion via ``$variable`` and ``$subdict/var``
    - Parent-chain lookup for ``$var`` (searches up through parents)
    """

    def __init__(self, *args: Any, parent: Optional[FoamDict] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._parent = parent

    @property
    def parent(self) -> Optional[FoamDict]:
        """Return the parent dictionary (None for top-level)."""
        return self._parent

    def __missing__(self, key: str) -> Any:
        """Support path-based lookup: ``d["a/b/c"]`` navigates nested dicts."""
        if "/" in key:
            parts = key.split("/", 1)
            sub = self[parts[0]]
            if isinstance(sub, FoamDict):
                return sub[parts[1]]
        raise KeyError(key)

    def get_path(self, path: str, default: Any = None) -> Any:
        """Get a value by ``/``-separated path.

        Args:
            path: ``/``-separated key path (e.g., ``"subDict/key"``).
            default: Default value if path not found.

        Returns:
            Value at the path, or *default*.
        """
        try:
            return self[path]
        except KeyError:
            return default

    def resolve_macro(self, name: str) -> Any:
        """Resolve a ``$variable`` reference.

        Searches this dict, then parent dicts (for ``$var``).
        For ``$subdict/var``, searches only the named subdict.

        Args:
            name: Variable name (without ``$``).

        Returns:
            Resolved value.

        Raises:
            KeyError: If the variable cannot be resolved.
        """
        # Handle path-like macros: $subdict/var
        if "/" in name:
            parts = name.split("/", 1)
            sub = self.get(parts[0])
            if isinstance(sub, FoamDict):
                return sub.resolve_macro(parts[1])
            raise KeyError(f"Macro ${name}: subdict '{parts[0]}' not found")

        # Search this dict
        if name in self:
            return self[name]

        # Search parent chain
        if self._parent is not None:
            return self._parent.resolve_macro(name)

        raise KeyError(f"Macro ${name} not found")

    def set_parent(self, parent: FoamDict) -> None:
        """Set the parent dictionary."""
        self._parent = parent


class FoamList(list):
    """An OpenFOAM list with type information.

    Attributes:
        foam_type: Optional type tag (e.g., ``"vector"``, ``"scalar"``).
    """

    def __init__(self, *args: Any, foam_type: str = "", **kwargs: Any) -> None:
        super().__init__(*args)
        self.foam_type = foam_type

    def __repr__(self) -> str:
        type_str = f"<{self.foam_type}>" if self.foam_type else ""
        return f"FoamList{type_str}({list.__repr__(self)})"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Token:
    """A single token from the OpenFOAM dictionary lexer."""

    __slots__ = ("type", "value")

    # Token types
    WORD = "WORD"
    STRING = "STRING"
    NUMBER = "NUMBER"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    SEMICOLON = "SEMICOLON"
    DOLLAR = "DOLLAR"
    HASH = "HASH"
    EOF = "EOF"

    def __init__(self, type_: str, value: str) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return self.type == other.type and self.value == other.value


class Tokenizer:
    """Lexer for OpenFOAM dictionary syntax.

    Handles:
    - Whitespace and comments (``//``, ``/* */``)
    - Quoted strings (with escape sequences)
    - Numbers (integers, floats, scientific notation)
    - Words (identifiers, keywords)
    - Braces, parentheses, semicolons
    - Dollar-sign macro references
    - Hash directives (``#include``, ``#calc``, etc.)
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._len = len(text)

    def _peek(self) -> str:
        """Peek at current character."""
        if self._pos >= self._len:
            return ""
        return self._text[self._pos]

    def _advance(self) -> str:
        """Return current character and advance position."""
        ch = self._text[self._pos]
        self._pos += 1
        return ch

    def _skip_whitespace_and_comments(self) -> None:
        """Skip whitespace and C-style comments."""
        while self._pos < self._len:
            ch = self._text[self._pos]
            if ch.isspace():
                self._pos += 1
            elif ch == "/" and self._pos + 1 < self._len:
                next_ch = self._text[self._pos + 1]
                if next_ch == "/":
                    # Line comment: skip to end of line
                    self._pos += 2
                    while self._pos < self._len and self._text[self._pos] != "\n":
                        self._pos += 1
                    if self._pos < self._len:
                        self._pos += 1  # skip newline
                elif next_ch == "*":
                    # Block comment: skip to */
                    self._pos += 2
                    while self._pos + 1 < self._len:
                        if self._text[self._pos] == "*" and self._text[self._pos + 1] == "/":
                            self._pos += 2
                            break
                        self._pos += 1
                else:
                    break
            else:
                break

    def _read_string(self) -> str:
        """Read a quoted string (without quotes)."""
        self._advance()  # skip opening quote
        chars: list[str] = []
        while self._pos < self._len:
            ch = self._advance()
            if ch == "\\":
                # Escape sequence
                if self._pos < self._len:
                    esc = self._advance()
                    if esc == "n":
                        chars.append("\n")
                    elif esc == "t":
                        chars.append("\t")
                    elif esc == "\\":
                        chars.append("\\")
                    elif esc == '"':
                        chars.append('"')
                    else:
                        chars.append(esc)
            elif ch == '"':
                return "".join(chars)
            else:
                chars.append(ch)
        return "".join(chars)

    def _read_word(self) -> str:
        """Read a word (identifier/keyword)."""
        start = self._pos
        while self._pos < self._len:
            ch = self._text[self._pos]
            if ch.isalnum() or ch in "_-.:+*^/\\<>":
                self._pos += 1
            else:
                break
        return self._text[start:self._pos]

    def _read_number(self) -> str:
        """Read a number (integer or float, possibly scientific)."""
        start = self._pos
        has_dot = False
        has_exp = False

        # Optional sign
        if self._text[self._pos] in "+-":
            self._pos += 1

        while self._pos < self._len:
            ch = self._text[self._pos]
            if ch.isdigit():
                self._pos += 1
            elif ch == "." and not has_dot:
                has_dot = True
                self._pos += 1
            elif ch in "eE" and not has_exp:
                has_exp = True
                self._pos += 1
                # Optional sign after exponent
                if self._pos < self._len and self._text[self._pos] in "+-":
                    self._pos += 1
            else:
                break
        return self._text[start:self._pos]

    def _is_number_start(self) -> bool:
        """Check if current position starts a number."""
        ch = self._text[self._pos]
        if ch.isdigit():
            return True
        if ch in "+-" and self._pos + 1 < self._len:
            next_ch = self._text[self._pos + 1]
            if next_ch.isdigit() or next_ch == ".":
                return True
        return False

    def next_token(self) -> Token:
        """Return the next token."""
        self._skip_whitespace_and_comments()

        if self._pos >= self._len:
            return Token(Token.EOF, "")

        ch = self._peek()

        if ch == '"':
            return Token(Token.STRING, self._read_string())
        elif ch == "{":
            self._advance()
            return Token(Token.LBRACE, "{")
        elif ch == "}":
            self._advance()
            return Token(Token.RBRACE, "}")
        elif ch == "(":
            self._advance()
            return Token(Token.LPAREN, "(")
        elif ch == ")":
            self._advance()
            return Token(Token.RPAREN, ")")
        elif ch == "[":
            self._advance()
            return Token(Token.LBRACKET, "[")
        elif ch == "]":
            self._advance()
            return Token(Token.RBRACKET, "]")
        elif ch == ";":
            self._advance()
            return Token(Token.SEMICOLON, ";")
        elif ch == "$":
            self._advance()
            return Token(Token.DOLLAR, "$")
        elif ch == "#":
            self._advance()
            return Token(Token.HASH, "#")
        elif self._is_number_start():
            return Token(Token.NUMBER, self._read_number())
        else:
            return Token(Token.WORD, self._read_word())

    def tokenize(self) -> list[Token]:
        """Tokenize the entire input and return a list of tokens."""
        tokens: list[Token] = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type == Token.EOF:
                break
        return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    """Recursive descent parser for OpenFOAM dictionary syntax.

    Grammar (simplified)::

        dict     := entry*
        entry    := key value ';'
                   | key '{' dict '}'
                   | '$' name
                   | '#' directive
        value    := number | string | word | list | dict
        list     := '(' items ')'
        items    := (value)*
    """

    def __init__(self, tokens: list[Token], parent: Optional[FoamDict] = None) -> None:
        self._tokens = tokens
        self._pos = 0
        self._parent = parent

    def _peek(self) -> Token:
        if self._pos >= len(self._tokens):
            return Token(Token.EOF, "")
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, type_: str) -> Token:
        tok = self._advance()
        if tok.type != type_:
            raise ValueError(
                f"Expected token type {type_}, got {tok.type} ({tok.value!r}) "
                f"at position {self._pos}"
            )
        return tok

    def _at_end(self) -> bool:
        return self._pos >= len(self._tokens) or self._peek().type == Token.EOF

    def _is_next(self, type_: str) -> bool:
        return not self._at_end() and self._peek().type == type_

    def parse(self) -> FoamDict:
        """Parse the token stream into a FoamDict."""
        result = FoamDict(parent=self._parent)
        while not self._at_end():
            tok = self._peek()
            if tok.type == Token.RBRACE:
                break
            self._parse_entry(result)
        return result

    def _parse_entry(self, d: FoamDict) -> None:
        """Parse a single dictionary entry."""
        tok = self._peek()

        # Skip standalone semicolons (empty statements)
        if tok.type == Token.SEMICOLON:
            self._advance()
            return

        # Macro reference: $name
        if tok.type == Token.DOLLAR:
            self._advance()
            name_tok = self._advance()
            # Consume optional semicolon
            if self._is_next(Token.SEMICOLON):
                self._advance()
            return

        # Hash directive: #include, #calc, #if, etc.
        if tok.type == Token.HASH:
            self._advance()
            directive = self._advance()
            if directive.value == "include":
                # #include "file" or #includeEtc "file"
                if self._is_next(Token.STRING):
                    self._advance()
                elif self._is_next(Token.WORD):
                    self._advance()
                # Note: actual file inclusion is handled at a higher level
            elif directive.value == "includeEtc":
                if self._is_next(Token.STRING):
                    self._advance()
            elif directive.value == "calc":
                # #calc "expression" — consume the expression token only
                if not self._at_end():
                    self._advance()
            elif directive.value == "ifeq":
                # Skip condition
                while not self._at_end() and self._peek().type != Token.LBRACE:
                    self._advance()
            elif directive.value == "if":
                while not self._at_end() and self._peek().type != Token.LBRACE:
                    self._advance()
            elif directive.value == "else":
                pass
            elif directive.value == "endif":
                pass
            # Consume optional semicolon
            if self._is_next(Token.SEMICOLON):
                self._advance()
            return

        # Regular entry: key value; or key { ... };
        key_tok = self._advance()
        if key_tok.type not in (Token.WORD, Token.STRING):
            raise ValueError(
                f"Expected key (word or string), got {key_tok.type} ({key_tok.value!r})"
            )
        key = key_tok.value

        # Check if this is a sub-dictionary
        if self._is_next(Token.LBRACE):
            self._advance()  # consume '{'
            sub_parser = _Parser(self._tokens[self._pos:], parent=d)
            sub_dict = sub_parser.parse()
            self._pos += sub_parser._pos
            self._expect(Token.RBRACE)
            # Consume optional semicolon after closing brace
            if self._is_next(Token.SEMICOLON):
                self._advance()
            d[key] = sub_dict
            return

        # Parse value
        value = self._parse_value()
        # Handle multi-token values (e.g., "value uniform 0;")
        # Keep reading tokens until semicolon
        if isinstance(value, str) and not self._at_end():
            next_tok = self._peek()
            if next_tok.type not in (Token.SEMICOLON, Token.LBRACE, Token.EOF, Token.RBRACE):
                parts = [value]
                while not self._at_end():
                    next_tok = self._peek()
                    if next_tok.type in (Token.SEMICOLON, Token.LBRACE, Token.EOF, Token.RBRACE):
                        break
                    parts.append(self._advance().value)
                value = " ".join(parts)
        # Consume semicolon
        if self._is_next(Token.SEMICOLON):
            self._advance()
        d[key] = value

    def _parse_value(self) -> Any:
        """Parse a value (number, string, word, or list)."""
        tok = self._peek()

        if tok.type == Token.NUMBER:
            self._advance()
            return self._parse_number(tok.value)
        elif tok.type == Token.STRING:
            self._advance()
            return tok.value
        elif tok.type == Token.WORD:
            self._advance()
            return tok.value
        elif tok.type == Token.LPAREN:
            return self._parse_list()
        elif tok.type == Token.LBRACKET:
            return self._parse_bracket_list()
        elif tok.type == Token.DOLLAR:
            # Macro reference as value
            self._advance()
            name_tok = self._advance()
            return f"${name_tok.value}"
        elif tok.type == Token.LBRACE:
            # Inline sub-dictionary as value
            self._advance()
            sub_parser = _Parser(self._tokens[self._pos:], parent=self._parent)
            sub_dict = sub_parser.parse()
            self._pos += sub_parser._pos
            self._expect(Token.RBRACE)
            return sub_dict
        else:
            raise ValueError(f"Unexpected token for value: {tok}")

    def _parse_list(self) -> FoamList:
        """Parse a list: ``<type>N(...)`` or ``(items)``."""
        self._expect(Token.LPAREN)

        # Check for type tag before parenthesis: e.g., List<vector>
        # This is handled by the caller checking the previous word
        items: list[Any] = []
        while not self._at_end() and self._peek().type != Token.RPAREN:
            items.append(self._parse_value())
        self._expect(Token.RPAREN)
        return FoamList(items)

    def _parse_bracket_list(self) -> str:
        """Parse a bracket-enclosed list: ``[0 2 -2 0 0 0 0]``.

        Returns the entire bracket content as a string (dimensions are stored as-is).
        """
        self._expect(Token.LBRACKET)
        parts: list[str] = []
        while not self._at_end() and self._peek().type != Token.RBRACKET:
            tok = self._advance()
            parts.append(tok.value)
        self._expect(Token.RBRACKET)
        return f"[{' '.join(parts)}]"

    @staticmethod
    def _parse_number(text: str) -> Union[int, float]:
        """Parse a number string to int or float."""
        try:
            return int(text)
        except ValueError:
            return float(text)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def parse_dict(text: str, parent: Optional[FoamDict] = None) -> FoamDict:
    """Parse an OpenFOAM dictionary from text.

    Args:
        text: Dictionary text content.
        parent: Optional parent dictionary for macro resolution.

    Returns:
        Parsed :class:`FoamDict`.
    """
    # Strip the FoamFile header if present
    from pyfoam.io.foam_file import split_header_body
    try:
        _, body = split_header_body(text)
    except ValueError:
        body = text

    tokenizer = Tokenizer(body)
    tokens = tokenizer.tokenize()
    parser = _Parser(tokens, parent=parent)
    return parser.parse()


def parse_dict_file(path: Union[str, Path]) -> FoamDict:
    """Parse an OpenFOAM dictionary file.

    Args:
        path: Path to the dictionary file.

    Returns:
        Parsed :class:`FoamDict`.
    """
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_dict(content)


def expand_macros(d: FoamDict, *, top_level: Optional[FoamDict] = None) -> FoamDict:
    """Expand all ``$variable`` references in a FoamDict.

    Recursively resolves ``$var`` and ``$subdict/var`` references.

    Args:
        d: Dictionary to expand.
        top_level: Top-level dictionary for ``$!var`` lookups.

    Returns:
        New FoamDict with all macros expanded.
    """
    if top_level is None:
        top_level = d

    result = FoamDict(parent=d.parent)
    for key, value in d.items():
        if isinstance(value, str) and value.startswith("$"):
            var_name = value[1:]
            # $!var searches only top-level
            if var_name.startswith("!"):
                var_name = var_name[1:]
                try:
                    result[key] = top_level[var_name]
                except KeyError:
                    result[key] = value  # Keep unresolved
            else:
                try:
                    result[key] = d.resolve_macro(var_name)
                except KeyError:
                    result[key] = value  # Keep unresolved
        elif isinstance(value, FoamDict):
            result[key] = expand_macros(value, top_level=top_level)
        elif isinstance(value, FoamList):
            expanded_items = []
            for item in value:
                if isinstance(item, str) and item.startswith("$"):
                    var_name = item[1:]
                    try:
                        expanded_items.append(d.resolve_macro(var_name))
                    except KeyError:
                        expanded_items.append(item)
                else:
                    expanded_items.append(item)
            result[key] = FoamList(expanded_items, foam_type=value.foam_type)
        else:
            result[key] = value
    return result
