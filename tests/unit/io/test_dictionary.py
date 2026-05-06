"""Tests for OpenFOAM dictionary parser."""

import pytest

from pyfoam.io.dictionary import (
    FoamDict,
    FoamList,
    Token,
    Tokenizer,
    expand_macros,
    parse_dict,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_simple_tokens(self):
        """Tokenize simple key-value pair."""
        tok = Tokenizer("key value;")
        tokens = tok.tokenize()
        types = [t.type for t in tokens]
        assert types == [Token.WORD, Token.WORD, Token.SEMICOLON, Token.EOF]

    def test_string_token(self):
        """Tokenize quoted string."""
        tok = Tokenizer('"hello world"')
        tokens = tok.tokenize()
        assert tokens[0].type == Token.STRING
        assert tokens[0].value == "hello world"

    def test_number_token(self):
        """Tokenize number."""
        tok = Tokenizer("3.14")
        tokens = tok.tokenize()
        assert tokens[0].type == Token.NUMBER
        assert tokens[0].value == "3.14"

    def test_negative_number(self):
        """Tokenize negative number."""
        tok = Tokenizer("-1.5")
        tokens = tok.tokenize()
        assert tokens[0].type == Token.NUMBER
        assert tokens[0].value == "-1.5"

    def test_scientific_notation(self):
        """Tokenize scientific notation."""
        tok = Tokenizer("1.5e-3")
        tokens = tok.tokenize()
        assert tokens[0].type == Token.NUMBER
        assert tokens[0].value == "1.5e-3"

    def test_braces_and_parens(self):
        """Tokenize braces and parentheses."""
        tok = Tokenizer("{ } ( )")
        tokens = tok.tokenize()
        types = [t.type for t in tokens[:-1]]
        assert types == [
            Token.LBRACE, Token.RBRACE, Token.LPAREN, Token.RPAREN,
        ]

    def test_dollar_macro(self):
        """Tokenize dollar macro."""
        tok = Tokenizer("$variable")
        tokens = tok.tokenize()
        assert tokens[0].type == Token.DOLLAR
        assert tokens[1].type == Token.WORD
        assert tokens[1].value == "variable"

    def test_hash_directive(self):
        """Tokenize hash directive."""
        tok = Tokenizer('#include "file"')
        tokens = tok.tokenize()
        assert tokens[0].type == Token.HASH
        assert tokens[1].type == Token.WORD
        assert tokens[1].value == "include"

    def test_line_comment(self):
        """Skip line comments."""
        tok = Tokenizer("key // comment\nvalue;")
        tokens = tok.tokenize()
        types = [t.type for t in tokens]
        assert Token.WORD in types
        # Comment should be skipped
        assert all("comment" not in t.value for t in tokens if t.type != Token.EOF)

    def test_block_comment(self):
        """Skip block comments."""
        tok = Tokenizer("key /* block\ncomment */ value;")
        tokens = tok.tokenize()
        types = [t.type for t in tokens]
        assert types == [Token.WORD, Token.WORD, Token.SEMICOLON, Token.EOF]

    def test_empty_input(self):
        """Empty input produces EOF."""
        tok = Tokenizer("")
        tokens = tok.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == Token.EOF

    def test_word_with_special_chars(self):
        """Tokenize words with special characters."""
        tok = Tokenizer("List<vector>")
        tokens = tok.tokenize()
        assert tokens[0].type == Token.WORD
        assert tokens[0].value == "List<vector>"


# ---------------------------------------------------------------------------
# FoamDict
# ---------------------------------------------------------------------------


class TestFoamDict:
    def test_basic_dict(self):
        """Basic dictionary operations."""
        d = FoamDict()
        d["key"] = "value"
        assert d["key"] == "value"

    def test_path_lookup(self):
        """Path-based lookup with /."""
        d = FoamDict()
        sub = FoamDict()
        sub["inner"] = 42
        d["sub"] = sub
        assert d["sub/inner"] == 42

    def test_path_lookup_not_found(self):
        """Path lookup raises KeyError."""
        d = FoamDict()
        with pytest.raises(KeyError):
            d["nonexistent/key"]

    def test_parent_chain(self):
        """Parent chain lookup for macros."""
        parent = FoamDict()
        parent["var"] = 100
        child = FoamDict(parent=parent)
        assert child.resolve_macro("var") == 100

    def test_resolve_macro_not_found(self):
        """KeyError when macro not found."""
        d = FoamDict()
        with pytest.raises(KeyError):
            d.resolve_macro("nonexistent")

    def test_resolve_macro_path(self):
        """Resolve macro with path."""
        d = FoamDict()
        sub = FoamDict()
        sub["key"] = 42
        d["sub"] = sub
        assert d.resolve_macro("sub/key") == 42

    def test_get_path(self):
        """get_path with default."""
        d = FoamDict()
        assert d.get_path("missing", default=99) == 99

    def test_set_parent(self):
        """set_parent updates parent reference."""
        d = FoamDict()
        parent = FoamDict()
        parent["x"] = 1
        d.set_parent(parent)
        assert d.resolve_macro("x") == 1


# ---------------------------------------------------------------------------
# FoamList
# ---------------------------------------------------------------------------


class TestFoamList:
    def test_basic_list(self):
        """Basic list operations."""
        lst = FoamList([1, 2, 3])
        assert list(lst) == [1, 2, 3]

    def test_foam_type(self):
        """FoamList with type tag."""
        lst = FoamList([1, 2], foam_type="vector")
        assert lst.foam_type == "vector"

    def test_repr(self):
        """repr includes type tag."""
        lst = FoamList([1, 2], foam_type="scalar")
        r = repr(lst)
        assert "scalar" in r


# ---------------------------------------------------------------------------
# parse_dict
# ---------------------------------------------------------------------------


class TestParseDict:
    def test_simple_key_value(self):
        """Parse simple key-value pair."""
        d = parse_dict("key value;")
        assert d["key"] == "value"

    def test_multiple_entries(self):
        """Parse multiple entries."""
        d = parse_dict("""
            key1 value1;
            key2 value2;
        """)
        assert d["key1"] == "value1"
        assert d["key2"] == "value2"

    def test_numeric_values(self):
        """Parse numeric values."""
        d = parse_dict("""
            intVal 42;
            floatVal 3.14;
        """)
        assert d["intVal"] == 42
        assert d["floatVal"] == 3.14

    def test_string_values(self):
        """Parse quoted string values."""
        d = parse_dict('key "hello world";')
        assert d["key"] == "hello world"

    def test_sub_dictionary(self):
        """Parse nested sub-dictionary."""
        d = parse_dict("""
            subDict
            {
                key value;
            };
        """)
        assert isinstance(d["subDict"], FoamDict)
        assert d["subDict"]["key"] == "value"

    def test_nested_sub_dictionary(self):
        """Parse deeply nested sub-dictionary."""
        d = parse_dict("""
            level1
            {
                level2
                {
                    key deep;
                };
            };
        """)
        assert d["level1"]["level2"]["key"] == "deep"

    def test_list_values(self):
        """Parse list values."""
        d = parse_dict("key (1 2 3);")
        assert isinstance(d["key"], FoamList)
        assert list(d["key"]) == [1, 2, 3]

    def test_list_with_strings(self):
        """Parse list with string values."""
        d = parse_dict('key ("a" "b" "c");')
        assert list(d["key"]) == ["a", "b", "c"]

    def test_dimensions(self):
        """Parse dimensions vector."""
        d = parse_dict("dimensions [0 2 -2 0 0 0 0];")
        assert d["dimensions"] == "[0 2 -2 0 0 0 0]"

    def test_comment_handling(self):
        """Parse with comments."""
        d = parse_dict("""
            // Line comment
            key1 value1;
            /* Block
               comment */
            key2 value2;
        """)
        assert d["key1"] == "value1"
        assert d["key2"] == "value2"

    def test_macro_reference(self):
        """Parse macro references."""
        d = parse_dict("key $variable;")
        assert d["key"] == "$variable"

    def test_include_directive(self):
        """Parse #include directive."""
        d = parse_dict('#include "file.h"\nkey value;')
        assert d["key"] == "value"

    def test_calc_directive(self):
        """Parse #calc directive."""
        d = parse_dict('#calc "1+2"\nkey value;')
        assert d["key"] == "value"

    def test_empty_dict(self):
        """Parse empty dictionary."""
        d = parse_dict("")
        assert len(d) == 0

    def test_foam_file_header_stripped(self):
        """FoamFile header is stripped before parsing."""
        d = parse_dict("""
            FoamFile
            {
                version 2.0;
                format ascii;
            }
            key value;
        """)
        assert d["key"] == "value"

    def test_semicolons_optional(self):
        """Semicolons are optional for some entries."""
        d = parse_dict("""
            key1 value1;
            key2 value2
        """)
        assert d["key1"] == "value1"

    def test_control_dict_style(self):
        """Parse typical controlDict content."""
        d = parse_dict("""
            application     simpleFoam;

            startFrom       startTime;

            startTime       0;

            stopAt          endTime;

            endTime         1000;

            deltaT          1;
        """)
        assert d["application"] == "simpleFoam"
        assert d["startTime"] == 0
        assert d["endTime"] == 1000
        assert d["deltaT"] == 1

    def test_fv_schemes_style(self):
        """Parse typical fvSchemes content."""
        d = parse_dict("""
            ddtSchemes
            {
                default         Euler;
            }

            gradSchemes
            {
                default         Gauss;
            }
        """)
        assert d["ddtSchemes"]["default"] == "Euler"
        assert d["gradSchemes"]["default"] == "Gauss"

    def test_fv_solution_style(self):
        """Parse typical fvSolution content."""
        d = parse_dict("""
            solvers
            {
                p
                {
                    solver          PCG;
                    preconditioner  DIC;
                    tolerance       1e-06;
                    relTol          0.01;
                }

                U
                {
                    solver          PBiCGStab;
                    preconditioner  DILU;
                    tolerance       1e-05;
                    relTol          0;
                }
            }
        """)
        assert d["solvers"]["p"]["solver"] == "PCG"
        assert d["solvers"]["U"]["solver"] == "PBiCGStab"

    def test_typed_list(self):
        """Parse typed list (e.g., List<vector>)."""
        d = parse_dict("""
            myField   (1 2 3);
        """)
        # The value should be parsed as a FoamList
        assert "myField" in d
        assert isinstance(d["myField"], FoamList)

    def test_inline_subdict(self):
        """Parse inline sub-dictionary as value."""
        d = parse_dict("""
            entry
            {
                subkey subvalue;
            };
        """)
        assert isinstance(d["entry"], FoamDict)
        assert d["entry"]["subkey"] == "subvalue"


# ---------------------------------------------------------------------------
# expand_macros
# ---------------------------------------------------------------------------


class TestExpandMacros:
    def test_simple_macro(self):
        """Expand simple $var reference."""
        d = FoamDict()
        d["var"] = 42
        d["key"] = "$var"
        result = expand_macros(d)
        assert result["key"] == 42

    def test_macro_not_found(self):
        """Unresolved macro kept as-is."""
        d = FoamDict()
        d["key"] = "$missing"
        result = expand_macros(d)
        assert result["key"] == "$missing"

    def test_nested_macro(self):
        """Expand macro in nested dict."""
        d = FoamDict()
        d["var"] = 100
        sub = FoamDict(parent=d)
        sub["key"] = "$var"
        d["sub"] = sub
        result = expand_macros(d)
        assert result["sub"]["key"] == 100

    def test_path_macro(self):
        """Expand $subdict/var macro."""
        d = FoamDict()
        sub = FoamDict(parent=d)
        sub["inner"] = 42
        d["sub"] = sub
        d["key"] = "$sub/inner"
        result = expand_macros(d)
        assert result["key"] == 42

    def test_list_macro(self):
        """Expand macro in list."""
        d = FoamDict()
        d["var"] = 99
        d["key"] = FoamList(["$var", 2, 3])
        result = expand_macros(d)
        assert result["key"][0] == 99

    def test_no_macros(self):
        """No-op when no macros present."""
        d = FoamDict()
        d["key"] = "value"
        result = expand_macros(d)
        assert result["key"] == "value"
