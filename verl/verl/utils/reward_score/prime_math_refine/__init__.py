# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Refined answer checker API that uses sympy to simplify expressions and check for equality.

Key improvements over prime_math:
1. Only extracts answer from \\boxed{} (not from calculation process)
2. Strict matching for integers (no tolerance)
3. Flexible handling for mathematical expressions (using sympy)
4. No tolerance for large integers

Call compute_score(model_output: str, ground_truth: str).

FROM: https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py
"""

import contextlib
import math
import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from verl.utils.py_functional import timeout_limit

from . import math_normalize
from .grader import math_equal

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "liter",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    return all(re.search(bad_regex, expr) is None for bad_regex in BAD_REGEXES)


@timeout_limit(seconds=10)
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if (
        len(ground_truth_elems) > 1
        and (ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1])
        or len(ground_truth_elems) != len(given_elems)
    ):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=True):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match
                # (no sympy.simplify)
                is_correct = False
            else:
                try:
                    is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                except Exception as e:
                    # if there's an error, we'll just say it's not correct
                    is_correct = False
                    print(f"Error: {e} from are_equal_under_sympy, {ground_truth_elem}, {given_elem}")
            if not is_correct:
                break

    return is_correct


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _last_boxed_only_string(string):
    """
    Extract the last \\boxed{} content from a string.
    Only extracts from \\boxed{}, not from calculation process.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def compute_score(model_output: str, ground_truth: str) -> tuple[bool, bool, str | None]:
    """
    Compute score with the following requirements:
    1. Only extract answer from \\boxed{} (not from calculation process)
    2. Strict matching for integers (no tolerance)
    3. Flexible handling for mathematical expressions (using sympy)
    4. No tolerance for large integers

    Args:
        model_output: The model's output string
        ground_truth: The ground truth answer string

    Returns:
        Tuple of (is_correct, format_correctness, extracted_answer)
        - is_correct: True if the answer is correct, False otherwise
        - format_correctness: True if \\boxed{} is present in the output
        - extracted_answer: The extracted answer from \\boxed{}, or None if not found
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)

    # Requirement 1: Only extract from \\boxed{}
    boxed_content = _last_boxed_only_string(model_output)
    if boxed_content is None:
        # No \\boxed{} found, return False
        return False, False, None

    extracted_model_output = boxed_content.strip()
    format_correctness = "\\box" in model_output

    if not extracted_model_output:
        # Empty answer
        return False, format_correctness, None

    # Requirement 4: Strict matching for integers (no tolerance)
    # Check if both ground_truth and extracted answer are integers
    gt_is_int = _str_is_int(ground_truth)
    pred_is_int = _str_is_int(extracted_model_output)

    if gt_is_int and pred_is_int:
        # For integers, use strict matching (no tolerance, no sympy simplification)
        # Convert to int and compare (handles commas, whitespace, leading zeros, etc.)
        try:
            gt_int = _str_to_int(ground_truth)
            pred_int = _str_to_int(extracted_model_output)
            is_correct = gt_int == pred_int
        except Exception:
            # Fallback to string comparison if conversion fails
            gt_normalized = _strip_properly_formatted_commas(ground_truth).strip()
            pred_normalized = _strip_properly_formatted_commas(extracted_model_output).strip()
            is_correct = gt_normalized == pred_normalized
        return is_correct, format_correctness, extracted_model_output

    # Requirement 3: Flexible handling for mathematical expressions
    # Use grade_answer for mathematical equivalence checking (uses sympy)
    if grade_answer(extracted_model_output, ground_truth):
        return True, format_correctness, extracted_model_output

    # Fallback: use math_equal with tolerance=0.0 for strict matching
    try:
        if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
            equivs = []
            for pi in [math.pi, 3.14]:
                # Set tolerance=0.0 to disable tolerance for non-integer cases
                equivs.append(math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi, tolerance=0.0))
            is_correct = any(equivs)
        else:
            # Set tolerance=0.0 to disable tolerance
            is_correct = math_equal(extracted_model_output, ground_truth, timeout=True, tolerance=0.0)
    except Exception:
        is_correct = False

    return is_correct, format_correctness, extracted_model_output

