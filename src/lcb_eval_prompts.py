"""The prompts the model was fine-tuned on, reproduced for evaluation."""

import re

# Verbatim from the tokenized artifact: everything before the problem statement
# is identical across samples, and differs between the Python and C++ configs by
# more than the language -- solutions_py_decontaminated carries the long
# "Analyze the maximum input constraints" head that the C++ one does not.
PY_HEAD = (
    "\nYou will be given a competitive programming problem.\n"
    "Analyze the maximum input constraints and identify the optimal algorithmic "
    "approach and data structures needed to process the largest possible test "
    "cases within the time and memory limits, then explain why your chosen "
    "implementation strategy is the most efficient solution. Please reason step "
    "by step about your solution approach, then provide a complete "
    "implementation in Python 3 that is thoroughly optimized for both speed and "
    "memory usage.\n\n"
)

PY_STDIN = (
    "Your solution must read input from standard input (input()), write output "
    "to standard output (print()).\n"
    "Do not include any debug prints or additional output.\n\n"
    "Put your final solution within a single code block:\n"
    "```python\n<your code here>\n```\n\n"
)

# LeetCode-style problems: the checker calls a method rather than running a
# program, so asking for stdin/stdout scores zero however good the answer is
PY_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters.\n\n"
    "```python\n{starter_code}\n```\n\n"
)

CPP_HEAD = (
    "You will be given a competitive programming problem. Please reason step by "
    "step about the solution, then provide a complete implementation in C++17.\n\n"
)

CPP_STDIN = (
    "Your solution must read input from standard input (cin), write output to "
    "standard output (cout).\n"
    "Do not include any debug prints or additional output.\n\n"
    "Put your final solution within a single code block:\n"
    "```cpp\n<your code here>\n```\n\n"
)

# the fence each prompt asks for, so a generation can be checked for it
CODE_FENCE = {
    "python": re.compile(r"```python", re.IGNORECASE),
    "cpp": re.compile(r"```cpp", re.IGNORECASE),
}


def build_sft_template(tokenizer, language="python"):
    """The question -> prompt function LiveCodeBench should call."""

    def python_body(question):
        """Branches on `starter_code` the way every upstream template does."""
        if question.starter_code:
            return PY_HEAD + PY_STARTER.format(starter_code=question.starter_code)
        return PY_HEAD + PY_STDIN

    def cpp_body(question):
        # no starter-code branch: those problems are Python method stubs, which
        # this prompt has no way to express
        return CPP_HEAD + CPP_STDIN

    body = python_body if language == "python" else cpp_body

    def sft_question_template(question):
        content = body(question) + "# Problem\n\n" + question.question_content
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return sft_question_template
