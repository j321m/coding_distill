"""The prompt this model was fine-tuned on, reproduced for evaluation.

Wrapper-only fidelity. The training problem body carried `## Constraints` /
`## Input Format` sections assembled from the codeforces dataset's separate
fields, which LiveCodeBench does not expose -- reconstructing them from its
single `question_content` blob would match neither corpus.
"""

import re

# The invariant part of every training prompt, taken verbatim from the tokenized
# artifact (everything before the problem statement is identical across samples).
SFT_PROMPT_HEAD = (
    "\nYou will be given a competitive programming problem.\n"
    "Analyze the maximum input constraints and identify the optimal algorithmic "
    "approach and data structures needed to process the largest possible test "
    "cases within the time and memory limits, then explain why your chosen "
    "implementation strategy is the most efficient solution. Please reason step "
    "by step about your solution approach, then provide a complete "
    "implementation in Python 3 that is thoroughly optimized for both speed and "
    "memory usage.\n\n"
)

# Every codeforces-cots trace is stdin/stdout, so this half is what the model was
# actually trained on.
SFT_PROMPT_STDIN = (
    "Your solution must read input from standard input (input()), write output "
    "to standard output (print()).\n"
    "Do not include any debug prints or additional output.\n\n"
    "Put your final solution within a single code block:\n"
    "```python\n<your code here>\n```\n\n"
)

# A quarter of LiveCodeBench is LeetCode-style, where the checker calls a method
# rather than running a program. Asking those for stdin/stdout scores zero however
# good the answer is, so they get upstream's instruction instead -- off
# distribution for this model, but at least the right task.
SFT_PROMPT_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters.\n\n"
    "```python\n{starter_code}\n```\n\n"
)

# the fence SFT_PROMPT_STDIN asks for, so a generation can be checked for it
CODE_FENCE = re.compile(r"```python", re.IGNORECASE)


def build_sft_template(tokenizer):
    """The question -> prompt function LiveCodeBench should call.

    A closure over the tokenizer because the chat template has to be the same
    call that produced the training data, so the two cannot drift.
    """

    def sft_question_template(question):
        """Branches on `starter_code` the way every upstream template does.

        Without it a LeetCode-style problem is asked for a stdin/stdout program
        that its functional checker can never call -- zero by construction.
        """
        body = (
            SFT_PROMPT_STARTER.format(starter_code=question.starter_code)
            if question.starter_code
            else SFT_PROMPT_STDIN
        )
        content = SFT_PROMPT_HEAD + body + "# Problem\n\n" + question.question_content
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return sft_question_template
