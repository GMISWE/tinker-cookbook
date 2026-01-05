import math
import re
from functools import partial
from typing import Literal, Sequence

import chz


# Patterns that are known to cause grading timeouts
HARD_EXPRESSION_PATTERNS = [
    r"S_\{?n\}?",      # S_n type expressions (summation notation)
    r"\\cot",          # Cotangent expressions
]


def is_hard_expression(answer: str) -> bool:
    """Detect expressions that are known to cause grading timeouts."""
    for pattern in HARD_EXPRESSION_PATTERNS:
        if re.search(pattern, answer):
            return True
    return False
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
        timeout: float = 1.0,
        format_coef: float = 0.0,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Let's think step by step and output the final answer within \\boxed{}."

    def get_question(self) -> str:
        return self.problem

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    # Skip hard-to-grade expressions that are known to cause timeouts
    if is_hard_expression(ground_truth):
        logger.debug(f"[GRADE] Skipping hard expression: {ground_truth[:50]}...")
        return False

    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    logger.debug(f"[GRADE] Starting grade: {given_answer[:50]}... vs {ground_truth[:50]}...")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    logger.debug(f"[GRADE] Grade complete: result={out}")
    return out


class ReinforceAdaDataset(RLDataset):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        n_batches: int | None = None,
        seed: int = 0,
    ):
        self.ds = load_dataset(dataset_name, split="train").shuffle(seed=seed)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.question_suffix = (
            " Let's think step by step and output the final answer within \\boxed{}."
        )
        self.n_batches = n_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        full_len = math.ceil(len(self.ds) / self.batch_size)
        if self.n_batches is None:
            return full_len
        return min(self.n_batches, full_len)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        problem = x.get("problem", "")
        answer = x.get("gt", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem + self.question_suffix,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
            ),
            num_envs=group_size,
            dataset_name=self.dataset_name,
        )


@chz.chz
class ReinforceAdaDatasetBuilder(RLDatasetBuilder):
    dataset_name: str
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    n_batches: int | None = None
    seed: int = 0

    async def __call__(self) -> tuple[ReinforceAdaDataset, None]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            ReinforceAdaDataset(
                dataset_name=self.dataset_name,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                n_batches=self.n_batches,
                seed=self.seed,
            ),
            None,
        )
