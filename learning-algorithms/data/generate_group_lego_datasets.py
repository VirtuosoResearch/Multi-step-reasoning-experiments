#!/usr/bin/env python3
"""Generate cyclic and symmetric LEGO state-tracking datasets.

The generated JSON matches the local datasets in ``lego_dataset`` and
``symmetry_dataset``:

    question, answer, final_values, sequence_order, dependency_order

Each example is a shuffled chain of variable assignments.  The answer is the
progressive trace obtained by revealing variables in dependency order.  In this
script, ``length`` means the number of variables/progressive steps in the JSON,
matching ``get_length_lego`` in the local data module.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


Value = int
Trace = List[Union[Value, str]]


CYCLIC_HELP = (
    "Cyclic example: shift(2) 3 mod 6 = 5. "
    "This means add 3 to the input modulo 6."
)
PERMUTATION_HELP = (
    "Permutation example: permute(2) 1 3 5 4 = 3. "
    "This means when the input is 2, the output is the 2nd number in the list."
)


@dataclass(frozen=True)
class Assignment:
    lhs: str
    op: str
    rhs: Optional[str]
    value: Optional[int]
    permutation: Optional[Tuple[int, ...]] = None
    modulus: Optional[int] = None

    def to_text(self) -> str:
        if self.op == "val":
            return f"{self.lhs} = val {self.value}"
        if self.op == "shift":
            return f"{self.lhs} = shift({self.rhs}) {self.value} mod {self.modulus}"
        if self.op == "permute":
            perm = " ".join(str(x) for x in self.permutation)
            return f"{self.lhs} = permute({self.rhs}) {perm}"
        raise ValueError(f"Unsupported operation: {self.op}")


def build_variable_pool(count: int) -> List[str]:
    """Return a deterministic variable-name pool with at least ``count`` names."""

    names: List[str] = list(string.ascii_lowercase)
    if count <= len(names):
        return names[:count]

    for width in range(2, 5):
        for chars in itertools.product(string.ascii_lowercase, repeat=width):
            names.append("".join(chars))
            if count <= len(names):
                return names[:count]

    raise ValueError("Unable to build enough variable names.")


def format_trace(trace: Sequence[Union[Value, str]]) -> str:
    return "[" + " ".join(str(item) for item in trace) + "]"


def progressive_answer(
    sequence_order: Sequence[str],
    dependency_order: Sequence[str],
    values_by_var: Dict[str, int],
) -> str:
    index_by_var = {var: idx for idx, var in enumerate(sequence_order)}
    trace: Trace = ["n"] * len(sequence_order)
    steps: List[str] = []

    for var in dependency_order:
        trace[index_by_var[var]] = values_by_var[var]
        steps.append(format_trace(trace))

    final_values = [values_by_var[var] for var in sequence_order]
    return ", ".join(steps) + " | " + format_trace(final_values)


def make_question(
    task: str,
    assignments_in_sequence_order: Sequence[Assignment],
    sequence_order: Sequence[str],
) -> str:
    if task == "cyclic":
        task_help = f"\n{CYCLIC_HELP}"
    elif task == "symmetric":
        task_help = f"\n{PERMUTATION_HELP}"
    else:
        task_help = ""

    sequence_text = ", ".join(a.to_text() for a in assignments_in_sequence_order) + ","
    variables_text = " ".join(sequence_order)
    initial_trace = format_trace(["n"] * len(sequence_order))

    return (
        f"lego_reasoning:{task_help}\n"
        f"sequence: {sequence_text}\n"
        f"variables: {variables_text}\n"
        f"initial_trace: {initial_trace}\n"
        "progressive_steps | final_values:"
    )


def make_record(
    task: str,
    assignments_by_var: Dict[str, Assignment],
    sequence_order: Sequence[str],
    dependency_order: Sequence[str],
    values_by_var: Dict[str, int],
) -> Dict[str, object]:
    assignments = [assignments_by_var[var] for var in sequence_order]
    final_values = [values_by_var[var] for var in sequence_order]
    return {
        "question": make_question(task, assignments, sequence_order),
        "answer": progressive_answer(sequence_order, dependency_order, values_by_var),
        "final_values": final_values,
        "sequence_order": list(sequence_order),
        "dependency_order": list(dependency_order),
    }


def sample_chain_variables(rng: random.Random, length: int) -> List[str]:
    pool = build_variable_pool(max(length * 4, length))
    return rng.sample(pool, length)


def generate_cyclic_record(
    rng: random.Random,
    length: int,
    modulus: int,
    shuffle_sequence: bool,
) -> Dict[str, object]:
    dependency_order = sample_chain_variables(rng, length)
    y0 = rng.randrange(modulus)

    values_by_var: Dict[str, int] = {dependency_order[0]: y0}
    assignments_by_var: Dict[str, Assignment] = {
        dependency_order[0]: Assignment(
            lhs=dependency_order[0],
            op="val",
            rhs=None,
            value=y0,
        )
    }

    current = y0
    for prev_var, var in zip(dependency_order, dependency_order[1:]):
        shift = rng.randrange(modulus)
        current = (current + shift) % modulus
        values_by_var[var] = current
        assignments_by_var[var] = Assignment(
            lhs=var,
            op="shift",
            rhs=prev_var,
            value=shift,
            modulus=modulus,
        )

    sequence_order = list(dependency_order)
    if shuffle_sequence:
        rng.shuffle(sequence_order)

    return make_record(
        task="cyclic",
        assignments_by_var=assignments_by_var,
        sequence_order=sequence_order,
        dependency_order=dependency_order,
        values_by_var=values_by_var,
    )


def generate_symmetric_record(
    rng: random.Random,
    length: int,
    degree: int,
    shuffle_sequence: bool,
) -> Dict[str, object]:
    dependency_order = sample_chain_variables(rng, length)
    y0 = rng.randrange(degree)

    values_by_var: Dict[str, int] = {dependency_order[0]: y0}
    assignments_by_var: Dict[str, Assignment] = {
        dependency_order[0]: Assignment(
            lhs=dependency_order[0],
            op="val",
            rhs=None,
            value=y0,
        )
    }

    current = y0
    for prev_var, var in zip(dependency_order, dependency_order[1:]):
        permutation = list(range(degree))
        rng.shuffle(permutation)
        current = permutation[current]
        values_by_var[var] = current
        assignments_by_var[var] = Assignment(
            lhs=var,
            op="permute",
            rhs=prev_var,
            value=None,
            permutation=tuple(permutation),
        )

    sequence_order = list(dependency_order)
    if shuffle_sequence:
        rng.shuffle(sequence_order)

    return make_record(
        task="symmetric",
        assignments_by_var=assignments_by_var,
        sequence_order=sequence_order,
        dependency_order=dependency_order,
        values_by_var=values_by_var,
    )


def generate_split(
    task: str,
    size: int,
    rng: random.Random,
    length: int,
    group_size: int,
    shuffle_sequence: bool,
) -> List[Dict[str, object]]:
    if task == "cyclic":
        return [
            generate_cyclic_record(rng, length, group_size, shuffle_sequence)
            for _ in range(size)
        ]
    if task == "symmetric":
        return [
            generate_symmetric_record(rng, length, group_size, shuffle_sequence)
            for _ in range(size)
        ]
    raise ValueError(f"Unsupported task: {task}")


def write_json(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def parse_split_sizes(raw: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for part in raw.split(","):
        name, _, size = part.partition("=")
        if not name or not size:
            raise argparse.ArgumentTypeError(
                "Split sizes must look like train=5000,val=1000,test=100"
            )
        sizes[name.strip()] = int(size)

    required = {"train", "val", "test"}
    missing = required.difference(sizes)
    if missing:
        raise argparse.ArgumentTypeError(f"Missing split sizes: {sorted(missing)}")
    return sizes


def parse_lengths(raw: Union[str, Sequence[str]]) -> List[int]:
    if isinstance(raw, str):
        raw_parts = [raw]
    else:
        raw_parts = list(raw)

    lengths = [
        int(part.strip())
        for raw_part in raw_parts
        for part in raw_part.split(",")
        if part.strip()
    ]
    if not lengths:
        raise argparse.ArgumentTypeError("At least one length is required.")
    if any(length < 2 for length in lengths):
        raise argparse.ArgumentTypeError("All lengths must be at least 2.")
    return lengths


def add_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["cyclic", "symmetric"],
        default=["cyclic", "symmetric"],
        help="Which LEGO group-action datasets to generate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that will receive cyclic_lego_dataset/ and symmetric_lego_dataset/.",
    )
    parser.add_argument(
        "--split-sizes",
        type=parse_split_sizes,
        default=parse_split_sizes("train=5000,val=1000,test=100"),
        help=(
            "Comma-separated split sizes per requested length, "
            "e.g. train=5000,val=1000,test=100."
        ),
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=["6"],
        help=(
            "Comma-separated LEGO lengths to generate, measured as the number "
            "of variables/progressive steps, e.g. --lengths 10,20 or --lengths 10 20."
        ),
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=None,
        help=(
            "Deprecated alias for a single --lengths value. "
            "Kept for compatibility with earlier versions of this script."
        ),
    )
    parser.add_argument(
        "--cyclic-order",
        type=int,
        default=6,
        help="Order of the cyclic group Cn; the paper's experiments use C6.",
    )
    parser.add_argument(
        "--symmetric-degree",
        type=int,
        default=5,
        help="Degree of the symmetric group Sn; the paper's experiments use S5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Each task/split derives a deterministic child seed.",
    )
    parser.add_argument(
        "--no-shuffle-sequence",
        action="store_true",
        help="Keep clauses in dependency order instead of shuffling them.",
    )


def output_path(output_root: Path, task: str, split: str, length: int) -> Path:
    if task == "cyclic":
        return output_root / "cyclic_lego_dataset" / f"cyclic_{split}_length_{length}.json"
    if task == "symmetric":
        return output_root / "symmetric_lego_dataset" / f"symmetric_{split}_length_{length}.json"
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_task_args(parser)
    args = parser.parse_args()

    lengths = [args.num_vars] if args.num_vars is not None else parse_lengths(args.lengths)
    if any(length < 2 for length in lengths):
        raise ValueError("All --lengths values must be at least 2.")
    if args.cyclic_order < 2:
        raise ValueError("--cyclic-order must be at least 2.")
    if args.symmetric_degree < 2:
        raise ValueError("--symmetric-degree must be at least 2.")

    shuffle_sequence = not args.no_shuffle_sequence
    split_offsets = {"train": 11, "val": 17, "test": 23}
    task_offsets = {"cyclic": 101, "symmetric": 211}

    for task in args.tasks:
        group_size = args.cyclic_order if task == "cyclic" else args.symmetric_degree
        for split, size in args.split_sizes.items():
            for length in lengths:
                seed = (
                    args.seed
                    + task_offsets[task]
                    + split_offsets[split]
                    + length * 1009
                )
                rng = random.Random(seed)
                records = generate_split(
                    task=task,
                    size=size,
                    rng=rng,
                    length=length,
                    group_size=group_size,
                    shuffle_sequence=shuffle_sequence,
                )
                path = output_path(args.output_root, task, split, length)
                write_json(path, records)
                print(
                    f"Wrote {len(records):>5} {task:9s} {split:5s} "
                    f"records for length {length} to {path}"
                )


if __name__ == "__main__":
    main()
