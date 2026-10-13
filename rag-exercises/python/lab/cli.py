"""The lab's commands, added to rag.py's parser. See LAB.md for the story and .plan/SPEC.md for the design.

Every lab command takes --dataset NAME (datasets/NAME.toml). Without it: RAG_DATASET, and without that: contoso.
"""
import sys


def dataset_arg(p):
    p.add_argument("--dataset", metavar="NAME", help="datasets/NAME.toml (default: RAG_DATASET, else contoso)")


def add_commands(sub):
    p = sub.add_parser("selftest", help="the build checks: PASS/FAIL per check")
    p.add_argument("--upto", type=int, default=99, help="run the checks of steps 0..N")
    p.add_argument("--only", metavar="TEXT", help="only checks whose name contains TEXT")
    p.add_argument("--save-baseline", action="store_true", help="record today's Contoso numbers in .plan/baseline.json")

    p = sub.add_parser("check-data", help="is every gold passage where the golden set says?")
    dataset_arg(p)


COMMANDS = {"selftest", "check-data"}


def run(a):
    if a.cmd == "selftest":
        from . import selftest
        if a.save_baseline:
            selftest.save_baseline()
            return 0
        return selftest.run(a.upto, a.only)
    if a.cmd == "check-data":
        from . import datasets
        return 1 if datasets.check_data(datasets.load(a.dataset)) else 0
