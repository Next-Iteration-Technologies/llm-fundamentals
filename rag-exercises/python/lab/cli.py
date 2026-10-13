"""The lab's commands, added to rag.py's parser. See LAB.md for the story and .plan/SPEC.md for the design."""
import sys


def add_commands(sub):
    p = sub.add_parser("selftest", help="the build checks: PASS/FAIL per check")
    p.add_argument("--upto", type=int, default=99, help="run the checks of steps 0..N")
    p.add_argument("--only", metavar="TEXT", help="only checks whose name contains TEXT")
    p.add_argument("--save-baseline", action="store_true", help="record today's Contoso numbers in .plan/baseline.json")


COMMANDS = {"selftest"}


def run(a):
    if a.cmd == "selftest":
        from . import selftest
        if a.save_baseline:
            selftest.save_baseline()
            return 0
        return selftest.run(a.upto, a.only)
