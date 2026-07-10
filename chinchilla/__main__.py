"""CLI for Chinchilla scaling experiments"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Chinchilla scaling-law experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    sweep_p = sub.add_parser("sweep", help="Run training sweep")
    sweep_p.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced grid (3 presets x 3 budgest x 2 archs)",
    )
    sweep_p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and re-run all",
    )

    sub.add_parser("report", help="Generate markdown report from results")

    args = parser.parse_args()
    if args.command == "sweep":
        from chinchilla.sweep import quick_sweep, run_sweep

        if args.quick:
            quick_sweep()
        else:
            run_sweep(resume=not args.no_resume)
    elif args.command == "report":
        from chinchilla.report import generate_report

        generate_report()

if __name__=="__main__":
    main()
