"""
cli.py — command-line interface for docwhisper.

Usage:
    docwhisper ingest --docs-dir ./docs
    docwhisper ask "What is the refund policy?"
    docwhisper ask "How do I reset my password?" --docs-dir ./my_docs
"""

import argparse
import logging
import sys
from pathlib import Path


def cmd_ingest(args):
    from .pipeline import DocWhisper
    dw = DocWhisper(
        docs_dir=Path(args.docs_dir) if args.docs_dir else None,
        index_dir=Path(args.index_dir) if args.index_dir else None,
    )
    dw.ingest()
    print(f"\n✓ Done! Index saved. Run: docwhisper ask \"your question\"")


def cmd_ask(args):
    from .pipeline import DocWhisper
    dw = DocWhisper(
        docs_dir=Path(args.docs_dir) if args.docs_dir else None,
        index_dir=Path(args.index_dir) if args.index_dir else None,
    )
    answer = dw.ask(args.question)
    print(answer.format())

    if not answer.has_citations:
        print("⚠  Warning: answer may not be grounded in your documents.")
        sys.exit(1)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="docwhisper",
        description="Ask questions about your documents — with cited answers.",
    )
    parser.add_argument("--docs-dir", default=None, help="Directory containing your documents")
    parser.add_argument("--index-dir", default=None, help="Where to store/load the index")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Index documents")
    ingest_p.set_defaults(func=cmd_ingest)

    # ask
    ask_p = subparsers.add_parser("ask", help="Ask a question")
    ask_p.add_argument("question", help="Your question in quotes")
    ask_p.set_defaults(func=cmd_ask)

    args = parser.parse_args()

    # pass global flags down to subcommand args (a bit hacky but simple)
    if not hasattr(args, "docs_dir"):
        args.docs_dir = None
    if not hasattr(args, "index_dir"):
        args.index_dir = None

    args.func(args)


if __name__ == "__main__":
    main()
