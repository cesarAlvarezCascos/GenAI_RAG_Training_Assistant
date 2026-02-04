#!/usr/bin/env python3
"""Call the `ask` function defined in `api.main` directly from the command line.

This bypasses HTTP and uses the same code-path as the FastAPI endpoint (imports
the Ask model, calls the function), so it uses `search_kb` and the OpenAI client
set up in `api.main`.
"""
import argparse
import json
import sys

try:
    # Import the Ask model and the ask function from the API module
    from api.main import Ask, ask as ask_func
except Exception as e:
    print(f"Failed to import api.main: {e}", file=sys.stderr)
    raise


def main():
    parser = argparse.ArgumentParser(description='Call api.main.ask locally using the same infrastructure')
    parser.add_argument('--role', '-r', default='analyst', help='Role (default: analyst)')
    parser.add_argument('--user_id', '-u', default=None, help='Optional user_id')
    parser.add_argument('--product_version', '-p', default=None, help='Optional product_version')
    parser.add_argument('--time_budget', '-t', type=int, default=30, help='Optional time_budget (int)')
    parser.add_argument('--level', '-l', type=int, default=2, help='Optional level (int)')
    parser.add_argument('query', nargs='+', help='Query text (wrap in quotes)')

    args = parser.parse_args()
    query_text = ' '.join(args.query)

    req = Ask(
        user_id=args.user_id,
        role=args.role,
        query=query_text,
        product_version=args.product_version,
        time_budget=args.time_budget,
        level=args.level,
    )

    try:
        resp = ask_func(req)
    except Exception as e:
        print(f"Error calling api.main.ask: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    # Pretty-print result
    print('\n=== Answer ===\n')
    print(resp.get('answer', '<no answer>'))

    print('\n=== Citations ===\n')
    print(json.dumps(resp.get('citations', []), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
