import argparse
import json
from datetime import datetime, timezone

from openai import OpenAI


def _to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return getattr(obj, "__dict__", {})


def _fmt_ts(ts):
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _get(b, key, default="-"):
    val = b.get(key)
    if val in (None, ""):
        return default
    return val


def list_batches(client, limit=20, after=None):
    page = client.batches.list(limit=limit, after=after)
    if hasattr(page, "data"):
        data = page.data
    else:
        data = list(page)
    next_after = data[-1].id if data else None
    return data, next_after


def print_batch_list(batches):
    if not batches:
        print("No batches found.")
        return
    print("Recent batches:")
    for i, batch in enumerate(batches, start=1):
        b = _to_dict(batch)
        counts = b.get("request_counts") or {}
        total = counts.get("total", "-")
        completed = counts.get("completed", "-")
        failed = counts.get("failed", "-")
        print(
            f"{i:>2}. {b.get('id', '-')}"
            f" | status={_get(b, 'status')}"
            f" | created={_fmt_ts(b.get('created_at'))}"
            f" | counts={completed}/{failed}/{total}"
        )


def print_batch_details(batch):
    b = _to_dict(batch)
    counts = b.get("request_counts") or {}
    print("")
    print("Batch details")
    print(f"  id: {_get(b, 'id')}")
    print(f"  status: {_get(b, 'status')}")
    print(f"  endpoint: {_get(b, 'endpoint')}")
    print(f"  model: {_get(b, 'model')}")
    print(f"  input_file_id: {_get(b, 'input_file_id')}")
    print(f"  output_file_id: {_get(b, 'output_file_id')}")
    print(f"  error_file_id: {_get(b, 'error_file_id')}")
    print(f"  created_at: {_fmt_ts(b.get('created_at'))}")
    print(f"  in_progress_at: {_fmt_ts(b.get('in_progress_at'))}")
    print(f"  completed_at: {_fmt_ts(b.get('completed_at'))}")
    print(f"  failed_at: {_fmt_ts(b.get('failed_at'))}")
    print(f"  expired_at: {_fmt_ts(b.get('expired_at'))}")
    print(
        "  request_counts:"
        f" completed={counts.get('completed', '-')}"
        f" failed={counts.get('failed', '-')}"
        f" total={counts.get('total', '-')}"
    )
    errors = b.get("errors") or []
    if errors:
        print("  errors:")
        for err in errors:
            e = _to_dict(err)
            code = e.get("code", "-")
            msg = e.get("message", "-")
            param = e.get("param", "-")
            print(f"    - code={code} param={param} message={msg}")
    print("")


def print_file_content(client, file_id, max_lines=200, full=False):
    if not file_id:
        print("No file id provided.")
        return
    file_response = client.files.content(file_id)
    text = file_response.text if hasattr(file_response, "text") else str(file_response)
    if full:
        print(text)
        return

    lines = text.splitlines()
    head = lines[:max_lines]
    print("\n".join(head))
    if len(lines) > max_lines:
        print("")
        print(f"... truncated ({len(lines)} total lines). Use --full to print all.")


def summarize_file_text(text):
    lines = text.splitlines()
    total_lines = len(lines)
    total_bytes = len(text.encode("utf-8"))

    json_lines = 0
    parse_errors = 0
    response_count = 0
    error_count = 0
    status_counts = {}
    model_counts = {}
    error_type_counts = {}
    custom_id_count = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            json_lines += 1
        except json.JSONDecodeError:
            parse_errors += 1
            continue

        if "custom_id" in obj and obj.get("custom_id"):
            custom_id_count += 1

        err = obj.get("error")
        if err:
            error_count += 1
            if isinstance(err, dict):
                etype = err.get("type") or err.get("code") or "unknown"
            else:
                etype = "unknown"
            error_type_counts[etype] = error_type_counts.get(etype, 0) + 1

        resp = obj.get("response")
        if resp:
            response_count += 1
            status = None
            if isinstance(resp, dict):
                status = resp.get("status_code") or resp.get("status")
                body = resp.get("body")
                if isinstance(body, dict):
                    model = body.get("model")
                    if model:
                        model_counts[model] = model_counts.get(model, 0) + 1
            if status is None:
                status = "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1

    print("")
    print("File summary")
    print(f"  lines: {total_lines}")
    print(f"  bytes: {total_bytes}")
    print(f"  jsonl_lines: {json_lines}")
    print(f"  parse_errors: {parse_errors}")
    print(f"  custom_id_lines: {custom_id_count}")
    print(f"  responses: {response_count}")
    print(f"  errors: {error_count}")
    if status_counts:
        items = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        print(f"  status_counts: {items}")
    if error_type_counts:
        items = ", ".join(
            f"{k}={v}" for k, v in sorted(error_type_counts.items())
        )
        print(f"  error_types: {items}")
    if model_counts:
        items = ", ".join(f"{k}={v}" for k, v in sorted(model_counts.items()))
        print(f"  models: {items}")
    if json_lines == 0 and total_lines > 0:
        first_nonempty = next((l for l in lines if l.strip()), "")
        snippet = first_nonempty[:200]
        print(f"  first_line: {snippet}")


def fetch_file_text(client, file_id):
    file_response = client.files.content(file_id)
    return file_response.text if hasattr(file_response, "text") else str(file_response)


def interactive_loop(client, limit=20, after=None, max_lines=200, full=False):
    cursor_stack = []
    while True:
        batches, next_after = list_batches(client, limit=limit, after=after)
        print_batch_list(batches)
        print("")
        print("Select by number or paste a batch id.")
        print("Commands: r=refresh, n=next page, p=prev page, f=file id, q=quit")
        choice = input("> ").strip()

        if not choice:
            continue
        if choice.lower() == "q":
            return
        if choice.lower() == "r":
            continue
        if choice.lower() == "n":
            if not next_after:
                print("No next page.")
                continue
            cursor_stack.append(after)
            after = next_after
            continue
        if choice.lower() == "p":
            if not cursor_stack:
                print("No previous page.")
                continue
            after = cursor_stack.pop()
            continue
        if choice.lower() == "f":
            file_id = input("file id> ").strip()
            if file_id:
                text = fetch_file_text(client, file_id)
                summarize_file_text(text)
            continue

        batch_id = None
        if choice.startswith("file-") or choice.startswith("file_"):
            file_id = choice
            text = fetch_file_text(client, file_id)
            summarize_file_text(text)
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(batches):
                batch_id = _to_dict(batches[idx - 1]).get("id")
        else:
            batch_id = choice

        if not batch_id:
            print("Invalid selection.")
            continue

        batch = client.batches.retrieve(batch_id)
        print_batch_details(batch)
        b = _to_dict(batch)
        while True:
            print("File shortcuts: o=output, e=error, i=input, b=back")
            file_choice = input("file> ").strip().lower()
            if not file_choice or file_choice == "b":
                break
            if file_choice == "o":
                file_id = b.get("output_file_id")
            elif file_choice == "e":
                file_id = b.get("error_file_id")
            elif file_choice == "i":
                file_id = b.get("input_file_id")
            else:
                print("Unknown file option.")
                continue
            if not file_id:
                print("No file id available for that option.")
                continue
            text = fetch_file_text(client, file_id)
            summarize_file_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="List and inspect OpenAI batches interactively."
    )
    parser.add_argument("--limit", type=int, default=20, help="Batches per page.")
    parser.add_argument("--after", default=None, help="Cursor to start listing after.")
    parser.add_argument(
        "--batch",
        default=None,
        help="Batch id to inspect directly (skips interactive list).",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="File id to print directly (skips interactive list).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print file content after summary when using --file.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="Max lines to print when showing file content.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full file content (no truncation).",
    )
    args = parser.parse_args()

    client = OpenAI()
    if args.file:
        text = fetch_file_text(client, args.file)
        summarize_file_text(text)
        if args.show:
            print("")
            print_file_content(
                client, args.file, max_lines=args.max_lines, full=args.full
            )
        return
    if args.batch:
        batch = client.batches.retrieve(args.batch)
        print_batch_details(batch)
        return

    interactive_loop(
        client,
        limit=args.limit,
        after=args.after,
        max_lines=args.max_lines,
        full=args.full,
    )


if __name__ == "__main__":
    main()
