#!/usr/bin/env python3
"""Static file server + API for paged Tiny VQA visualization queries.

This server keeps the large JSON on the backend, applies filters server-side,
returns only paged rows to the browser, and supports gzip for JSON responses.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

OBJECT_COUNT_RE = re.compile(r"/(?:random|random-cam-stationary)/([^/]+)")


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_model_name(file_name: str) -> str:
    if file_name.endswith("_val.json"):
        return file_name[: -len("_val.json")]
    return Path(file_name).stem


def get_object_count(simulation_id: str) -> str:
    match = OBJECT_COUNT_RE.search(simulation_id or "")
    return match.group(1) if match else ""


def parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_image_type(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in {"single", "multi"} else ""


def object_count_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def parse_scene_ids(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        for chunk in str(raw or "").split(","):
            scene_id = chunk.strip()
            if not scene_id or scene_id in seen:
                continue
            seen.add(scene_id)
            ordered.append(scene_id)
    return tuple(sorted(ordered))


@dataclass
class FilteredResult:
    indices: list[int]
    summary_counts: dict[str, int]
    summary_total_with_correct: int


class DatasetState:
    def __init__(
        self,
        *,
        run_name: str,
        question_file: Path,
        results_dirs: list[Path],
        correct_answers_file: Path | None,
        scenes_file: Path | None,
        filter_cache_size: int,
    ) -> None:
        self.run_name = run_name
        self.question_file = question_file
        self.results_dirs = results_dirs
        self.correct_answers_file = correct_answers_file
        self.scenes_file = scenes_file
        self.filter_cache_size = max(1, filter_cache_size)

        self._lock = threading.RLock()
        self._questions_loaded = False
        self._correct_loaded = False
        self._answers_loaded = False
        self._model_names_loaded = False

        self.questions: list[dict[str, Any]] = []
        self.idx_values: list[str] = []
        self.search_haystacks: list[str] = []
        self.question_id_index: dict[str, list[int]] = {}
        self.object_count_index: dict[str, list[int]] = {}
        self.image_type_index: dict[str, list[int]] = {}
        self.scene_index: dict[str, list[int]] = {}
        self.question_ids: list[str] = []
        self.object_counts: list[str] = []
        self.scene_ids: list[str] = []
        self.scene_info_by_id: dict[str, dict[str, str]] = {}

        self.correct_by_idx: dict[str, str] = {}
        self.answers_by_idx: dict[str, tuple[tuple[str, str], ...]] = {}
        self.model_names: list[str] = []

        self.filter_cache: OrderedDict[tuple[str, str, str, str, tuple[str, ...]], FilteredResult] = OrderedDict()

        self._load_scene_info()

    def _load_scene_info(self) -> None:
        if self.scenes_file is None or not self.scenes_file.is_file():
            self.scene_info_by_id = {}
            return
        try:
            raw = read_json(self.scenes_file)
        except Exception:
            self.scene_info_by_id = {}
            return
        if not isinstance(raw, dict):
            self.scene_info_by_id = {}
            return

        info: dict[str, dict[str, str]] = {}
        for scene_id, value in raw.items():
            entry = value if isinstance(value, dict) else {}
            info[str(scene_id)] = {
                "scene_id": str(scene_id),
                "name": str(entry.get("name") or ""),
                "batch": str(entry.get("batch") or ""),
                "comment": str(entry.get("comment") or ""),
            }
        self.scene_info_by_id = info

    def _ensure_questions_loaded(self) -> None:
        with self._lock:
            if self._questions_loaded:
                return

        started = time.time()
        raw = read_json(self.question_file)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list in {self.question_file}, got {type(raw).__name__}")

        question_id_index: dict[str, list[int]] = defaultdict(list)
        object_count_index: dict[str, list[int]] = defaultdict(list)
        image_type_index: dict[str, list[int]] = {"single": [], "multi": []}
        scene_index: dict[str, list[int]] = defaultdict(list)
        idx_values: list[str] = []
        search_haystacks: list[str] = []

        for i, item in enumerate(raw):
            idx = str(item.get("idx") or "")
            question_id = str(item.get("question_id") or "")
            question = str(item.get("question") or "")
            simulation_id = str(item.get("simulation_id") or "")
            scene_id = str(item.get("scene") or "")

            idx_values.append(idx)
            search_haystacks.append(f"{idx} {question} {question_id}".lower())

            if question_id:
                question_id_index[question_id].append(i)
            object_count = get_object_count(simulation_id)
            if object_count:
                object_count_index[object_count].append(i)
            if idx.endswith("_i"):
                image_type_index["single"].append(i)
            elif idx.endswith("_g"):
                image_type_index["multi"].append(i)
            if scene_id:
                scene_index[scene_id].append(i)

        with self._lock:
            if self._questions_loaded:
                return
            self.questions = raw
            self.idx_values = idx_values
            self.search_haystacks = search_haystacks
            self.question_id_index = dict(question_id_index)
            self.object_count_index = dict(object_count_index)
            self.image_type_index = image_type_index
            self.scene_index = dict(scene_index)
            self.question_ids = sorted(self.question_id_index.keys())
            self.object_counts = sorted(self.object_count_index.keys(), key=object_count_sort_key)
            self.scene_ids = sorted(self.scene_index.keys())
            self._questions_loaded = True
            elapsed = time.time() - started
            print(
                f"[load] questions: {len(self.questions)} rows from {self.question_file} "
                f"in {elapsed:.2f}s"
            )

    def _discover_model_names(self) -> None:
        with self._lock:
            if self._model_names_loaded:
                return

        model_names: set[str] = set()
        for results_dir in self.results_dirs:
            if not results_dir.is_dir():
                continue
            for path in results_dir.glob("*_val.json"):
                model_names.add(to_model_name(path.name))

        with self._lock:
            if self._model_names_loaded:
                return
            self.model_names = sorted(model_names)
            self._model_names_loaded = True

    def _ensure_correct_loaded(self) -> None:
        with self._lock:
            if self._correct_loaded:
                return

        selected = self.correct_answers_file
        if selected is None:
            candidates = sorted(self.question_file.parent.glob("val_answer*.json"))
            selected = candidates[0] if candidates else None

        if selected is None or not selected.is_file():
            with self._lock:
                self.correct_by_idx = {}
                self._correct_loaded = True
            return

        started = time.time()
        raw = read_json(selected)
        correct_by_idx: dict[str, str] = {}
        if isinstance(raw, list):
            for entry in raw:
                idx = str((entry or {}).get("idx") or "")
                if not idx:
                    continue
                correct_by_idx[idx] = str((entry or {}).get("answer") or "")

        with self._lock:
            if self._correct_loaded:
                return
            self.correct_answers_file = selected
            self.correct_by_idx = correct_by_idx
            self._correct_loaded = True
            elapsed = time.time() - started
            print(f"[load] correct answers: {len(self.correct_by_idx)} rows from {selected} in {elapsed:.2f}s")

    def _ensure_answers_loaded(self) -> None:
        with self._lock:
            if self._answers_loaded:
                return

        started = time.time()
        answers_temp: dict[str, dict[str, str]] = defaultdict(dict)
        model_names: set[str] = set()

        for results_dir in self.results_dirs:
            if not results_dir.is_dir():
                continue
            for path in sorted(results_dir.glob("*_val.json")):
                model_name = to_model_name(path.name)
                model_names.add(model_name)
                raw = read_json(path)
                if not isinstance(raw, list):
                    continue
                for entry in raw:
                    idx = str((entry or {}).get("idx") or "")
                    if not idx:
                        continue
                    answer = str((entry or {}).get("answer") or "")
                    answers_temp[idx][model_name] = answer

        compact: dict[str, tuple[tuple[str, str], ...]] = {}
        for idx, model_map in answers_temp.items():
            compact[idx] = tuple(sorted(model_map.items(), key=lambda pair: pair[0]))

        with self._lock:
            if self._answers_loaded:
                return
            self.answers_by_idx = compact
            self.model_names = sorted(model_names)
            self._model_names_loaded = True
            self._answers_loaded = True
            elapsed = time.time() - started
            print(
                f"[load] model answers: {len(self.answers_by_idx)} idx rows, "
                f"{len(self.model_names)} models in {elapsed:.2f}s"
            )

    def _compute_filtered_indices(
        self,
        *,
        question_id: str,
        search: str,
        image_type: str,
        object_count: str,
        selected_scene_ids: tuple[str, ...],
    ) -> list[int]:
        self._ensure_questions_loaded()

        candidates: list[list[int]] = []
        if question_id:
            candidates.append(self.question_id_index.get(question_id, []))
        if object_count:
            candidates.append(self.object_count_index.get(object_count, []))
        if image_type:
            candidates.append(self.image_type_index.get(image_type, []))
        if selected_scene_ids:
            scene_lists = [self.scene_index.get(scene_id, []) for scene_id in selected_scene_ids]
            scene_lists = [values for values in scene_lists if values]
            if not scene_lists:
                return []
            if len(scene_lists) == 1:
                candidates.append(list(scene_lists[0]))
            else:
                scene_union: set[int] = set()
                for values in scene_lists:
                    scene_union.update(values)
                candidates.append(sorted(scene_union))

        if candidates:
            candidates.sort(key=len)
            filtered = list(candidates[0])
            for values in candidates[1:]:
                allowed = set(values)
                filtered = [idx for idx in filtered if idx in allowed]
                if not filtered:
                    break
        else:
            filtered = list(range(len(self.questions)))

        if search:
            needle = search.lower()
            filtered = [idx for idx in filtered if needle in self.search_haystacks[idx]]

        return filtered

    def _compute_summary(self, indices: list[int]) -> tuple[dict[str, int], int]:
        self._ensure_correct_loaded()
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        total_with_correct = 0
        for i in indices:
            idx = self.idx_values[i]
            answer = self.correct_by_idx.get(idx)
            if answer in counts:
                counts[answer] += 1
                total_with_correct += 1
        return counts, total_with_correct

    def get_filtered_result(
        self,
        *,
        question_id: str,
        search: str,
        image_type: str,
        object_count: str,
        selected_scene_ids: tuple[str, ...],
    ) -> tuple[FilteredResult, bool]:
        key = (question_id, search.strip().lower(), image_type, object_count, selected_scene_ids)
        with self._lock:
            cached = self.filter_cache.get(key)
            if cached is not None:
                self.filter_cache.move_to_end(key)
                return cached, True

        indices = self._compute_filtered_indices(
            question_id=question_id,
            search=search,
            image_type=image_type,
            object_count=object_count,
            selected_scene_ids=selected_scene_ids,
        )
        counts, total_with_correct = self._compute_summary(indices)
        result = FilteredResult(indices=indices, summary_counts=counts, summary_total_with_correct=total_with_correct)

        with self._lock:
            self.filter_cache[key] = result
            self.filter_cache.move_to_end(key)
            while len(self.filter_cache) > self.filter_cache_size:
                self.filter_cache.popitem(last=False)
        return result, False

    def list_runs(self) -> list[str]:
        return [self.run_name]

    def get_metadata(self) -> dict[str, Any]:
        self._ensure_questions_loaded()
        self._discover_model_names()
        self._ensure_correct_loaded()
        scene_options: list[dict[str, str]] = []
        for scene_id in self.scene_ids:
            scene_info = self.scene_info_by_id.get(scene_id, {})
            scene_options.append(
                {
                    "scene_id": scene_id,
                    "name": scene_info.get("name", ""),
                    "batch": scene_info.get("batch", ""),
                    "comment": scene_info.get("comment", ""),
                }
            )
        return {
            "run": self.run_name,
            "question_set": "full",
            "available_question_sets": ["full"],
            "question_file": str(self.question_file),
            "total_questions": len(self.questions),
            "question_ids": self.question_ids,
            "object_counts": self.object_counts,
            "scene_ids": self.scene_ids,
            "scene_options": scene_options,
            "model_count": len(self.model_names),
            "results_dirs": [str(path) for path in self.results_dirs if path.is_dir()],
            "correct_answers_file": str(self.correct_answers_file) if self.correct_answers_file else None,
        }

    def get_questions_page(
        self,
        *,
        page: int,
        page_size: int,
        question_id: str,
        search: str,
        image_type: str,
        object_count: str,
        selected_scene_ids: tuple[str, ...],
        include_answers: bool,
    ) -> dict[str, Any]:
        self._ensure_questions_loaded()
        if include_answers:
            self._ensure_answers_loaded()
        else:
            self._discover_model_names()
        self._ensure_correct_loaded()

        filtered, cache_hit = self.get_filtered_result(
            question_id=question_id,
            search=search,
            image_type=image_type,
            object_count=object_count,
            selected_scene_ids=selected_scene_ids,
        )

        total = len(filtered.indices)
        page_count = max(1, math.ceil(total / page_size)) if total > 0 else 1
        page = max(1, min(page, page_count))
        start = (page - 1) * page_size
        end = start + page_size
        page_indices = filtered.indices[start:end]

        items: list[dict[str, Any]] = []
        for i in page_indices:
            row = dict(self.questions[i])
            idx = self.idx_values[i]
            row["correct_answer"] = self.correct_by_idx.get(idx)
            if include_answers:
                model_pairs = self.answers_by_idx.get(idx, ())
                row["model_answers"] = [{"model": model, "answer": answer} for model, answer in model_pairs]
            else:
                row["model_answers"] = []
            items.append(row)

        return {
            "run": self.run_name,
            "question_set": "full",
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "total": total,
            "selected_scene_ids": list(selected_scene_ids),
            "model_count": len(self.model_names),
            "items": items,
            "summary": {
                "counts": filtered.summary_counts,
                "total_with_correct": filtered.summary_total_with_correct,
            },
            "cache_hit": cache_hit,
        }

    def get_summary(
        self,
        *,
        question_id: str,
        search: str,
        image_type: str,
        object_count: str,
        selected_scene_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        self._ensure_questions_loaded()
        filtered, cache_hit = self.get_filtered_result(
            question_id=question_id,
            search=search,
            image_type=image_type,
            object_count=object_count,
            selected_scene_ids=selected_scene_ids,
        )
        return {
            "run": self.run_name,
            "question_set": "full",
            "total": len(filtered.indices),
            "selected_scene_ids": list(selected_scene_ids),
            "summary": {
                "counts": filtered.summary_counts,
                "total_with_correct": filtered.summary_total_with_correct,
            },
            "cache_hit": cache_hit,
        }

    def get_answers_for_idx(self, idx: str) -> dict[str, Any]:
        idx = str(idx or "").strip()
        if not idx:
            raise ValueError("Missing idx parameter.")

        self._ensure_answers_loaded()
        self._ensure_correct_loaded()
        answers = [{"model": model, "answer": answer} for model, answer in self.answers_by_idx.get(idx, ())]
        correct_answer = self.correct_by_idx.get(idx)
        correct_count = 0
        if correct_answer:
            correct_count = sum(1 for entry in answers if entry["answer"] == correct_answer)
        return {
            "run": self.run_name,
            "idx": idx,
            "correct_answer": correct_answer,
            "answered_count": len(answers),
            "model_count": len(self.model_names),
            "correct_count": correct_count,
            "answers": answers,
        }


class VisualizationHandler(SimpleHTTPRequestHandler):
    server_version = "TinyVQAApiServer/1.0"

    def __init__(self, *args: Any, dataset: DatasetState, max_page_size: int, **kwargs: Any) -> None:
        self.dataset = dataset
        self.max_page_size = max_page_size
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path, parse_qs(parsed.query, keep_blank_values=True))
            return
        super().do_GET()

    def end_headers(self) -> None:
        # Disable browser caching for static HTML/JS while iterating quickly.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        try:
            if path == "/api/runs":
                self._send_json({"runs": self.dataset.list_runs()})
                return

            if path == "/api/metadata":
                self._assert_run_param(query)
                self._send_json(self.dataset.get_metadata())
                return

            if path == "/api/questions":
                self._assert_run_param(query)
                page = max(1, parse_int(self._get_query(query, "page", "1"), 1))
                page_size = max(1, parse_int(self._get_query(query, "page_size", "50"), 50))
                page_size = min(page_size, self.max_page_size)
                include_answers = self._get_query(query, "include_answers", "0").strip() not in {"0", "false", "False"}
                selected_scene_ids = parse_scene_ids(query.get("scene_id", []))

                response = self.dataset.get_questions_page(
                    page=page,
                    page_size=page_size,
                    question_id=self._get_query(query, "question_id", "").strip(),
                    search=self._get_query(query, "search", "").strip(),
                    image_type=normalize_image_type(self._get_query(query, "image_type", "")),
                    object_count=self._get_query(query, "object_count", "").strip(),
                    selected_scene_ids=selected_scene_ids,
                    include_answers=include_answers,
                )
                self._send_json(response)
                return

            if path == "/api/answers":
                self._assert_run_param(query)
                idx = self._get_query(query, "idx", "")
                response = self.dataset.get_answers_for_idx(idx)
                self._send_json(response)
                return

            if path == "/api/summary":
                self._assert_run_param(query)
                selected_scene_ids = parse_scene_ids(query.get("scene_id", []))
                response = self.dataset.get_summary(
                    question_id=self._get_query(query, "question_id", "").strip(),
                    search=self._get_query(query, "search", "").strip(),
                    image_type=normalize_image_type(self._get_query(query, "image_type", "")),
                    object_count=self._get_query(query, "object_count", "").strip(),
                    selected_scene_ids=selected_scene_ids,
                )
                self._send_json(response)
                return

            if path == "/api/health":
                payload = {
                    "status": "ok",
                    "run": self.dataset.run_name,
                    "questions_loaded": self.dataset._questions_loaded,
                    "answers_loaded": self.dataset._answers_loaded,
                    "correct_loaded": self.dataset._correct_loaded,
                    "filter_cache_entries": len(self.dataset.filter_cache),
                }
                self._send_json(payload)
                return

            self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError as err:
            self._send_json({"error": str(err)}, status=HTTPStatus.NOT_FOUND)
        except Exception as err:  # pylint: disable=broad-except
            self._send_json({"error": str(err)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _assert_run_param(self, query: dict[str, list[str]]) -> None:
        run_name = self._get_query(query, "run", "").strip()
        if run_name and run_name != self.dataset.run_name:
            raise FileNotFoundError(f"Run '{run_name}' is not available. Expected '{self.dataset.run_name}'.")

    @staticmethod
    def _get_query(query: dict[str, list[str]], key: str, default: str) -> str:
        values = query.get(key)
        if not values:
            return default
        return values[0]

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        supports_gzip = "gzip" in (self.headers.get("Accept-Encoding") or "").lower()
        use_gzip = supports_gzip and len(raw) >= 1024
        if use_gzip:
            raw = gzip.compress(raw, compresslevel=5)

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def build_results_dirs(run_dir: Path, run_name: str, explicit_dirs: list[str] | None) -> list[Path]:
    if explicit_dirs:
        return [Path(value).resolve() for value in explicit_dirs]
    dirs = [path for path in run_dir.glob(f"results_{run_name}*") if path.is_dir()]
    return sorted(dirs, key=lambda path: path.name)


def parse_args() -> argparse.Namespace:
    default_question = Path(
        "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
        "run_28_general/test_run_28_general.json"
    )
    default_scenes = Path(__file__).resolve().parent / "scenes.json"

    parser = argparse.ArgumentParser(description="Tiny VQA API + static server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8086, help="Bind port")
    parser.add_argument("--serve-root", default="/", help="Static file root (default: /)")
    parser.add_argument("--run-name", default="run_28_general", help="Run name exposed by the API")
    parser.add_argument(
        "--question-file",
        default=str(default_question),
        help="Absolute path to the question JSON file to serve",
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        default=None,
        help="Optional results directory (can be passed multiple times)",
    )
    parser.add_argument(
        "--correct-answers-file",
        default="",
        help="Optional explicit correct answers JSON path (defaults to auto-discovery)",
    )
    parser.add_argument(
        "--scenes-file",
        default=str(default_scenes),
        help="Optional scenes metadata JSON path (used for scene labels/filter menu)",
    )
    parser.add_argument("--max-page-size", type=int, default=200, help="Maximum page_size accepted by /api/questions")
    parser.add_argument(
        "--filter-cache-size",
        type=int,
        default=24,
        help="Number of filter results cached in memory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    question_file = Path(args.question_file).resolve()
    if not question_file.is_file():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    run_dir = question_file.parent
    results_dirs = build_results_dirs(run_dir, args.run_name, args.results_dir)
    correct_file = Path(args.correct_answers_file).resolve() if args.correct_answers_file else None
    scenes_file = Path(args.scenes_file).resolve() if args.scenes_file else None

    dataset = DatasetState(
        run_name=args.run_name,
        question_file=question_file,
        results_dirs=results_dirs,
        correct_answers_file=correct_file,
        scenes_file=scenes_file,
        filter_cache_size=args.filter_cache_size,
    )

    handler = lambda *handler_args, **handler_kwargs: VisualizationHandler(  # noqa: E731
        *handler_args,
        dataset=dataset,
        max_page_size=args.max_page_size,
        directory=args.serve_root,
        **handler_kwargs,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"[start] listening on http://{args.host}:{args.port} | run={args.run_name} "
        f"| question_file={question_file}"
    )
    print(f"[start] static root: {args.serve_root}")
    if results_dirs:
        print(f"[start] results dirs: {len(results_dirs)} discovered")
    else:
        print("[start] results dirs: none discovered (answer lists will be empty)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
