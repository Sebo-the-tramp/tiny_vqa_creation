#!/usr/bin/env python3
"""FastAPI server for paged Tiny VQA visualization."""

from __future__ import annotations

import math
import mimetypes
import os
import re
import sys
import time
import zipfile
from collections import OrderedDict, defaultdict
from io import BytesIO
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import json
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response

def _import_pillow() -> tuple[bool, Any, Any, Any]:
    try:
        from PIL import Image as _Image
        from PIL import ImageOps as _ImageOps
        from PIL import UnidentifiedImageError as _UnidentifiedImageError

        return True, _Image, _ImageOps, _UnidentifiedImageError
    except Exception:
        pass

    # Fallback: some environments install Pillow only in system dist-packages.
    for extra_site in (
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3.12/dist-packages",
    ):
        if extra_site not in sys.path and Path(extra_site).is_dir():
            sys.path.append(extra_site)

    try:
        from PIL import Image as _Image
        from PIL import ImageOps as _ImageOps
        from PIL import UnidentifiedImageError as _UnidentifiedImageError

        return True, _Image, _ImageOps, _UnidentifiedImageError
    except Exception:
        class _UnidentifiedImageError(Exception):
            """Fallback placeholder when Pillow is unavailable."""

        return False, None, None, _UnidentifiedImageError


PIL_AVAILABLE, Image, ImageOps, UnidentifiedImageError = _import_pillow()


DEFAULT_QUESTION_FILE = (
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
    "run_28_general/test_run_28_general.json"
)
DEFAULT_ANSWER_FILE = (
    "/data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output/"
    "run_28_general/val_answer_run_26_general.json"
)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENES_FILE = str(BASE_DIR / "scenes.json")
DEFAULT_EXCLUDE_FILE = str(BASE_DIR / "exclude_scenes.txt")

OBJECT_COUNT_FROM_SCENE_RE = re.compile(r"/(?:random|random-cam-stationary)/(\d+)(?:/|$)")
OBJECT_COUNT_FROM_NAME_RE = re.compile(r"(?:^|[_/])no-(\d+)(?:[_/]|$)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_PREVIEW_FORMATS = {"webp", "jpeg", "orig"}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_object_count(file_names: list[str]) -> str:
    for file_name in file_names:
        scene_match = OBJECT_COUNT_FROM_SCENE_RE.search(file_name)
        if scene_match:
            return scene_match.group(1)
    for file_name in file_names:
        name_match = OBJECT_COUNT_FROM_NAME_RE.search(file_name)
        if name_match:
            return name_match.group(1)
    return ""


def _sort_object_count(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (ValueError, TypeError):
        return (1, value)


def _parse_scene_ids(raw_values: list[str] | None) -> tuple[str, ...]:
    if not raw_values:
        return ()
    scene_ids: set[str] = set()
    for raw in raw_values:
        for part in str(raw or "").split(","):
            scene_id = part.strip()
            if scene_id:
                scene_ids.add(scene_id)
    return tuple(sorted(scene_ids))


def _safe_name(value: str, default: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", str(value or "")).strip("._")
    return cleaned if cleaned else default


def _placeholder_svg(width: int, height: int, message: str = "image not found") -> bytes:
    safe_width = max(64, min(4096, int(width)))
    safe_height = max(48, min(4096, int(height)))
    safe_message = (
        str(message or "image not found")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{safe_width}" height="{safe_height}" '
        f'viewBox="0 0 {safe_width} {safe_height}">'
        '<defs>'
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#eef1f5"/>'
        '<stop offset="100%" stop-color="#d8dee7"/>'
        "</linearGradient>"
        "</defs>"
        f'<rect x="0" y="0" width="{safe_width}" height="{safe_height}" fill="url(#bg)"/>'
        f'<text x="{safe_width / 2}" y="{safe_height / 2}" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="14" fill="#5b6572">'
        f"{safe_message}"
        "</text>"
        "</svg>"
    )
    return svg.encode("utf-8")


class DatasetStore:
    def __init__(
        self,
        *,
        question_file: Path,
        answer_file: Path,
        scenes_file: Path,
        exclude_file: Path,
        filter_cache_size: int = 64,
    ) -> None:
        self.question_file = question_file
        self.answer_file = answer_file
        self.scenes_file = scenes_file
        self.exclude_file = exclude_file
        self.filter_cache_size = max(1, filter_cache_size)

        self.items: list[dict[str, Any]] = []
        self.idx_index: dict[str, int] = {}
        self.question_id_index: dict[str, list[int]] = {}
        self.object_count_index: dict[str, list[int]] = {}
        self.scene_index: dict[str, list[int]] = {}

        self.question_ids: list[str] = []
        self.object_counts: list[str] = []
        self.scene_ids: list[str] = []
        self.scene_options: list[dict[str, str]] = []
        self.excluded_scene_ids: set[str] = set()
        self.default_selected_scene_ids: list[str] = []

        self._all_indices: list[int] = []
        self.image_available_index: list[int] = []
        self.rows_with_available_images = 0
        self.rows_without_available_images = 0
        self._filter_cache: OrderedDict[
            tuple[str, str, str, tuple[str, ...], bool],
            list[int],
        ] = OrderedDict()
        self._cache_lock = RLock()

        self._load_all()

    def _load_all(self) -> None:
        started = time.time()

        answer_by_idx = self._load_answer_map(self.answer_file)
        scene_meta = self._load_scene_meta(self.scenes_file)
        self.excluded_scene_ids = self._load_excluded_scene_ids(self.exclude_file)

        raw_questions = _read_json(self.question_file)
        if not isinstance(raw_questions, list):
            raise ValueError(f"Question file must contain an array: {self.question_file}")

        question_id_index: dict[str, list[int]] = defaultdict(list)
        object_count_index: dict[str, list[int]] = defaultdict(list)
        scene_index: dict[str, list[int]] = defaultdict(list)
        file_exists_cache: dict[str, bool] = {}

        for entry in raw_questions:
            if not isinstance(entry, dict):
                continue

            idx = str(entry.get("idx") or "")
            question_id = str(entry.get("question_id") or "")
            question = str(entry.get("question") or "")
            scene_id = str(entry.get("scene") or "")

            file_name_raw = entry.get("file_name")
            if isinstance(file_name_raw, list):
                raw_file_names = [str(value) for value in file_name_raw if value is not None]
            elif file_name_raw:
                raw_file_names = [str(file_name_raw)]
            else:
                raw_file_names = []

            available_file_names: list[str] = []
            for file_name in raw_file_names:
                exists = file_exists_cache.get(file_name)
                if exists is None:
                    exists = Path(file_name).is_file()
                    file_exists_cache[file_name] = exists
                if exists:
                    available_file_names.append(file_name)
            missing_image_count = max(0, len(raw_file_names) - len(available_file_names))

            object_count = _to_object_count(raw_file_names)
            image_count = len(raw_file_names)
            item_type = "single" if image_count <= 1 else "multi"

            item = {
                "idx": idx,
                "question_id": question_id,
                "question": question,
                "scene": scene_id,
                "file_name": available_file_names,
                "file_name_raw": raw_file_names,
                "image_count": image_count,
                "available_image_count": len(available_file_names),
                "missing_image_count": missing_image_count,
                "has_available_images": bool(available_file_names),
                "item_type": item_type,
                "object_count": object_count,
                "correct_answer": answer_by_idx.get(idx, ""),
            }
            pos = len(self.items)
            self.items.append(item)
            if item["has_available_images"]:
                self.image_available_index.append(pos)

            if idx:
                self.idx_index[idx] = pos
            if question_id:
                question_id_index[question_id].append(pos)
            if object_count:
                object_count_index[object_count].append(pos)
            if scene_id:
                scene_index[scene_id].append(pos)

        self.question_id_index = dict(question_id_index)
        self.object_count_index = dict(object_count_index)
        self.scene_index = dict(scene_index)
        self.question_ids = sorted(self.question_id_index.keys())
        self.object_counts = sorted(self.object_count_index.keys(), key=_sort_object_count)
        self.scene_ids = sorted(self.scene_index.keys())
        self.scene_options = [
            {
                "scene_id": scene_id,
                "name": str(scene_meta.get(scene_id, {}).get("name") or ""),
                "batch": str(scene_meta.get(scene_id, {}).get("batch") or ""),
                "comment": str(scene_meta.get(scene_id, {}).get("comment") or ""),
            }
            for scene_id in self.scene_ids
        ]

        defaults = [scene_id for scene_id in self.scene_ids if scene_id not in self.excluded_scene_ids]
        self.default_selected_scene_ids = defaults if defaults else list(self.scene_ids)

        self._all_indices = list(range(len(self.items)))
        self.rows_with_available_images = len(self.image_available_index)
        self.rows_without_available_images = max(0, len(self.items) - self.rows_with_available_images)
        elapsed = time.time() - started
        print(
            "[load] questions=%d (with_images=%d, missing_images=%d), answers=%d, question_ids=%d, scenes=%d in %.2fs"
            % (
                len(self.items),
                self.rows_with_available_images,
                self.rows_without_available_images,
                len(answer_by_idx),
                len(self.question_ids),
                len(self.scene_ids),
                elapsed,
            )
        )

    @staticmethod
    def _load_answer_map(path: Path) -> dict[str, str]:
        if not path.is_file():
            return {}
        raw = _read_json(path)
        if not isinstance(raw, list):
            return {}
        answer_by_idx: dict[str, str] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            idx = str(entry.get("idx") or "")
            if idx:
                answer_by_idx[idx] = str(entry.get("answer") or "")
        return answer_by_idx

    @staticmethod
    def _load_scene_meta(path: Path) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        raw = _read_json(path)
        if not isinstance(raw, dict):
            return {}
        scene_meta: dict[str, dict[str, Any]] = {}
        for scene_id, data in raw.items():
            if isinstance(data, dict):
                scene_meta[str(scene_id)] = data
            else:
                scene_meta[str(scene_id)] = {}
        return scene_meta

    @staticmethod
    def _load_excluded_scene_ids(path: Path) -> set[str]:
        if not path.is_file():
            return set()
        excluded: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                scene_id = line.strip()
                if not scene_id or scene_id.startswith("#"):
                    continue
                excluded.add(scene_id)
        return excluded

    def _put_cache(self, key: tuple[str, str, str, tuple[str, ...], bool], indices: list[int]) -> None:
        with self._cache_lock:
            self._filter_cache[key] = indices
            self._filter_cache.move_to_end(key)
            while len(self._filter_cache) > self.filter_cache_size:
                self._filter_cache.popitem(last=False)

    def _get_cache(self, key: tuple[str, str, str, tuple[str, ...], bool]) -> list[int] | None:
        with self._cache_lock:
            cached = self._filter_cache.get(key)
            if cached is None:
                return None
            self._filter_cache.move_to_end(key)
            return cached

    def _compute_indices(
        self,
        *,
        idx: str,
        question_id: str,
        object_count: str,
        scene_ids: tuple[str, ...],
        include_missing_images: bool,
    ) -> list[int]:
        key = (idx, question_id, object_count, scene_ids, include_missing_images)
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        if idx:
            pos = self.idx_index.get(idx)
            if pos is None:
                result = []
            else:
                row = self.items[pos]
                ok = True
                if question_id and row["question_id"] != question_id:
                    ok = False
                if object_count and row["object_count"] != object_count:
                    ok = False
                if scene_ids and row["scene"] not in scene_ids:
                    ok = False
                if not include_missing_images and not row.get("has_available_images", False):
                    ok = False
                result = [pos] if ok else []
            self._put_cache(key, result)
            return result

        candidates: list[list[int]] = []

        if question_id:
            candidates.append(self.question_id_index.get(question_id, []))
        if object_count:
            candidates.append(self.object_count_index.get(object_count, []))
        if scene_ids:
            scene_union: set[int] = set()
            for scene_id in scene_ids:
                scene_union.update(self.scene_index.get(scene_id, []))
            candidates.append(sorted(scene_union))
        if not include_missing_images:
            candidates.append(self.image_available_index)

        if not candidates:
            result = self._all_indices
            self._put_cache(key, result)
            return result

        candidates.sort(key=len)
        result = list(candidates[0])
        for values in candidates[1:]:
            allowed = set(values)
            result = [pos for pos in result if pos in allowed]
            if not result:
                break

        self._put_cache(key, result)
        return result

    def get_metadata(self) -> dict[str, Any]:
        return {
            "question_file": str(self.question_file),
            "answer_file": str(self.answer_file),
            "total": len(self.items),
            "rows_with_available_images": self.rows_with_available_images,
            "rows_without_available_images": self.rows_without_available_images,
            "question_ids": self.question_ids,
            "object_counts": self.object_counts,
            "scene_options": self.scene_options,
            "excluded_scene_ids": sorted(self.excluded_scene_ids),
            "default_selected_scene_ids": self.default_selected_scene_ids,
        }

    def get_questions_page(
        self,
        *,
        page: int,
        page_size: int,
        idx: str,
        question_id: str,
        object_count: str,
        scene_ids: tuple[str, ...],
        include_missing_images: bool,
    ) -> dict[str, Any]:
        filtered = self._compute_indices(
            idx=idx,
            question_id=question_id,
            object_count=object_count,
            scene_ids=scene_ids,
            include_missing_images=include_missing_images,
        )

        total = len(filtered)
        page_count = max(1, math.ceil(total / page_size)) if total > 0 else 1
        page = max(1, min(page, page_count))

        start = (page - 1) * page_size
        end = start + page_size
        page_indices = filtered[start:end]

        items = [self.items[pos] for pos in page_indices]
        return {
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "total": total,
            "filters": {
                "idx": idx,
                "question_id": question_id,
                "object_count": object_count,
                "scene_ids": list(scene_ids),
                "include_missing_images": include_missing_images,
            },
            "items": items,
        }

    def build_question_bundle_zip(self, idx: str) -> tuple[bytes, str]:
        idx = str(idx or "").strip()
        if not idx:
            raise ValueError("Missing idx.")

        pos = self.idx_index.get(idx)
        if pos is None:
            raise KeyError(f"Unknown idx: {idx}")

        item = self.items[pos]
        files: list[str] = list(item.get("file_name_raw") or item.get("file_name") or [])
        folder_name = f"folder_{_safe_name(idx, 'question')}"

        image_entries: list[dict[str, str]] = []
        for i, raw_path in enumerate(files):
            source = Path(str(raw_path))
            ext = source.suffix or ".png"
            stem = _safe_name(source.stem, f"image_{i:02d}")
            zip_image_name = f"{i:02d}_{stem}{ext}"
            image_entries.append(
                {
                    "source_path": str(source),
                    "zip_path": f"{folder_name}/images/{zip_image_name}",
                    "relative_path": f"./images/{zip_image_name}",
                }
            )

        missing_images: list[str] = []
        for image_entry in image_entries:
            if not Path(image_entry["source_path"]).is_file():
                missing_images.append(image_entry["source_path"])

        question_payload = {
            "idx": item.get("idx", ""),
            "question_id": item.get("question_id", ""),
            "question": item.get("question", ""),
            "correct_answer": item.get("correct_answer", ""),
            "scene": item.get("scene", ""),
            "object_count": item.get("object_count", ""),
            "item_type": item.get("item_type", ""),
            "image_count": item.get("image_count", 0),
            "images": [entry["relative_path"] for entry in image_entries],
            "original_file_name": files,
            "missing_images": missing_images,
        }

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                f"{folder_name}/question.json",
                json.dumps(question_payload, ensure_ascii=False, indent=2),
            )
            for entry in image_entries:
                source = Path(entry["source_path"])
                if not source.is_file():
                    continue
                archive.write(source, arcname=entry["zip_path"])

        return buffer.getvalue(), f"{folder_name}.zip"


def _env_path(name: str, default_value: str) -> Path:
    return Path(os.getenv(name, default_value)).resolve()


QUESTION_FILE = _env_path("VQA_QUESTION_FILE", DEFAULT_QUESTION_FILE)
ANSWER_FILE = _env_path("VQA_ANSWER_FILE", DEFAULT_ANSWER_FILE)
SCENES_FILE = _env_path("VQA_SCENES_FILE", DEFAULT_SCENES_FILE)
EXCLUDE_FILE = _env_path("VQA_EXCLUDE_FILE", DEFAULT_EXCLUDE_FILE)
MAX_PAGE_SIZE = int(os.getenv("VQA_MAX_PAGE_SIZE", "200"))
FILTER_CACHE_SIZE = int(os.getenv("VQA_FILTER_CACHE_SIZE", "64"))
IMAGE_PREVIEW_MAX_WIDTH = int(os.getenv("VQA_IMAGE_PREVIEW_MAX_WIDTH", "1280"))
IMAGE_PREVIEW_QUALITY = int(os.getenv("VQA_IMAGE_PREVIEW_QUALITY", "70"))
IMAGE_PREVIEW_FORMAT = str(os.getenv("VQA_IMAGE_PREVIEW_FORMAT", "webp")).lower()
IMAGE_PREVIEW_CACHE_SIZE = int(os.getenv("VQA_IMAGE_PREVIEW_CACHE_SIZE", "256"))
SLOW_REQUEST_LOG_MS = int(os.getenv("VQA_SLOW_REQUEST_LOG_MS", "1200"))
REQUIRE_EXISTING_IMAGES = str(os.getenv("VQA_REQUIRE_EXISTING_IMAGES", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
}

app = FastAPI(title="Tiny VQA Visualization API")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def _log_slow_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if elapsed_ms >= max(1, SLOW_REQUEST_LOG_MS):
        query = request.url.query
        if len(query) > 240:
            query = query[:240] + "..."
        suffix = f"?{query}" if query else ""
        print(
            "[slow] %.1fms %s %s%s status=%s"
            % (elapsed_ms, request.method, request.url.path, suffix, response.status_code)
        )
    return response

_STORE: DatasetStore | None = None
_STORE_LOCK = Lock()
_IMAGE_CACHE_LOCK = Lock()
_IMAGE_CACHE: OrderedDict[tuple[str, int, int, int, int, str], tuple[bytes, str]] = OrderedDict()


def _image_cache_get(key: tuple[str, int, int, int, int, str]) -> tuple[bytes, str] | None:
    with _IMAGE_CACHE_LOCK:
        cached = _IMAGE_CACHE.get(key)
        if cached is None:
            return None
        _IMAGE_CACHE.move_to_end(key)
        return cached


def _image_cache_put(key: tuple[str, int, int, int, int, str], value: tuple[bytes, str]) -> None:
    with _IMAGE_CACHE_LOCK:
        _IMAGE_CACHE[key] = value
        _IMAGE_CACHE.move_to_end(key)
        while len(_IMAGE_CACHE) > max(1, IMAGE_PREVIEW_CACHE_SIZE):
            _IMAGE_CACHE.popitem(last=False)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _prepare_compressed_image(
    image_path: Path,
    *,
    width: int,
    quality: int,
    fmt: str,
) -> tuple[bytes, str] | None:
    if not PIL_AVAILABLE or fmt == "orig":
        return None

    if fmt not in ALLOWED_PREVIEW_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported fmt '{fmt}'. Use webp, jpeg, or orig.")

    file_stat = image_path.stat()
    cache_key = (
        str(image_path),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_size),
        width,
        quality,
        fmt,
    )
    cached = _image_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        with Image.open(image_path) as source_image:  # type: ignore[union-attr]
            image = ImageOps.exif_transpose(source_image)  # type: ignore[union-attr]

            if width > 0 and image.width > width:
                target_height = max(1, int(image.height * (width / float(image.width))))
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                image = image.resize((width, target_height), resample)

            buffer = BytesIO()
            if fmt == "jpeg":
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )
                payload = (buffer.getvalue(), "image/jpeg")
            else:
                if image.mode == "P":
                    image = image.convert("RGBA")
                image.save(
                    buffer,
                    format="WEBP",
                    quality=quality,
                    method=6,
                )
                payload = (buffer.getvalue(), "image/webp")
    except UnidentifiedImageError:
        return None

    _image_cache_put(cache_key, payload)
    return payload


def get_store() -> DatasetStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = DatasetStore(
                    question_file=QUESTION_FILE,
                    answer_file=ANSWER_FILE,
                    scenes_file=SCENES_FILE,
                    exclude_file=EXCLUDE_FILE,
                    filter_cache_size=FILTER_CACHE_SIZE,
                )
    return _STORE


@app.on_event("startup")
def _warm_start() -> None:
    get_store()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    store = get_store()
    return {
        "status": "ok",
        "loaded_rows": len(store.items),
        "question_file": str(store.question_file),
    }


@app.get("/api/metadata")
async def metadata() -> dict[str, Any]:
    return get_store().get_metadata()


@app.get("/api/questions")
async def questions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1),
    idx: str = Query(default=""),
    question_id: str = Query(default=""),
    object_count: str = Query(default=""),
    scene_id: list[str] | None = Query(default=None),
    include_missing_images: bool = Query(default=False),
) -> dict[str, Any]:
    store = get_store()
    page_size = min(page_size, MAX_PAGE_SIZE)

    idx = idx.strip()
    question_id = question_id.strip()
    object_count = object_count.strip()
    scene_ids = _parse_scene_ids(scene_id)

    return store.get_questions_page(
        page=page,
        page_size=page_size,
        idx=idx,
        question_id=question_id,
        object_count=object_count,
        scene_ids=scene_ids,
        include_missing_images=include_missing_images or (not REQUIRE_EXISTING_IMAGES),
    )


@app.get("/api/image")
def image(
    path: str = Query(..., min_length=1),
    width: int = Query(default=0, ge=0, le=4096),
    quality: int = Query(default=0, ge=0, le=100),
    fmt: str = Query(default=""),
) -> Response:
    image_path = Path(path).expanduser()
    target_width = _clamp(width if width > 0 else IMAGE_PREVIEW_MAX_WIDTH, 64, 4096)
    target_height = _clamp(int(target_width * 0.66), 48, 3072)
    if not image_path.is_file() or not os.access(image_path, os.R_OK):
        return Response(
            content=_placeholder_svg(target_width, target_height, "image not found"),
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=300",
                "X-Image-Placeholder": "1",
            },
        )

    chosen_fmt = (fmt.strip().lower() or IMAGE_PREVIEW_FORMAT or "orig")
    if chosen_fmt not in ALLOWED_PREVIEW_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported fmt '{chosen_fmt}'. Use webp, jpeg, or orig.")

    target_quality = _clamp(quality if quality > 0 else IMAGE_PREVIEW_QUALITY, 20, 95)

    try:
        compressed = _prepare_compressed_image(
            image_path,
            width=target_width,
            quality=target_quality,
            fmt=chosen_fmt,
        )
    except Exception:
        compressed = None
    if compressed is not None:
        payload, media_type = compressed
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=86400",
                "X-Image-Preview": "compressed",
            },
        )

    media_type, _ = mimetypes.guess_type(str(image_path))
    try:
        return FileResponse(
            image_path,
            media_type=media_type or "application/octet-stream",
            headers={
                "Cache-Control": "public, max-age=86400",
                "X-Image-Preview": "original",
            },
        )
    except Exception:
        return Response(
            content=_placeholder_svg(target_width, target_height, "image unavailable"),
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=300",
                "X-Image-Placeholder": "1",
            },
        )


@app.get("/api/download")
def download(idx: str = Query(..., min_length=1)) -> Response:
    store = get_store()
    try:
        payload, file_name = store.build_question_bundle_zip(idx)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except KeyError as err:
        raise HTTPException(status_code=404, detail=err.args[0] if err.args else str(err)) from err

    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )
