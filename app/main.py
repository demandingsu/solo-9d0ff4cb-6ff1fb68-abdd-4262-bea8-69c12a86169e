"""FastAPI service for pairing tunnel radar passages into travel-time samples."""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import storage
from .inputs import MAX_UPLOAD_BYTES, InputError, parse_csv_file
from .pairing import (
    ENTRY,
    EXIT,
    PairingError,
    PendingObservation,
    Trip,
    pair_observations,
)
from .verification import verify_snapshot

logger = logging.getLogger("tunnel_trips")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Response models (these drive the OpenAPI page at /docs)
# ---------------------------------------------------------------------------


class SourceFileOut(BaseModel):
    filename: str = Field(description="上传时的文件名")
    sha256: str = Field(description="文件内容 SHA-256")
    size_bytes: int = Field(description="文件字节数")


class SourcesOut(BaseModel):
    entry: SourceFileOut
    exit: SourceFileOut


class SummaryOut(BaseModel):
    max_trips: int = Field(description="最优配对可形成的最大行程数")
    confirmed_trips: int = Field(description="在所有最优配对中都出现的行程数（可交付）")
    pending_observations: int = Field(description="待核观测数")
    min_total_deviation_seconds: int = Field(
        description="最少行程数下，通行时长偏离 5 分钟的绝对值总和（秒）"
    )
    optimal_pairings: int = Field(description="最优配对（同样行程数、同样代价）的总数")


class TripOut(BaseModel):
    entry_id: str = Field(description="入口观测号")
    exit_id: str = Field(description="出口观测号")
    entry_time: str = Field(description="入口时间（UTC，整分钟）")
    exit_time: str = Field(description="出口时间（UTC，整分钟）")
    duration_seconds: int = Field(description="通行时长（秒），3~8 分钟")
    deviation_seconds: int = Field(description="与 5 分钟之差的绝对值（秒）")


class PendingOut(BaseModel):
    observation_id: str = Field(description="观测号")
    radar: str = Field(description="雷达：entry 或 exit")
    time: str = Field(description="观测时间（UTC，整分钟）")
    reason: str = Field(
        description=(
            "待核原因：ambiguous_pairing（出现在多个等价最优配对中，"
            "无法唯一确定对象）或 no_optimal_pair（不属于任何最优配对）"
        )
    )


class RoundOut(BaseModel):
    version: int
    created_at: str
    sources: SourcesOut
    summary: SummaryOut
    confirmed_trips: List[TripOut]
    pending_observations: List[PendingOut]


class UploadOut(RoundOut):
    replaced_version: Optional[int] = Field(
        description="被整版替换掉的上一版版本号；首次上传为 null"
    )


class DigestCheckOut(BaseModel):
    radar: str = Field(description="雷达：entry 或 exit")
    filename: str = Field(description="入库时记录的原始文件名")
    recorded_sha256: str = Field(description="入库时记录的 SHA-256")
    actual_sha256: str = Field(description="对库内原始文件 BLOB 重新计算的 SHA-256")
    recorded_size_bytes: int = Field(description="入库时记录的字节数")
    actual_size_bytes: int = Field(description="库内原始文件 BLOB 的实际字节数")
    sha256_ok: bool = Field(description="重新计算的 SHA-256 是否与记录一致")
    size_ok: bool = Field(description="实际字节数是否与记录一致")


class DifferenceOut(BaseModel):
    scope: str = Field(
        description="差异范围：observation（观测）、trip（行程）或 summary（统计）"
    )
    field: str = Field(description="差异字段名")
    expected: Optional[object] = Field(
        description="按原始文件重新解析/配对应得到的值"
    )
    actual: Optional[object] = Field(description="入库的实际值")
    radar: Optional[str] = Field(None, description="适用时：entry 或 exit")
    observation_id: Optional[str] = Field(None, description="适用时：观测号")
    entry_id: Optional[str] = Field(None, description="适用时：行程入口观测号")
    exit_id: Optional[str] = Field(None, description="适用时：行程出口观测号")
    detail: Optional[str] = Field(None, description="差异的人类可读说明")


class VerificationOut(BaseModel):
    version: int = Field(description="本次核验所读取的已提交轮次版本号")
    created_at: str = Field(description="该轮次入库时间（UTC）")
    status: str = Field(
        description=(
            "核验结论：verified（已核实、可交付）、"
            "mismatch（有差异、不可交付）、"
            "unverifiable（摘要不符或无法解析，无法核实）"
        )
    )
    deliverable: bool = Field(
        description="是否可交付：仅 status=verified 时为 true"
    )
    status_reason: str = Field(description="结论的中文说明")
    digest_checks: List[DigestCheckOut] = Field(
        description="两份原始雷达 CSV 的摘要核对结果"
    )
    parse_errors: List[str] = Field(
        description="重新解析原始文件时的错误；为空表示两份文件均可解析"
    )
    differences: List[DifferenceOut] = Field(
        description=(
            "重新解析并按原规则配对后，与入库观测、已确认行程、待核原因及"
            "统计的逐项差异；无法核实时为空（不能据库内行程判定可信）"
        )
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    storage.init_db()
    logger.info("database ready at %s", storage.database_path())
    yield


app = FastAPI(
    title="隧道夜间试跑通行时长样本 API",
    description=(
        "上传同一轮入口/出口雷达 CSV（各不超过 12 条，含观测号与 UTC 整分钟），"
        "按 3~8 分钟可行区间做一对一配对：先最大化行程数，再最小化"
        "|通行时长 - 5 分钟| 的总和。只有出现在**所有**最优配对中的行程才会交付，"
        "其余观测进入待核列表。\n\n"
        "每次成功上传整版替换旧结果；文件校验失败或写入失败时旧版本保持不变。"
        "读取使用单一 SQLite 快照，不会混合新旧版本。\n\n"
        "交付前可用 GET /api/rounds/current/verification 做只读核验："
        "在同一已提交轮次的快照内重算原始 CSV 摘要、重新解析并按原规则配对，"
        "对照入库观测、已确认行程、待核原因与统计。摘要不符或无法解析为"
        "unverifiable（无法核实），存在差异为 mismatch（不可交付），"
        "完全相符为 verified（可交付）。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz", summary="健康检查")
def healthz() -> dict:
    try:
        storage.load_current_round()
    except storage.StorageError as exc:  # pragma: no cover - environment issue
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "database": str(storage.database_path())}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _read_upload(upload: UploadFile, label: str) -> bytes:
    try:
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
    except Exception as exc:  # noqa: BLE001 - client-side read failure
        raise HTTPException(
            status_code=400, detail=f"{label} 文件读取失败: {exc}"
        ) from exc
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"{label} 文件超过 {MAX_UPLOAD_BYTES} 字节上限",
        )
    return data


def _trip_payload(trip: Trip) -> dict:
    return {
        "entry_id": trip.entry.observation_id,
        "exit_id": trip.exit.observation_id,
        "entry_time": storage.utc_minute_to_iso(trip.entry.utc_minute),
        "exit_time": storage.utc_minute_to_iso(trip.exit.utc_minute),
        "duration_seconds": trip.duration_seconds,
        "deviation_seconds": trip.deviation_seconds,
    }


def _pending_payload(pending: PendingObservation) -> dict:
    return {
        "observation_id": pending.observation.observation_id,
        "radar": pending.observation.radar,
        "time": storage.utc_minute_to_iso(pending.observation.utc_minute),
        "reason": pending.reason,
    }


def _summary_payload(
    *, max_trips: int, confirmed: int, pending: int, min_dev: int, optimal: int
) -> dict:
    return {
        "max_trips": max_trips,
        "confirmed_trips": confirmed,
        "pending_observations": pending,
        "min_total_deviation_seconds": min_dev,
        "optimal_pairings": optimal,
    }


def _parse_and_pair(
    entry_data: bytes, exit_data: bytes, entry_name: str, exit_name: str
):
    """CPU-bound parsing and exact pairing, run off the event loop."""

    parsed_entry = parse_csv_file(entry_data, ENTRY, entry_name)
    parsed_exit = parse_csv_file(exit_data, EXIT, exit_name)
    result = pair_observations(parsed_entry.observations, parsed_exit.observations)
    return parsed_entry, parsed_exit, result


@app.post(
    "/api/rounds",
    response_model=UploadOut,
    status_code=201,
    summary="上传一轮两份 CSV，整版替换当前结果",
    response_description="新版本及其可交付行程、待核观测",
)
async def upload_round(
    entry_file: UploadFile = File(
        ..., description="入口雷达 CSV（≤12 行，观测号 + UTC 整分钟）"
    ),
    exit_file: UploadFile = File(
        ..., description="出口雷达 CSV（≤12 行，观测号 + UTC 整分钟）"
    ),
) -> dict:
    entry_data = await _read_upload(entry_file, "入口")
    exit_data = await _read_upload(exit_file, "出口")

    entry_name = Path(entry_file.filename or "entry.csv").name
    exit_name = Path(exit_file.filename or "exit.csv").name
    await entry_file.close()
    await exit_file.close()

    try:
        parsed_entry, parsed_exit, result = await run_in_threadpool(
            _parse_and_pair, entry_data, exit_data, entry_name, exit_name
        )
    except InputError as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors}) from exc
    except PairingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        written = await run_in_threadpool(
            storage.replace_round,
            result=result,
            entry_observations=parsed_entry.observations,
            exit_observations=parsed_exit.observations,
            entry_name=entry_name,
            exit_name=exit_name,
            entry_sha256=_sha256(entry_data),
            exit_sha256=_sha256(exit_data),
            entry_bytes=entry_data,
            exit_bytes=exit_data,
        )
    except storage.StorageError as exc:
        logger.exception("replacing round failed; previous version kept")
        raise HTTPException(
            status_code=500,
            detail=f"写入失败，已保留上一版结果: {exc}",
        ) from exc

    trips = [_trip_payload(t) for t in result.confirmed]
    pending = [_pending_payload(p) for p in result.pending]
    logger.info(
        "round %s stored: %s confirmed trips, %s pending observations",
        written.version,
        len(trips),
        len(pending),
    )
    return {
        "version": written.version,
        "created_at": written.created_at,
        "sources": _sources_payload(
            entry_name=entry_name,
            exit_name=exit_name,
            entry_sha256=_sha256(entry_data),
            exit_sha256=_sha256(exit_data),
            entry_size=len(entry_data),
            exit_size=len(exit_data),
        ),
        "summary": _summary_payload(
            max_trips=result.max_trips,
            confirmed=len(trips),
            pending=len(pending),
            min_dev=result.min_total_deviation_seconds,
            optimal=result.optimal_matchings,
        ),
        "confirmed_trips": trips,
        "pending_observations": pending,
        "replaced_version": written.replaced_version,
    }


def _sources_payload(
    *,
    entry_name: str,
    exit_name: str,
    entry_sha256: str,
    exit_sha256: str,
    entry_size: int,
    exit_size: int,
) -> dict:
    return {
        "entry": {
            "filename": entry_name,
            "sha256": entry_sha256,
            "size_bytes": entry_size,
        },
        "exit": {
            "filename": exit_name,
            "sha256": exit_sha256,
            "size_bytes": exit_size,
        },
    }


def _stored_payload(stored: storage.StoredRound) -> dict:
    meta = stored.meta
    return {
        "version": meta.version,
        "created_at": meta.created_at,
        "sources": _sources_payload(
            entry_name=meta.entry_name,
            exit_name=meta.exit_name,
            entry_sha256=meta.entry_sha256,
            exit_sha256=meta.exit_sha256,
            entry_size=meta.entry_size,
            exit_size=meta.exit_size,
        ),
        "summary": _summary_payload(
            max_trips=meta.max_trips,
            confirmed=len(stored.trips),
            pending=len(stored.pending),
            min_dev=meta.min_deviation_seconds,
            optimal=meta.optimal_matchings,
        ),
        "confirmed_trips": stored.trips,
        "pending_observations": stored.pending,
    }


def _current_or_404() -> storage.StoredRound:
    try:
        stored = storage.load_current_round()
    except storage.StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if stored is None:
        raise HTTPException(status_code=404, detail="尚未上传任何一轮数据")
    return stored


@app.get(
    "/api/rounds/current",
    response_model=RoundOut,
    summary="查询当前版本的可交付行程与待核观测",
)
def get_current_round() -> dict:
    return _stored_payload(_current_or_404())


@app.get(
    "/api/trips",
    response_model=List[TripOut],
    summary="查询当前版本已确认（可交付）行程",
)
def get_trips() -> List[dict]:
    return _current_or_404().trips


@app.get(
    "/api/pending",
    response_model=List[PendingOut],
    summary="查询当前版本待核观测",
)
def get_pending() -> List[dict]:
    return _current_or_404().pending


def _verification_payload(report) -> dict:
    return {
        "version": report.version,
        "created_at": report.created_at,
        "status": report.status,
        "deliverable": report.deliverable,
        "status_reason": report.status_reason,
        "digest_checks": [
            {
                "radar": check.radar,
                "filename": check.filename,
                "recorded_sha256": check.recorded_sha256,
                "actual_sha256": check.actual_sha256,
                "recorded_size_bytes": check.recorded_size_bytes,
                "actual_size_bytes": check.actual_size_bytes,
                "sha256_ok": check.sha256_ok,
                "size_ok": check.size_ok,
            }
            for check in report.digest_checks
        ],
        "parse_errors": list(report.parse_errors),
        "differences": [
            {
                "scope": diff.scope,
                "field": diff.field,
                "expected": diff.expected,
                "actual": diff.actual,
                "radar": diff.radar,
                "observation_id": diff.observation_id,
                "entry_id": diff.entry_id,
                "exit_id": diff.exit_id,
                "detail": diff.detail,
            }
            for diff in report.differences
        ],
    }


@app.get(
    "/api/rounds/current/verification",
    response_model=VerificationOut,
    summary="只读核验：重算原始文件摘要并重解析配对，对照入库结果",
    response_description=(
        "所读轮次版本、摘要核对、解析错误与逐项差异；"
        "verified 可交付，mismatch 不可交付，unverifiable 无法核实"
    ),
)
def verify_current_round() -> dict:
    """Verify the committed round this request reads.

    All data for one verification comes from a single SQLite read snapshot,
    so an upload replacing the round concurrently can never produce a report
    mixing two rounds: the version field names the exact round verified.
    The endpoint performs no writes.
    """

    try:
        snapshot = storage.load_verification_snapshot()
    except storage.StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail="尚未上传任何一轮数据")
    report = verify_snapshot(snapshot)
    logger.info(
        "verification of round %s: status=%s, digest_failed=%s, "
        "parse_errors=%d, differences=%d",
        report.version,
        report.status,
        [c.radar for c in report.digest_checks if not c.ok],
        len(report.parse_errors),
        len(report.differences),
    )
    return _verification_payload(report)
