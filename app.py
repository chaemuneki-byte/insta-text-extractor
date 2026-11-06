"""인스타그램 릴스·캐러셀(카드뉴스) → 다운로드 → 음성/화면 글자/이미지 글자 추출 → 한국어 번역 웹앱."""

import base64
import difflib
import io
import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

DOWNLOADS = BASE / "downloads"
DOWNLOADS.mkdir(exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")  # GPU용 (정확도 최우선)
WHISPER_MODEL_CPU = os.getenv("WHISPER_MODEL_CPU", "small")  # GPU가 안 될 때 (large는 CPU에서 너무 느림)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto | cuda | cpu
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
COOKIES_FROM_BROWSER = os.getenv("COOKIES_FROM_BROWSER", "").strip()
SCREEN_OCR = os.getenv("SCREEN_OCR", "1") != "0"  # 영상 화면 속 글자(자막) 읽기

# instagram.com/p/ID, /reel/ID 외에 프로필에서 복사한 instagram.com/계정/p/ID 형태도 허용
INSTAGRAM_RE = re.compile(
    r"https?://(?:www\.|m\.)?instagram\.com/(?:[\w.]+/)?(reel|reels|p|tv)/([\w-]+)", re.I)

LANG_NAMES = {
    "ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어", "zh-CN": "중국어", "es": "스페인어",
    "fr": "프랑스어", "de": "독일어", "it": "이탈리아어", "pt": "포르투갈어", "ru": "러시아어",
    "vi": "베트남어", "th": "태국어", "id": "인도네시아어", "ar": "아랍어", "hi": "힌디어",
    "tr": "터키어", "nl": "네덜란드어", "pl": "폴란드어",
}

app = FastAPI(title="인스타 영상 텍스트 추출기")


@app.middleware("http")
async def no_cache_pages(request, call_next):
    """프로그램을 업데이트했을 때 브라우저가 예전 화면을 쓰지 않도록."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

jobs: dict[str, dict] = {}
job_order: list[str] = []
jobs_lock = threading.Lock()
job_queue: "queue.Queue[str]" = queue.Queue()


def has_claude() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def claude_text(content, max_tokens: int = 16000, effort: str = "medium") -> str:
    """Claude 호출 공통 함수. 안전 필터에 걸려 거절되면 서버가 자동으로 다른 모델로 이어서 처리(fallbacks)."""
    import anthropic

    client = anthropic.Anthropic()
    msg = client.beta.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": effort},
        messages=[{"role": "user", "content": content}],
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude가 이 요청을 처리하지 않았어요 (안전 정책)")
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _json_from(text: str, opener: str = "{"):
    closer = "}" if opener == "{" else "]"
    return json.loads(text[text.index(opener): text.rindex(closer) + 1])


def looks_korean(text: str) -> bool:
    letters = re.sub(r"[\W\d_]", "", text)
    return bool(letters) and len(re.findall(r"[가-힣]", letters)) / len(letters) > 0.3


# ----------------------------------------------------------------- ffmpeg ---

_ffmpeg_path: str | None = None


def ffmpeg_exe() -> str:
    """시스템 ffmpeg가 있으면 그걸, 없으면 pip(imageio-ffmpeg)에 들어있는 ffmpeg를 사용."""
    global _ffmpeg_path
    if _ffmpeg_path:
        return _ffmpeg_path
    import shutil

    found = shutil.which("ffmpeg")
    if not found:
        import imageio_ffmpeg

        # yt-dlp는 파일 이름이 정확히 ffmpeg.exe 여야 인식하므로 복사해서 사용
        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
        local = BASE / ".bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not local.exists():
            local.parent.mkdir(exist_ok=True)
            shutil.copy2(bundled, local)
        found = str(local)
    _ffmpeg_path = found
    return found


def load_audio(path: Path):
    """ffmpeg로 16kHz 모노 float32 배열 추출. 오디오가 없으면 None."""
    import numpy as np

    cmd = [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(path),
           "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or len(proc.stdout) < 16000 * 4 // 2:  # 0.5초 미만이면 소리 없음
        return None
    audio = np.frombuffer(proc.stdout, np.float32)
    if float(np.abs(audio).max(initial=0)) < 1e-4:  # 무음 트랙
        return None
    return audio


# ---------------------------------------------------------------- Whisper ---

_whisper = {"model": None, "device": None}
_whisper_lock = threading.Lock()
_cuda_dlls_done = False


def _register_cuda_dlls():
    """pip로 설치한 nvidia-cublas/cudnn DLL 경로를 Windows DLL 검색 경로에 추가."""
    global _cuda_dlls_done
    if os.name != "nt" or _cuda_dlls_done:
        return
    _cuda_dlls_done = True
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _load_whisper(device: str):
    from faster_whisper import WhisperModel

    if device == "cuda":
        return WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    return WhisperModel(WHISPER_MODEL_CPU, device="cpu", compute_type="int8")


def whisper_label() -> str:
    if _whisper["device"] == "cuda":
        return f"Whisper {WHISPER_MODEL} (GPU)"
    if _whisper["device"] == "cpu":
        return f"Whisper {WHISPER_MODEL_CPU} (CPU)"
    return ""


def transcribe(audio, on_status) -> tuple[str, list[dict]]:
    """(감지된 언어 코드, [{start, end, text}]) 반환. GPU 실패 시 CPU로 재시도."""
    with _whisper_lock:
        devices = ["cuda", "cpu"] if WHISPER_DEVICE == "auto" else [WHISPER_DEVICE]
        if _whisper["device"] in devices:  # 이미 성공한 장치를 우선
            devices.remove(_whisper["device"])
            devices.insert(0, _whisper["device"])

        last_err = None
        for device in devices:
            try:
                if _whisper["device"] != device:
                    on_status("음성 인식 모델 준비 중… (처음 한 번은 모델 파일을 내려받느라 "
                              "몇 분 걸릴 수 있어요. 창을 닫지 마세요)")
                    if device == "cuda":
                        _register_cuda_dlls()
                    _whisper["model"] = _load_whisper(device)
                    _whisper["device"] = device
                    on_status("음성을 텍스트로 변환 중…")
                # VAD(말소리 구간 감지)로 먼저 시도 → 아무것도 안 잡히면(노래 등) VAD 없이 재시도
                for vad in (True, False):
                    segments, info = _whisper["model"].transcribe(audio, beam_size=5, vad_filter=vad)
                    segs = [
                        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                        for s in segments
                        if s.text.strip() and s.no_speech_prob < 0.8
                    ]
                    if segs:
                        break
                return info.language, segs
            except Exception as e:  # noqa: BLE001
                last_err = e
                _whisper["model"] = None
                _whisper["device"] = None
                print(f"[whisper] {device} 실패: {e}")
        raise RuntimeError(f"음성 인식 실패: {last_err}")


# -------------------------------------------------------------------- OCR ---

_ocr = {"reader": None, "label": ""}
_ocr_lock = threading.Lock()


def get_ocr_reader(on_status):
    """글자 위치 찾기 = EasyOCR(GPU 가능), 글자 읽기 = RapidOCR PP-OCRv5 한국어 모델.
    (EasyOCR은 위치는 잘 찾지만 한글을 자주 틀리고, PP-OCRv5는 한글 인식이 정확함)"""
    if _ocr["reader"] is None:
        on_status("글자 인식 모델 준비 중… (처음 한 번은 모델을 내려받느라 1~2분 걸려요)")
        _register_cuda_dlls()
        import easyocr
        import torch
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        gpu = torch.cuda.is_available() and WHISPER_DEVICE != "cpu"
        _ocr["reader"] = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False)
        def rec(lang):
            return RapidOCR(params={
                "Global.log_level": "error", "Rec.lang_type": lang,
                "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            })
        _ocr["rec"] = rec(LangRec.KOREAN)
        _ocr["rec_en"] = rec(LangRec.EN)
        _ocr["label"] = f"EasyOCR+PP-OCRv5 ({'GPU' if gpu else 'CPU'})"
    return _ocr["reader"]


_OCR_FIXES = [(re.compile(r"\bAl\b"), "AI")]  # 대문자 I ↔ 소문자 l 혼동


def _respace(base: str, alt: str) -> str:
    """글자는 같은데 띄어쓰기만 빠진 경우(긴 영어 줄), alt의 띄어쓰기를 base에 옮겨 넣음."""
    if not alt or alt.count(" ") <= base.count(" "):
        return base
    nb, na = _norm(base), _norm(alt)
    if difflib.SequenceMatcher(None, nb, na).ratio() < 0.8:
        return base
    alt_breaks, k, gap = set(), 0, False
    for ch in alt:
        if re.match(r"[\W_]", ch):
            gap = gap or ch.isspace()
            continue
        if gap and k:
            alt_breaks.add(k)
        gap, k = False, k + 1
    # 글자가 일치하는 구간끼리 위치를 맞춰서, alt의 띄어쓰기 위치를 base 위치로 옮김
    breaks = set()
    for i, j, size in difflib.SequenceMatcher(None, nb, na, autojunk=False).get_matching_blocks():
        for kk in range(j, j + size):
            if kk in alt_breaks:
                breaks.add(i + (kk - j))
    out, k = [], 0
    for ch in base:
        if not re.match(r"[\W_]", ch):
            if k in breaks and out and re.match(r"\w", out[-1]):
                out.append(" ")
            k += 1
        out.append(ch)
    return "".join(out)


def ocr_lines(img, on_status, min_conf: float = 0.35) -> list[dict]:
    """이미지(경로 또는 numpy) → 읽는 순서대로 정렬된 줄 목록 [{text, conf, h, y}]."""
    import numpy as np
    from PIL import Image

    if not isinstance(img, np.ndarray):
        img = np.asarray(Image.open(img).convert("RGB"))
    H, W = img.shape[:2]
    boxes = []
    with _ocr_lock:
        reader = get_ocr_reader(on_status)
        raw = reader.readtext(img)
        for bbox, text, conf in raw:
            ys = [int(p[1]) for p in bbox]
            xs = [int(p[0]) for p in bbox]
            pad = int((max(ys) - min(ys)) * 0.15)
            crop = img[max(0, min(ys) - pad): min(H, max(ys) + pad),
                       max(0, min(xs) - pad): min(W, max(xs) + pad)]
            if crop.size:
                try:
                    r = _ocr["rec"](crop, use_det=False, use_cls=False, use_rec=True)
                    rt, rs = (r.txts or [""])[0], float((r.scores or [0])[0])
                    if rt.strip() and rs >= 0.5:
                        easy_text = text
                        text, conf = rt, max(conf, rs)
                        if not HANGUL_RE.search(text):  # 영어 줄: 띄어쓰기 복원
                            e = _ocr["rec_en"](crop, use_det=False, use_cls=False, use_rec=True)
                            text = _respace(_respace(text, easy_text), (e.txts or [""])[0])
                except Exception as e:  # noqa: BLE001
                    print(f"[ocr] PP-OCR 인식 실패: {e}")
            text = text.strip()
            for pat, rep in _OCR_FIXES:
                text = pat.sub(rep, text)
            if not text or conf < min_conf:
                continue
            boxes.append({"text": text, "conf": float(conf), "x": min(xs),
                          "y0": min(ys), "y1": max(ys), "h": max(ys) - min(ys)})
    # 세로 위치가 비슷한 조각들을 한 줄로 합치기
    boxes.sort(key=lambda b: (b["y0"] + b["y1"]) / 2)
    lines: list[list[dict]] = []
    for b in boxes:
        cy = (b["y0"] + b["y1"]) / 2
        if lines:
            last = lines[-1]
            ly = sum((x["y0"] + x["y1"]) / 2 for x in last) / len(last)
            lh = max(x["h"] for x in last)
            if abs(cy - ly) < 0.5 * max(lh, b["h"]):
                last.append(b)
                continue
        lines.append([b])
    out = []
    for parts in lines:
        parts.sort(key=lambda b: b["x"])
        out.append({
            "text": " ".join(p["text"] for p in parts),
            "conf": sum(p["conf"] for p in parts) / len(parts),
            "h": max(p["h"] for p in parts),
            "y": min(p["y0"] for p in parts),
        })
    return out


def _image_block(path: Path) -> dict:
    data = path.read_bytes()
    media = ("image/png" if data[:4] == b"\x89PNG" else
             "image/webp" if data[8:12] == b"WEBP" else "image/jpeg")
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.standard_b64encode(data).decode()}}


def ocr_image_claude(path: Path) -> str:
    return claude_text([
        _image_block(path),
        {"type": "text", "text": (
            "인스타그램 카드뉴스 이미지입니다. 이미지에 적힌 글자를 사람이 읽는 순서대로 "
            "원문 그대로 옮겨 적으세요. 제목·본문의 줄바꿈은 유지하고, 워터마크와 계정 아이디(@...)는 "
            "빼세요. 번역이나 설명은 하지 말고 옮겨 적은 글자만 출력하세요. 글자가 없으면 아무것도 쓰지 마세요."
        )},
    ], max_tokens=4000, effort="low")


def ocr_image(path: Path, on_status) -> tuple[str, str]:
    """(추출 텍스트, 사용 엔진)."""
    if has_claude():
        try:
            return ocr_image_claude(path), "Claude"
        except Exception as e:  # noqa: BLE001
            print(f"[ocr] Claude 실패, EasyOCR로 대체: {e}")
    lines = ocr_lines(path, on_status)
    lines = [l for l in lines if not re.fullmatch(r"@[\w.]+", l["text"])]
    return "\n".join(l["text"] for l in lines), _ocr["label"]


def _norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t).lower()


def extract_frames(video: Path, work: Path) -> tuple[list[Path], int]:
    """영상을 1초에 2장(GPU) / 1장(CPU)씩 이미지로 뽑음. 장면 감지와 화면 글자 읽기에 같이 사용."""
    import shutil

    fps = 2 if _gpu_available() else 1
    frames_dir = work / "_frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir()
    subprocess.run(
        [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps},scale='min(720,iw)':-2", "-q:v", "3", str(frames_dir / "%05d.jpg")],
        check=True,
    )
    return sorted(frames_dir.glob("*.jpg")), fps


def cleanup_frames(work: Path):
    import shutil

    shutil.rmtree(work / "_frames", ignore_errors=True)


MAX_SCENES = 16


def detect_scenes(files: list[Path], fps: int, work: Path, slide_index: int) -> list[dict]:
    """화면이 확 바뀌는 지점(컷)을 찾아 장면으로 나누고, 장면마다 대표 이미지를 저장."""
    import numpy as np
    from PIL import Image

    if not files:
        return []
    smalls = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((36, 64))
        smalls.append(np.asarray(img, dtype=np.int16))
    diffs = [0.0] + [float(np.abs(smalls[i] - smalls[i - 1]).mean()) for i in range(1, len(smalls))]
    # 움직임이 많은 영상은 평소 차이도 커서, 영상마다 기준을 다르게 (평소 변화량의 2배 이상 = 컷)
    thr = max(28.0, 2.0 * float(np.median(diffs[1:]))) if len(diffs) > 2 else 28.0
    min_len = max(1, fps)  # 장면은 최소 1초
    cuts = [0]
    for i in range(1, len(files)):
        if diffs[i] >= thr and i - cuts[-1] >= min_len:
            cuts.append(i)
    bounds = [[c, n] for c, n in zip(cuts, cuts[1:] + [len(files)])]
    # 컷 없이 길게 이어지는 장면(토킹헤드 등)은 약 8초 단위로 나눠서 타임라인을 보기 쉽게
    chunk = 8 * fps
    split = []
    for a, b in bounds:
        n = max(1, round((b - a) / chunk)) if b - a > 10 * fps else 1
        step = (b - a) / n
        split += [[a + round(k * step), a + round((k + 1) * step)] for k in range(n)]
    bounds = split
    # 장면이 너무 많으면 가장 짧은 장면부터 옆 장면에 합침
    while len(bounds) > MAX_SCENES:
        k = min(range(len(bounds)), key=lambda j: bounds[j][1] - bounds[j][0])
        if k == 0:
            bounds[1][0] = bounds[0][0]
        else:
            bounds[k - 1][1] = bounds[k][1]
        bounds.pop(k)

    for old in work.glob(f"scene_{slide_index:02d}_*.jpg"):
        old.unlink(missing_ok=True)
    scenes = []
    for n, (a, b) in enumerate(bounds, 1):
        mid = files[(a + b - 1) // 2]
        thumb = f"scene_{slide_index:02d}_{n:02d}.jpg"
        img = Image.open(mid).convert("RGB")
        img.thumbnail((360, 640))
        img.save(work / thumb, quality=85)
        scenes.append({"index": n, "start": round(a / fps, 2), "end": round(b / fps, 2), "thumb": thumb})
    return scenes


def ocr_frames(files: list[Path], fps: int, on_status) -> dict:
    """영상 화면 속 글자(자막)를 시간대별로 추출. {segments, pinned}."""
    import numpy as np
    from PIL import Image

    if not files:
        return {"segments": [], "pinned": []}

    get_ocr_reader(on_status)
    frame_h = Image.open(files[0]).height
    per_frame: list[list[dict]] = []
    prev_small, prev_lines = None, []
    for i, f in enumerate(files):
        if i % 10 == 0:
            on_status(f"화면 속 글자 읽는 중… ({i + 1}/{len(files)})")
        img = Image.open(f).convert("RGB")
        small = np.asarray(img.convert("L").resize((48, 85)), dtype=np.int16)
        if prev_small is not None and np.abs(small - prev_small).mean() < 1.5:
            lines = prev_lines  # 화면이 거의 안 바뀌었으면 재사용
        else:
            lines = ocr_lines(np.asarray(img), on_status, min_conf=0.45)
            # 숫자·시간만 있는 줄(재생 시간, 페이지 번호 등 UI 잡음)은 제외
            lines = [l for l in lines if re.search(r"[^\W\d_]", l["text"])]
        per_frame.append(lines)
        prev_small, prev_lines = small, lines

    # 계속 떠 있는 글자: 작으면(워터마크·아이디) 제거, 크면 '고정 문구'로 분리
    counts: dict[str, list[dict]] = {}
    for lines in per_frame:
        for key in {_norm(l["text"]): l for l in lines}.items():
            counts.setdefault(key[0], []).append(key[1])
    persistent = {k for k, v in counts.items() if k and len(v) >= max(3, 0.6 * len(per_frame))}
    pinned = []
    for k in persistent:
        sample = max(counts[k], key=lambda l: l["conf"])
        small_text = sample["h"] < 0.03 * frame_h or sample["text"].startswith("@")
        if not small_text:
            pinned.append(sample)
    # 고정 문구는 여러 줄이어도 한 문장인 경우가 많아서 하나로 묶음 (번역도 자연스러워짐)
    pinned_text = "\n".join(p["text"] for p in sorted(pinned, key=lambda l: l["y"]))
    pinned = [pinned_text] if pinned_text else []

    segments: list[dict] = []
    step = 1 / fps
    for i, lines in enumerate(per_frame):
        text = "\n".join(l["text"] for l in lines if _norm(l["text"]) not in persistent)
        t = round(i * step, 2)
        if not _norm(text):
            continue
        n = _norm(text)
        if segments and segments[-1]["end"] >= t - step * 1.5:
            last = segments[-1]
            ln = _norm(last["text"])
            same = difflib.SequenceMatcher(None, ln, n).ratio() >= 0.75
            grow = ln in n or n in ln  # 단어가 하나씩 나타나는 자막
            if same or grow:
                if len(n) > len(ln):
                    last["text"] = text
                last["end"] = round(t + step, 2)
                continue
        segments.append({"start": t, "end": round(t + step, 2), "text": text})
    # 한 프레임만 스친 짧은 글자는 잡음일 가능성이 커서 제외 (2fps 기준)
    if fps >= 2:
        segments = [s for s in segments if s["end"] - s["start"] > step + 0.01]
    return {"segments": segments, "pinned": pinned}


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and WHISPER_DEVICE != "cpu"
    except Exception:  # noqa: BLE001
        return False


def ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------ Translation ---

def translate_claude(texts: list[str]) -> list[str]:
    prompt = (
        "다음은 인스타그램 게시물(릴스 음성 받아쓰기, 화면 자막, 카드뉴스 문구, 캡션, 해시태그)에서 "
        "뽑은 문장 목록입니다.\n"
        "각 항목을 자연스러운 한국어로 번역하세요. 말투와 뉘앙스(구어체, 강조, 유머)를 살리고, "
        "고유명사·브랜드명은 원문을 유지하거나 통용되는 한국어 표기를 쓰세요. 줄바꿈은 유지하세요.\n"
        f"입력과 같은 개수({len(texts)}개)의 JSON 문자열 배열만 출력하고, 다른 말은 쓰지 마세요.\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    arr = _json_from(claude_text(prompt, effort="low"), "[")
    if not isinstance(arr, list) or len(arr) != len(texts):
        raise ValueError("번역 결과 개수가 원문과 다릅니다")
    return [str(x) for x in arr]


def _google(text: str) -> tuple[str, str]:
    """(번역, 감지 언어)."""
    import requests

    status = None
    for attempt in range(3):
        r = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "ko", "dt": "t", "q": text},
            timeout=15,
        )
        status = r.status_code
        if status == 200:
            data = r.json()
            return "".join(p[0] for p in data[0] if p and p[0]), (data[2] or "")
        time.sleep(2 ** (attempt + 1))  # 2, 4, 8초 기다렸다가 다시
    raise RuntimeError(f"Google 번역 실패 (HTTP {status})")


def detect_lang_google(text: str) -> str:
    try:
        return _google(text[:500])[1] or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _join_layout_lines(text: str) -> str:
    """카드·자막의 줄바꿈은 디자인 때문이라 문장이 끊겨 있음 → 한 문단으로 이어서 번역해야 자연스러움.
    빈 줄(문단 구분)과 문장부호로 끝나는 줄은 유지."""
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        buf = ""
        for l in lines:
            if buf and re.search(r"[.!?。…:)\]]$|[\U0001F300-\U0001FAFF]$", buf):
                buf += "\n" + l
            else:
                buf = f"{buf} {l}".strip()
        out.append(buf)
    return "\n\n".join(out)


def translate_google(texts: list[str]) -> list[str]:
    """짧은 문장들은 줄바꿈으로 묶어서 한 번에 요청 (요청 수를 크게 줄여서 차단을 피함)."""
    prepared = [_join_layout_lines(t) for t in texts]
    out = [""] * len(texts)
    batch: list[tuple[int, str]] = []

    def one(i, t):
        out[i] = _google(t)[0]
        time.sleep(0.3)

    def flush():
        if len(batch) > 1:
            res = _google("\n".join(t for _, t in batch))[0].split("\n")
            time.sleep(0.3)
            if len(res) == len(batch):
                for (i, _), t in zip(batch, res):
                    out[i] = t.strip()
                batch.clear()
                return
        for i, t in batch:  # 묶음 결과가 어긋나면 하나씩
            one(i, t)
        batch.clear()

    for i, t in enumerate(prepared):
        if not t.strip():
            continue
        if "\n" in t:  # 여러 문단짜리(캡션 등)는 따로
            one(i, t)
            continue
        if batch and sum(len(x) + 1 for _, x in batch) + len(t) > 1500:
            flush()
        batch.append((i, t))
    flush()
    return out


# 내 PC 번역 (Facebook M2M100, MIT 라이선스): Google 번역이 막혔을 때 영어 문장을 한국어로
_mt = {"model": None, "tok": None, "dev": None}
_mt_lock = threading.Lock()
LOCAL_MT_LABEL = "내 PC 번역 (M2M100)"


def _is_english(text: str) -> bool:
    letters = re.sub(r"[\W\d_]", "", text)
    return bool(letters) and all(ord(c) < 128 for c in letters)


def translate_local(texts: list[str], src: str = "en") -> list[str]:
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    with _mt_lock:
        if _mt["model"] is None:
            _register_cuda_dlls()
            gpu = _gpu_available()
            mid = "facebook/m2m100_418M"
            _mt["tok"] = M2M100Tokenizer.from_pretrained(mid)
            model = M2M100ForConditionalGeneration.from_pretrained(
                mid, dtype=torch.float16 if gpu else torch.float32)
            _mt["dev"] = "cuda" if gpu else "cpu"
            _mt["model"] = model.to(_mt["dev"]).eval()
        tok, model = _mt["tok"], _mt["model"]
        tok.src_lang = src
        out = []
        for k in range(0, len(texts), 16):
            chunk = texts[k:k + 16]
            batch = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(_mt["dev"])
            with torch.inference_mode():
                ids = model.generate(**batch, forced_bos_token_id=tok.get_lang_id("ko"),
                                     num_beams=4, max_new_tokens=256)
            out += tok.batch_decode(ids, skip_special_tokens=True)
        return out


_google_blocked_until = 0.0  # Google이 막히면 잠시 건너뛰고 바로 내 PC 번역 사용


def _detect_lang(text: str, hint: str | None) -> str:
    """M2M100에 넘길 원문 언어 코드. 감지 실패 시 hint(영상 음성 언어) → 영어."""
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        code = detect(text) if len(re.sub(r"\W", "", text)) >= 4 else ""
    except Exception:  # noqa: BLE001
        code = ""
    code = {"zh-cn": "zh", "zh-tw": "zh", "tl": "tl", "he": "he", "iw": "he"}.get(code, code)
    return code or hint or "en"


def translate_local_any(texts: list[str], hint: str | None = None) -> list[str]:
    """언어를 감지해서 언어별로 묶어 내 PC 번역. 지원하지 않는 언어는 원문 유지."""
    out = list(texts)
    groups: dict[str, list[int]] = {}
    for k, t in enumerate(texts):
        groups.setdefault(_detect_lang(t, hint), []).append(k)
    for lang, ks in groups.items():
        if lang == "ko":
            continue
        try:
            res = translate_local([_join_layout_lines(texts[k]) for k in ks], src=lang)
        except Exception as e:  # noqa: BLE001  (지원 안 하는 언어 등)
            print(f"[translate] 내 PC 번역 실패 ({lang}): {e}")
            continue
        for k, t in zip(ks, res):
            out[k] = t.replace("<unk>", "").strip()
    return out


def translate(texts: list[str], hint: str | None = None) -> tuple[list[str], str]:
    """한국어가 아닌 항목만 번역. (번역 목록, 사용한 엔진) 반환. hint = 영상 음성 언어 코드."""
    global _google_blocked_until
    idx = [i for i, t in enumerate(texts) if t.strip() and not looks_korean(t)]
    out = list(texts)
    if not idx:
        return out, ""
    todo = [texts[i] for i in idx]
    engine = "Google 번역"
    done = None
    if has_claude():
        try:
            done, engine = translate_claude(todo), "Claude"
        except Exception as e:  # noqa: BLE001
            print(f"[translate] Claude 실패, Google로 대체: {e}")
    if done is None and time.time() >= _google_blocked_until:
        try:
            done = translate_google(todo)
        except Exception as e:  # noqa: BLE001
            print(f"[translate] Google 번역 실패 → 10분간 내 PC 번역 사용: {e}")
            _google_blocked_until = time.time() + 600
    if done is None:
        # Google이 막혔을 때: 내 PC 번역 (100개 언어, 언어 자동 감지)
        try:
            done = translate_local_any(todo, hint)
            engine = LOCAL_MT_LABEL
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print(f"[translate] 내 PC 번역도 실패: {e}")
            return out, "번역 실패 (원문 유지)"
        if sum(1 for a, b in zip(todo, done) if a == b) > len(todo) // 2:
            engine = f"{LOCAL_MT_LABEL} · 일부 원문 유지"
    for i, t in zip(idx, done):
        out[i] = t
    return out, engine


# --------------------------------------------------------------- Download ---

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")


def _save_image(url: str, dest_stem: Path) -> Path:
    import requests

    r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://www.instagram.com/"}, timeout=30)
    r.raise_for_status()
    data = r.content
    ext = ".png" if data[:4] == b"\x89PNG" else ".webp" if data[8:12] == b"WEBP" else ".jpg"
    path = dest_stem.with_suffix(ext)
    path.write_bytes(data)
    return path


def download(url: str, job_dir: Path) -> dict:
    """게시물의 모든 장(영상/이미지)을 내려받고 메타데이터 반환."""
    import yt_dlp

    opts = {
        "outtmpl": str(job_dir / "raw_%(playlist_index)s.%(ext)s"),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b",
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_exe(),
        "ignore_no_formats_error": True,  # 이미지 장은 영상 포맷이 없음
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    cookie_file = BASE / "cookies.txt"
    if cookie_file.exists():
        opts["cookiefile"] = str(cookie_file)
    elif COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_FROM_BROWSER,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        # 한 번만 요청해서 정보를 받고, 영상인 장만 골라서 내려받기
        # (한꺼번에 받으면 이미지 장에서 'No video formats' 오류로 멈춤)
        info = ydl.extract_info(url, download=False)
        entries = [e for e in (info.get("entries") or [info]) if e]
        for e in entries:
            if e.get("formats") and (e.get("url") or e.get("requested_formats")):
                ydl.process_info(e)

    slides = []
    for n, e in enumerate(entries, 1):
        idx = e.get("playlist_index")
        raw = sorted(job_dir.glob(f"raw_{idx if idx is not None else 'NA'}.*"))
        raw = [p for p in raw if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")]
        stem = job_dir / f"slide_{n:02d}"
        if raw:
            dest = stem.with_suffix(raw[0].suffix)
            raw[0].replace(dest)
            slides.append({"index": n, "type": "video", "file": dest.name,
                           "duration": e.get("duration")})
            continue
        thumbs = [t for t in (e.get("thumbnails") or []) if t.get("url")]
        if not thumbs:
            continue
        best = max(thumbs, key=lambda t: (t.get("width") or 0) * (t.get("height") or 0))
        path = _save_image(best["url"], stem)
        slides.append({"index": n, "type": "image", "file": path.name})
    for p in job_dir.glob("raw_*"):
        p.unlink(missing_ok=True)

    if not slides:
        raise RuntimeError("게시물에서 영상이나 이미지를 찾지 못했어요")
    kind = ("carousel" if len(slides) > 1 else
            "image" if slides[0]["type"] == "image" else "reel")
    return {
        "kind": kind,
        "uploader": info.get("channel") or info.get("uploader") or "",
        "uploader_name": info.get("uploader") or "",
        "caption": info.get("description") or "",
        "slides": slides,
    }


# ------------------------------------------------------- Caption/Hashtag ---

HASHTAG_RE = re.compile(r"#[^\s#.,!?;:()\[\]{}\"'“”‘’]+")
MENTION_RE = re.compile(r"@[\w.]+")
HANGUL_RE = re.compile(r"[가-힣]")


def split_caption(caption: str) -> dict:
    caption = (caption or "").strip()
    hashtags = list(dict.fromkeys(HASHTAG_RE.findall(caption)))  # 중복 제거, 순서 유지
    mentions = list(dict.fromkeys(m.rstrip(".") for m in MENTION_RE.findall(caption)))
    body = HASHTAG_RE.sub("", caption)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return {"full": caption, "body": body, "body_ko": body, "is_korean": looks_korean(body),
            "hashtags": [{"tag": h, "ko": ""} for h in hashtags], "mentions": mentions}


# -------------------------------------------------------------------- Job ---

def set_job(job_id: str, **kw):
    with jobs_lock:
        jobs[job_id].update(kw)


def process_slide(slide: dict, job_dir: Path, status) -> None:
    path = job_dir / slide["file"]
    if slide["type"] == "image":
        status("ocr", f"{slide['index']}장 이미지 글자 읽는 중…")
        if has_claude() or ocr_available():
            text, engine = ocr_image(path, lambda m: status("ocr", m))
        else:
            text, engine = "", ""
        slide["ocr"] = {"text": text, "ko": text, "engine": engine}
        return

    # 영상: 음성 + 화면 글자
    status("transcribing", f"{slide['index']}장 음성을 텍스트로 변환 중…")
    audio = load_audio(path)
    speech = {"language": "", "language_name": "", "segments": [], "has_audio": audio is not None}
    if audio is not None:
        lang, segs = transcribe(audio, lambda m: status("transcribing", m))
        full = " ".join(s["text"] for s in segs)
        if segs and lang == "ko" and not looks_korean(full):
            lang = detect_lang_google(full)  # 억양 때문에 언어 감지가 틀린 경우 보정
        elif segs and lang != "ko" and looks_korean(full):
            lang = "ko"
        for s in segs:
            s["ko"] = s["text"]
        speech.update(language=lang, language_name=LANG_NAMES.get(lang, lang), segments=segs,
                      engine=whisper_label())
    slide["speech"] = speech

    slide["screen"] = {"segments": [], "pinned": [], "pinned_ko": [], "engine": ""}
    status("ocr", f"{slide['index']}장 장면 나누는 중…")
    try:
        files, fps = extract_frames(path, job_dir)
        slide["scenes"] = detect_scenes(files, fps, job_dir, slide["index"])
        if SCREEN_OCR and ocr_available():
            status("ocr", f"{slide['index']}장 화면 속 글자 읽는 중…")
            screen = ocr_frames(files, fps, lambda m: status("ocr", m))
            for s in screen["segments"]:
                s["ko"] = s["text"]
            slide["screen"] = {**screen, "pinned_ko": list(screen["pinned"]), "engine": _ocr["label"]}
    finally:
        cleanup_frames(job_dir)


def translate_result(result: dict, status) -> str:
    """결과 안의 모든 외국어 텍스트를 한 번에 번역해서 'ko' 필드에 채움."""
    refs: list[tuple[dict | list, object]] = []
    texts: list[str] = []

    def add(container, key, text):
        if text and text.strip():
            refs.append((container, key))
            texts.append(text)

    for sl in result["slides"]:
        if sl["type"] == "image":
            add(sl["ocr"], "ko", sl["ocr"]["text"])
        else:
            for s in sl["speech"]["segments"]:
                add(s, "ko", s["text"])
            for s in sl["screen"]["segments"]:
                add(s, "ko", s["text"])
            for i, p in enumerate(sl["screen"]["pinned"]):
                add(sl["screen"]["pinned_ko"], i, p)
    cap = result["caption"]
    add(cap, "body_ko", cap["body"])
    for h in cap["hashtags"]:
        if not HANGUL_RE.search(h["tag"]):
            add(h, "ko", h["tag"][1:])

    if not any(not looks_korean(t) for t in texts):
        return ""
    status("translating", "한국어로 번역 중…")
    hint = next((sl["speech"].get("language") for sl in result["slides"]
                 if sl["type"] == "video" and sl["speech"].get("language")), None)
    out, engine = translate(texts, hint)
    for (container, key), t in zip(refs, out):
        container[key] = t
    return engine


# ------------------------------------------------- Scene description (무료) ---
# Microsoft Florence-2 (내 PC에서 실행, 무료): 장면 이미지를 보고 "무엇이 보이는지" 영어로 설명 → 한국어 번역

_cap = {"model": None, "proc": None, "dev": None, "dtype": None, "label": ""}
_cap_lock = threading.Lock()


def captioner_available() -> bool:
    try:
        from transformers import Florence2ForConditionalGeneration  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _load_captioner(on_status):
    if _cap["model"] is not None:
        return
    on_status("화면 설명 AI 준비 중… (처음 한 번은 모델을 내려받느라 2~3분 걸려요)")
    _register_cuda_dlls()
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    gpu = _gpu_available()
    mid = "florence-community/Florence-2-large" if gpu else "florence-community/Florence-2-base"
    dev, dtype = ("cuda", torch.float16) if gpu else ("cpu", torch.float32)
    _cap["model"] = Florence2ForConditionalGeneration.from_pretrained(mid, dtype=dtype).to(dev).eval()
    _cap["proc"] = AutoProcessor.from_pretrained(mid)
    _cap.update(dev=dev, dtype=dtype, label=f"Florence-2 {'large (GPU)' if gpu else 'base (CPU)'}")


def _florence(img, task: str) -> str:
    import torch

    inputs = _cap["proc"](text=task, images=img, return_tensors="pt").to(_cap["dev"], _cap["dtype"])
    with torch.inference_mode():
        ids = _cap["model"].generate(**inputs, max_new_tokens=96, num_beams=3, do_sample=False)
    raw = _cap["proc"].batch_decode(ids, skip_special_tokens=False)[0]
    return str(_cap["proc"].post_process_generation(raw, task=task, image_size=img.size)[task]).strip()


_TEXTY = re.compile(r"\b(text|reads|translat|written|font|caption|title|word|letter|logo|says)\w*", re.I)
_LEAD = re.compile(r"^(in )?(the|this) (image|picture|photo|frame|screenshot)( is| shows| depicts| features| appears to (be|show))?\s*,?\s*", re.I)


def _clean_detail(text: str, max_sentences: int = 2) -> str:
    """긴 설명에서 화면 글자를 추측하는 문장(자주 틀림)은 빼고 앞의 1~2문장만."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    keep = [s for s in sents if not _TEXTY.search(s)][:max_sentences]
    out = " ".join(keep)
    out = _LEAD.sub("", out)
    return out[:1].upper() + out[1:] if out else ""


TEXT_SCREEN = "글자(자막)가 크게 나오는 화면"


def _copies_text(desc: str, screen_text: str) -> bool:
    """설명 대신 화면 글자를 그대로 베껴 쓴 경우 (영어 설명에 다른 언어 글자가 섞이거나, OCR 글자와 거의 같음)."""
    if re.search(r"[^\x00-\x7F]", desc):
        return True
    a, b = _norm(desc), _norm(screen_text)
    return bool(a and b) and difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size > 0.6 * len(a)


def describe_scenes(paths: list[Path], on_status, screen_texts: list[str] | None = None) -> tuple[list[tuple[str, str]], str]:
    """장면 이미지마다 (짧은 설명, 자세한 설명) 영어 문장. (결과, 엔진 이름)."""
    from PIL import Image

    out = []
    with _cap_lock:
        _load_captioner(on_status)
        for i, p in enumerate(paths, 1):
            on_status(f"장면 화면 설명 만드는 중… ({i}/{len(paths)})")
            img = Image.open(p).convert("RGB")
            text = (screen_texts or [""] * len(paths))[i - 1]
            short = _florence(img, "<CAPTION>")
            detail = _clean_detail(_florence(img, "<DETAILED_CAPTION>"))
            if _copies_text(short, text):
                short = TEXT_SCREEN
            if not detail or _copies_text(detail, text):
                detail = short
            out.append((short, detail))
    return out, _cap["label"]


def _free_title(sl: dict, result: dict) -> tuple[str, str]:
    """무료 모드 제목 추정: 화면에 계속 떠 있는 문구 → 첫 화면 자막(후킹) → 캡션 첫 줄 → 첫 음성 문장."""
    sc = sl["screen"]
    if sc["pinned"]:
        ko = sc["pinned_ko"][0] if sc.get("pinned_ko") else sc["pinned"][0]
        return ko.replace("\n", " "), "화면에 계속 떠 있는 제목 문구"
    first = [s for s in sc["segments"] if s["start"] < 6]
    if first:
        return first[0]["ko"].replace("\n", " "), "영상 첫 화면 자막(후킹 문구)"
    cap = result["caption"]
    line = (cap.get("body_ko") or cap.get("body") or "").strip().split("\n")[0]
    if line:
        return line[:120], "게시물 캡션 첫 줄"
    sp = sl["speech"]["segments"]
    if sp:
        return sp[0]["ko"], "영상 첫 문장"
    return "", ""


# --------------------------------------------------------------- Analysis ---
# 릴스(영상 1개) 분석 리포트: 1) 영상의 주제  2) 화면 구성 및 타임라인  3) 스크립트
# - 무료 모드: 장면 나누기 + 장면별 자막/음성 + 스크립트까지 자동 정리
# - Claude 모드: 장면 이미지를 보고 주제·화면 설명까지 작성

def _mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def analysis_target(result: dict) -> dict | None:
    """분석 대상 영상 (카드뉴스·캐러셀은 제외, 릴스·단일 영상만)."""
    vids = [s for s in result["slides"] if s["type"] == "video"]
    if result["meta"].get("kind") == "reel" and len(vids) == 1:
        return vids[0]
    return None


def _uniq_join(items: list[str], sep: str) -> str:
    return sep.join(dict.fromkeys(i.replace("\n", " ").strip() for i in items if i.strip()))


def analyze_with_claude(result: dict, sl: dict, timeline: list[dict], job_dir: Path) -> dict:
    sp = sl["speech"]
    lines = [f"[계정] @{result['meta'].get('uploader', '')}"]
    if result["caption"].get("full"):
        lines.append(f"[게시물 캡션]\n{result['caption']['full'][:1500]}")
    if sp["segments"]:
        lines.append(f"[음성 받아쓰기] (원어: {sp.get('language_name') or '알 수 없음'})")
        for s in sp["segments"]:
            ko = f"  → {s['ko']}" if s["ko"] != s["text"] else ""
            lines.append(f"{_mmss(s['start'])} {s['text']}{ko}")
    else:
        lines.append("[음성 받아쓰기] 없음 (말소리가 없는 영상)")
    sc = sl["screen"]
    if sc["pinned"]:
        lines.append("[계속 떠 있는 화면 문구] " + " / ".join(p.replace("\n", " ") for p in sc["pinned"]))
    if sc["segments"]:
        lines.append("[화면 속 자막 (OCR, 오탈자가 있을 수 있음)]")
        for s in sc["segments"]:
            lines.append(f"{_mmss(s['start'])}~{_mmss(s['end'])} {s['text'].replace(chr(10), ' / ')}")
    lines.append(f"[장면 목록] 컷 전환 기준으로 {len(timeline)}개 장면으로 나눴고, 아래 이미지가 장면별 대표 화면입니다.")

    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for t in timeline:
        content.append({"type": "text", "text": f"[장면 {t['scene']}] {_mmss(t['start'])} ~ {_mmss(t['end'])}"})
        content.append(_image_block(job_dir / t["thumb"]))
    content.append({"type": "text", "text": (
        "당신은 인스타그램 릴스를 벤치마킹하는 콘텐츠 분석가입니다. 위 영상 정보를 바탕으로 아래 JSON 하나만 출력하세요 "
        "(코드블록이나 다른 설명 없이).\n"
        "{\n"
        '  "topic": {\n'
        '    "title": "영상 주제를 한 문장으로. 예) 출근하지 않고 시간을 버는 삶, 평범한 사람들이 콘텐츠로 수익을 만드는 \'크리에이터 클럽\' 소개",\n'
        '    "summary": "영상 전체 내용을 2~3문장으로 요약",\n'
        '    "points": ["핵심 포인트 3~5개. 각각 \'키워드: 설명\' 형식"]\n'
        "  },\n"
        '  "timeline": [{"scene": 1, "visual": "이 장면의 화면 구성 1~2문장 (인물·장소·행동·카메라 구도·그래픽/텍스트 연출·수익 인증 같은 화면 요소)", '
        '"caption": "이 장면에 나온 화면 자막을 한국어로 (없으면 빈 문자열)"}],\n'
        '  "script": [{"start": 0.0, "orig": "원어 문장", "ko": "자연스러운 한국어 문장"}]\n'
        "}\n"
        "규칙:\n"
        "- 모든 설명은 한국어로 씁니다.\n"
        "- timeline에는 장면 목록의 모든 장면을 순서대로 넣습니다.\n"
        "- script: 음성이 있으면 음성을 바탕으로 나레이션을 문장 단위로 다듬어 정리하고, 받아쓰기 오류는 문맥에 맞게 바로잡습니다. "
        "음성이 없으면 화면 자막을 순서대로 이어서 스크립트로 정리합니다. 원어가 한국어면 orig와 ko를 같게 씁니다. start는 초 단위 숫자입니다.\n"
        "- 영상과 자료에 없는 내용은 지어내지 않습니다."
    )})
    data = _json_from(claude_text(content, max_tokens=16000, effort="medium"))
    return data


def build_analysis(result: dict, job_dir: Path, status=None) -> dict | None:
    status = status or (lambda st, m: None)
    sl = analysis_target(result)
    if not sl:
        return None
    speech, screen = sl["speech"]["segments"], sl["screen"]["segments"]
    scenes = sl.get("scenes") or []

    def assign(segs):
        """각 문장을 가장 많이 겹치는 장면 하나에만 넣음 (여러 장면에 중복으로 나오지 않게)."""
        out = {sc["index"]: [] for sc in scenes}
        for s in segs:
            best = max(scenes, key=lambda sc: min(s["end"], sc["end"]) - max(s["start"], sc["start"]), default=None)
            if best is not None:
                out[best["index"]].append(s)
        return out

    caps_by, sps_by = assign(screen), assign(speech)
    timeline = []
    for sc in scenes:
        caps, sps = caps_by[sc["index"]], sps_by[sc["index"]]
        timeline.append({
            "scene": sc["index"], "start": sc["start"], "end": sc["end"], "thumb": sc["thumb"], "visual": "",
            "caption": _uniq_join([c["text"] for c in caps], " / "),
            "caption_ko": _uniq_join([c["ko"] for c in caps], " / "),
            "speech": _uniq_join([s["text"] for s in sps], " "),
            "speech_ko": _uniq_join([s["ko"] for s in sps], " "),
        })
    src = speech if speech else screen
    script = [{"start": s["start"], "orig": s["text"].replace("\n", " "), "ko": s["ko"].replace("\n", " ")}
              for s in src]
    analysis = {"ai": False, "engine": "", "topic": None, "timeline": timeline, "script": script,
                "script_source": "speech" if speech else ("screen" if screen else ""), "error": ""}

    if has_claude() and timeline:
        status("analyzing", "AI가 영상을 보고 주제·화면 구성·스크립트를 정리하는 중…")
        try:
            ai = analyze_with_claude(result, sl, timeline, job_dir)
            topic = ai.get("topic") or {}
            analysis["topic"] = {"title": str(topic.get("title", "")), "summary": str(topic.get("summary", "")),
                                 "points": [str(p) for p in topic.get("points", [])]}
            by_scene = {int(t.get("scene", 0)): t for t in ai.get("timeline", []) if isinstance(t, dict)}
            for t in timeline:
                got = by_scene.get(t["scene"], {})
                t["visual"] = str(got.get("visual", ""))
                if got.get("caption"):
                    t["caption_ko"] = str(got["caption"])
            ai_script = [s for s in ai.get("script", []) if isinstance(s, dict) and s.get("ko")]
            if ai_script:
                analysis["script"] = [{"start": float(s.get("start", 0) or 0), "orig": str(s.get("orig", "")),
                                       "ko": str(s["ko"])} for s in ai_script]
            analysis["ai"], analysis["engine"] = True, f"Claude ({ANTHROPIC_MODEL})"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            analysis["error"] = f"AI 분석 실패: {str(e)[:300]}"

    if not analysis["ai"] and timeline:
        # 무료 모드: 제목은 화면 문구·캡션에서 추정, 화면 설명은 Florence-2가 장면 이미지를 보고 작성
        title, src = _free_title(sl, result)
        if title:
            analysis["topic"] = {"title": title, "summary": "", "points": [], "auto": True, "source": src}
        if captioner_available():
            try:
                descs, label = describe_scenes([job_dir / t["thumb"] for t in timeline],
                                               lambda m: status("analyzing", m),
                                               [t["caption"] for t in timeline])
                status("analyzing", "화면 설명을 한국어로 옮기는 중…")
                ko, tr_engine = translate([s for pair in descs for s in pair])
                if tr_engine.startswith("번역 실패"):
                    analysis["error"] = ("화면 설명을 한국어로 옮기지 못해서 영어로 보여드려요 (Google 번역이 잠시 막힘). "
                                         "몇 분 뒤 '다시 분석'을 누르면 한국어로 바뀌어요.")
                for i, t in enumerate(timeline):
                    t["visual_short"], t["visual"] = ko[2 * i], ko[2 * i + 1]
                    t["visual_en"] = descs[i][1]
                analysis["engine"] = label
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                analysis["error"] = f"화면 설명 만들기 실패: {str(e)[:300]}"

    # 전체 화면 구성 요약 (예: 00:00 ~ 00:57: 카페에 앉아 있는 남자, 노트북 화면, …)
    shorts = []
    for t in timeline:
        s = (t.get("visual_short") or t.get("visual") or "").strip().rstrip(".")
        if s and all(difflib.SequenceMatcher(None, s, x).ratio() < 0.7 for x in shorts):
            shorts.append(s)
    analysis["visual_summary"] = (
        f"{_mmss(timeline[0]['start'])} ~ {_mmss(timeline[-1]['end'])}: " + ", ".join(shorts)
        if timeline and shorts else "")
    analysis["report_text"] = report_text(analysis)
    return analysis


def report_text(a: dict) -> str:
    """다운로드·복사용 텍스트: 1) 제목  2) 스크립트  3) 타임라인과 화면 구성."""
    L = ["1) 제목"]
    t = a.get("topic")
    if t and t.get("title"):
        L.append(t["title"])
        if t.get("auto") and t.get("source"):
            L.append(f"  (자동 추정 기준: {t['source']})")
        if t.get("summary"):
            L.append(f"요약: {t['summary']}")
        if t.get("points"):
            L.append("핵심 포인트:")
            L += [f"  - {p}" for p in t["points"]]
    else:
        L.append("(제목으로 쓸 만한 화면 문구나 캡션이 없어요)")

    L += ["", "2) 스크립트"]
    if a.get("script_source") == "screen":
        L.append("(말소리가 없어서 화면 자막으로 정리했어요)")
    for s in a["script"]:
        L.append(f"[{_mmss(s['start'])}] \"{s['ko']}\"")
        if s.get("orig") and s["orig"] != s["ko"]:
            L.append(f"        ({s['orig']})")
    if not a["script"]:
        L.append("(말소리와 화면 자막이 없는 영상이에요)")

    L += ["", "3) 타임라인과 화면 구성"]
    if a.get("visual_summary"):
        L += [f"[전체 화면 구성] {a['visual_summary']}", ""]
    for x in a["timeline"]:
        L.append(f"{_mmss(x['start'])} ~ {_mmss(x['end'])}")
        if x.get("visual"):
            L.append(f"  화면: {x['visual']}")
        cap = x.get("caption_ko") or x.get("caption")
        if cap:
            L.append(f"  자막: \"{cap}\"")
    if not a["timeline"]:
        L.append("(장면 정보 없음)")
    return "\n".join(L)


def run_job(job_id: str, url: str):
    job_dir = DOWNLOADS / job_id
    job_dir.mkdir(exist_ok=True)

    def status(stage, msg):
        set_job(job_id, status=stage, message=msg)

    try:
        status("downloading", "게시물 내려받는 중…")
        meta = download(url, job_dir)
        set_job(job_id, uploader=meta["uploader"], kind=meta["kind"], slide_count=len(meta["slides"]))

        slides = meta.pop("slides")
        for sl in slides:
            process_slide(sl, job_dir, status)

        result = {
            "version": 2,
            "id": job_id,
            "url": url,
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "meta": meta,
            "caption": split_caption(meta.get("caption", "")),
            "slides": slides,
        }
        engine = translate_result(result, status)
        engines = {"translate": engine}
        for sl in slides:
            if sl["type"] == "image" and sl["ocr"].get("engine"):
                engines["ocr"] = sl["ocr"]["engine"]
            if sl["type"] == "video":
                if sl["speech"].get("engine"):
                    engines["speech"] = sl["speech"]["engine"]
                if sl["screen"].get("engine"):
                    engines["screen"] = sl["screen"]["engine"]
        result["analysis"] = build_analysis(result, job_dir, status)
        if result["analysis"] and result["analysis"].get("engine"):
            engines["analysis"] = result["analysis"]["engine"]
        result["engines"] = engines
        result["preview"] = preview_text(result)

        (job_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        set_job(job_id, status="done", message="완료", result=result)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        msg = str(e)
        low = msg.lower()
        if any(k in low for k in ("login", "rate-limit", "cookies", "401", "empty media")):
            msg = ("인스타그램이 이 게시물을 로그인 없이 보여주지 않았어요. 비공개 계정이거나, "
                   "짧은 시간에 너무 많이 요청해서 잠깐 막힌 경우예요.\n"
                   "→ 몇 분 뒤 다시 시도하거나, README의 '로그인이 필요하다는 오류' 방법(cookies.txt)을 써 주세요."
                   f"\n\n(상세: {msg[:300]})")
        set_job(job_id, status="error", message=msg)


def worker():
    while True:
        job_id = job_queue.get()
        try:
            with jobs_lock:
                job = jobs.get(job_id)
            if not job or job.get("status") == "canceled":  # 대기 중에 삭제된 작업
                continue
            run_job(job_id, job["url"])
        finally:
            job_queue.task_done()
            time.sleep(3)  # 인스타그램 요청 간격 (너무 빠르면 잠깐 차단됨)


threading.Thread(target=worker, daemon=True).start()


# --------------------------------------------------- Result normalization ---

def preview_text(r: dict) -> str:
    for sl in r["slides"]:
        if sl["type"] == "image" and sl["ocr"]["ko"].strip():
            return sl["ocr"]["ko"].replace("\n", " ")[:80]
        if sl["type"] == "video":
            for key in ("speech", "screen"):
                segs = sl[key]["segments"]
                if segs:
                    return " ".join(s["ko"] for s in segs)[:80]
    return r["caption"].get("body_ko", "")[:80]


def normalize(r: dict) -> dict:
    """예전(v1, 영상 1개) 결과를 새 형식으로 변환."""
    if r.get("version") == 2:
        return r
    meta = r.get("meta", {})
    slide = {
        "index": 1, "type": "video", "file": meta.get("file", "video.mp4"),
        "duration": meta.get("duration"),
        "speech": {"language": r.get("language", ""), "language_name": r.get("language_name", ""),
                   "segments": r.get("segments", []), "has_audio": True},
        "screen": {"segments": [], "pinned": [], "pinned_ko": [], "engine": ""},
    }
    cap = r.get("caption") or split_caption(meta.get("caption", ""))
    out = {
        "version": 2, "id": r["id"], "url": r.get("url", ""), "created": r.get("created", ""),
        "meta": {"kind": "reel", "uploader": meta.get("uploader", ""), "caption": meta.get("caption", "")},
        "caption": cap, "slides": [slide],
        "engines": {"speech": "Whisper " + r.get("whisper", ""), "translate": r.get("engine", "")},
    }
    out["preview"] = preview_text(out)
    return out


def load_result(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if job and job.get("result"):
        return job["result"]
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    path = DOWNLOADS / job_id / "result.json"
    if not path.exists():
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return normalize(json.loads(path.read_text(encoding="utf-8")))


# ----------------------------------------------------------------- Export ---

def _clock(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _srt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def export_txt(r: dict) -> str:
    L = []
    m = r["meta"]
    L.append("=" * 60)
    L.append(f"게시물: {r['url']}")
    if m.get("uploader"):
        L.append(f"계정: @{m['uploader']}")
    L.append(f"추출일: {r.get('created', '')}")
    if r.get("analysis"):
        L += ["", "■■ 영상 분석 ■■", r["analysis"]["report_text"], "", "■■ 상세 추출 결과 ■■"]
    cap = r["caption"]
    if cap.get("full"):
        L += ["", "■ 캡션 (원문)", cap["body"] or "(본문 없음)"]
        if not cap.get("is_korean") and cap.get("body_ko") and cap["body_ko"] != cap["body"]:
            L += ["", "■ 캡션 (한국어)", cap["body_ko"]]
        if cap["hashtags"]:
            L += ["", "■ 해시태그", " ".join(h["tag"] for h in cap["hashtags"])]
            means = [f"{h['tag']} = {h['ko']}" for h in cap["hashtags"] if h.get("ko") and h["ko"] != h["tag"][1:]]
            if means:
                L.append("  (" + ", ".join(means) + ")")
    multi = len(r["slides"]) > 1
    for sl in r["slides"]:
        L += ["", "-" * 60]
        head = f"{sl['index']}장 " if multi else ""
        if sl["type"] == "image":
            L.append(f"■ {head}이미지 글자")
            L += [sl["ocr"]["text"] or "(글자 없음)"]
            if sl["ocr"]["ko"] != sl["ocr"]["text"]:
                L += ["", "→ 한국어", sl["ocr"]["ko"]]
            continue
        sp = sl["speech"]
        if sp["segments"]:
            L.append(f"■ {head}음성 ({sp.get('language_name') or '원문'})")
            for s in sp["segments"]:
                L.append(f"[{_clock(s['start'])}] {s['text']}")
                if s["ko"] != s["text"]:
                    L.append(f"        → {s['ko']}")
        sc = sl["screen"]
        if sc["pinned"] or sc["segments"]:
            L += ["", f"■ {head}화면 속 글자"]
            for p, pk in zip(sc["pinned"], sc["pinned_ko"]):
                L.append(f"[고정] {p}" + (f"  → {pk}" if pk != p else ""))
            for s in sc["segments"]:
                L.append(f"[{_clock(s['start'])}] " + s["text"].replace("\n", " / "))
                if s["ko"] != s["text"]:
                    L.append("        → " + s["ko"].replace("\n", " / "))
        if not sp["segments"] and not (sc["pinned"] or sc["segments"]):
            L.append(f"■ {head}영상 (추출된 텍스트 없음)")
    return "\n".join(L) + "\n"


def _download_response(body: str | bytes, filename: str, media: str):
    from urllib.parse import quote

    return Response(
        body if isinstance(body, bytes) else body.encode("utf-8"),
        media_type=media,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


# -------------------------------------------------------------------- API ---

class JobsIn(BaseModel):
    urls: list[str] | None = None
    url: str | None = None
    text: str | None = None  # 여러 링크를 붙여넣은 원문


@app.post("/api/jobs")
def create_jobs(body: JobsIn):
    raw = " ".join(filter(None, [body.text, body.url, *(body.urls or [])]))
    found = []
    for m in INSTAGRAM_RE.finditer(raw):
        kind = "reel" if m.group(1).lower().startswith("reel") else m.group(1).lower()
        found.append(f"https://www.instagram.com/{kind}/{m.group(2)}/")
    urls = list(dict.fromkeys(found))  # 중복 제거
    if not urls:
        raise HTTPException(400, "인스타그램 게시물/릴스 링크를 넣어주세요 (instagram.com/reel/... 또는 /p/...)")
    ids = []
    with jobs_lock:
        for url in urls:
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = {"id": job_id, "status": "queued", "message": "대기 중…", "url": url,
                            "uploader": "", "kind": "", "slide_count": 0}
            job_order.append(job_id)
            ids.append(job_id)
    for job_id in ids:
        job_queue.put(job_id)
    return {"ids": ids}


@app.get("/api/queue")
def get_queue():
    """이번 실행에서 넣은 작업들의 상태 (결과 본문 제외)."""
    with jobs_lock:
        out = []
        waiting = 0
        for jid in job_order:
            j = jobs[jid]
            item = {k: j.get(k) for k in ("id", "url", "status", "message", "uploader", "kind", "slide_count")}
            if j["status"] == "queued":
                waiting += 1
                item["position"] = waiting
            if j.get("result"):
                item["preview"] = j["result"].get("preview", "")
            out.append(item)
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job:
        return job
    return {"id": job_id, "status": "done", "message": "완료", "result": load_result(job_id)}


@app.get("/api/history")
def history():
    items = []
    for p in DOWNLOADS.glob("*/result.json"):
        try:
            r = normalize(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        items.append({
            "id": r["id"], "created": r.get("created", ""), "url": r.get("url", ""),
            "uploader": r["meta"].get("uploader", ""), "kind": r["meta"].get("kind", "reel"),
            "slide_count": len(r["slides"]), "preview": r.get("preview", ""),
        })
    items.sort(key=lambda x: x["created"], reverse=True)
    return items[:50]


# ----------------------------------------------------------- 정리(삭제) ---

ACTIVE = {"downloading", "transcribing", "ocr", "translating", "analyzing"}


def _folder_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _remove_folder(p: Path) -> bool:
    """폴더 삭제. 윈도우에서 영상이 재생 중이라 잠겨 있으면 잠깐 기다렸다 다시 시도."""
    import shutil

    for _ in range(5):
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            return True
        time.sleep(0.5)
    return False


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """결과 하나 삭제 (영상·이미지·텍스트 모두)."""
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    with jobs_lock:
        job = jobs.get(job_id)
        if job and job.get("status") in ACTIVE:
            raise HTTPException(409, "지금 처리 중인 작업은 끝난 뒤에 삭제할 수 있어요")
        if job:
            job["status"] = "canceled"  # 대기 중이었다면 처리하지 않도록
            jobs.pop(job_id, None)
            if job_id in job_order:
                job_order.remove(job_id)
    folder = DOWNLOADS / job_id
    if folder.exists() and not _remove_folder(folder):
        raise HTTPException(409, "파일이 사용 중이라 지우지 못했어요. 영상 재생을 멈추고 다시 시도해 주세요")
    return {"ok": True}


@app.delete("/api/history")
def delete_all():
    """처리 중인 작업을 뺀 모든 결과 삭제."""
    with jobs_lock:
        busy = {jid for jid, j in jobs.items() if j.get("status") in ACTIVE or j.get("status") == "queued"}
        for jid in [j for j in job_order if j not in busy]:
            job_order.remove(jid)
            jobs.pop(jid, None)
    removed, failed, freed = 0, 0, 0
    for folder in DOWNLOADS.iterdir():
        if not folder.is_dir() or folder.name in busy:
            continue
        size = _folder_size(folder)
        if _remove_folder(folder):
            removed, freed = removed + 1, freed + size
        else:
            failed += 1
    return {"removed": removed, "failed": failed, "freed_mb": round(freed / 1024 / 1024, 1)}


@app.post("/api/queue/clear")
def clear_queue():
    """작업 목록에서 끝난 항목만 지움 (결과 파일은 그대로, 기록에서 다시 볼 수 있음)."""
    with jobs_lock:
        for jid in list(job_order):
            if jobs.get(jid, {}).get("status") in ("done", "error", "canceled"):
                job_order.remove(jid)
    return {"ok": True}


@app.get("/api/storage")
def storage():
    folders = [p for p in DOWNLOADS.iterdir() if p.is_dir()]
    total = sum(_folder_size(p) for p in folders)
    return {"count": len(folders), "mb": round(total / 1024 / 1024, 1)}


def _slide_path(job_id: str, name: str) -> Path:
    r = load_result(job_id)
    allowed = {s["file"] for s in r["slides"]}
    allowed |= {sc["thumb"] for s in r["slides"] for sc in (s.get("scenes") or [])}
    if name not in allowed:
        raise HTTPException(404, "파일이 없습니다")
    path = DOWNLOADS / job_id / name
    if not path.exists():
        raise HTTPException(404, "파일이 없습니다")
    return path


@app.get("/api/jobs/{job_id}/file/{name}")
def get_file(job_id: str, name: str, download: int = 0):
    path = _slide_path(job_id, name)
    r = load_result(job_id)
    fname = f"{r['meta'].get('uploader') or 'instagram'}_{job_id}_{name}"
    return FileResponse(path, filename=fname if download else None)


@app.get("/api/jobs/{job_id}/zip")
def get_zip(job_id: str):
    r = load_result(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for s in r["slides"]:
            p = DOWNLOADS / job_id / s["file"]
            if p.exists():
                z.write(p, s["file"])
        z.writestr("텍스트.txt", export_txt(r))
    name = f"{r['meta'].get('uploader') or 'instagram'}_{job_id}.zip"
    return _download_response(buf.getvalue(), name, "application/zip")


@app.get("/api/jobs/{job_id}/txt")
def get_txt(job_id: str):
    r = load_result(job_id)
    name = f"{r['meta'].get('uploader') or 'instagram'}_{job_id}.txt"
    return _download_response(export_txt(r), name, "text/plain; charset=utf-8")


@app.get("/api/jobs/{job_id}/srt")
def get_srt(job_id: str, slide: int = 1, kind: str = "speech", lang: str = "orig"):
    r = load_result(job_id)
    sl = next((s for s in r["slides"] if s["index"] == slide and s["type"] == "video"), None)
    if not sl:
        raise HTTPException(404, "영상이 없습니다")
    segs = sl["screen" if kind == "screen" else "speech"]["segments"]
    key = "ko" if lang == "ko" else "text"
    body = "\n".join(f"{i}\n{_srt_ts(s['start'])} --> {_srt_ts(s['end'])}\n{s[key]}\n"
                     for i, s in enumerate(segs, 1))
    name = f"{job_id}_{slide}_{kind}_{'ko' if lang == 'ko' else 'orig'}.srt"
    return _download_response(body, name, "text/plain; charset=utf-8")


@app.get("/api/export")
def export_many(ids: str):
    parts = []
    for jid in [i for i in ids.split(",") if i]:
        try:
            parts.append(export_txt(load_result(jid)))
        except HTTPException:
            continue
    if not parts:
        raise HTTPException(404, "내보낼 결과가 없습니다")
    return _download_response("\n".join(parts), f"인스타_추출_{time.strftime('%Y%m%d_%H%M')}.txt",
                              "text/plain; charset=utf-8")


_analyze_lock = threading.Lock()


@app.post("/api/jobs/{job_id}/analyze")
def reanalyze(job_id: str):
    """영상 분석을 다시 실행 (예: 나중에 Claude API 키를 넣었을 때, 예전 결과에 분석 추가)."""
    r = load_result(job_id)
    sl = analysis_target(r)
    if not sl:
        raise HTTPException(400, "영상 분석은 릴스(영상 1개) 게시물만 할 수 있어요")
    job_dir = DOWNLOADS / job_id
    with _analyze_lock:
        if not sl.get("scenes"):
            try:
                files, fps = extract_frames(job_dir / sl["file"], job_dir)
                sl["scenes"] = detect_scenes(files, fps, job_dir, sl["index"])
            finally:
                cleanup_frames(job_dir)
        r["analysis"] = build_analysis(r, job_dir)
        r.setdefault("engines", {})
        if r["analysis"].get("engine"):
            r["engines"]["analysis"] = r["analysis"]["engine"]
        (job_dir / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["result"] = r
    return r


@app.post("/api/jobs/{job_id}/retranslate")
def retranslate(job_id: str):
    """번역을 다시 실행 (예: 번역이 막혀서 원문 그대로 남은 결과)."""
    r = load_result(job_id)
    job_dir = DOWNLOADS / job_id
    with _analyze_lock:
        for sl in r["slides"]:
            if sl["type"] == "image":
                sl["ocr"]["ko"] = sl["ocr"]["text"]
            else:
                for s in sl["speech"]["segments"] + sl["screen"]["segments"]:
                    s["ko"] = s["text"]
                sl["screen"]["pinned_ko"] = list(sl["screen"]["pinned"])
        cap = r["caption"]
        cap["body_ko"] = cap["body"]
        for h in cap["hashtags"]:
            h["ko"] = ""
        engine = translate_result(r, lambda st, m: None)
        r.setdefault("engines", {})["translate"] = engine
        if analysis_target(r):
            r["analysis"] = build_analysis(r, job_dir)
        r["preview"] = preview_text(r)
        (job_dir / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["result"] = r
    return r


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str):
    r = load_result(job_id)
    if not r.get("analysis"):
        raise HTTPException(404, "분석 결과가 없습니다")
    name = f"{r['meta'].get('uploader') or 'instagram'}_{job_id}_분석.txt"
    return _download_response(r["analysis"]["report_text"] + "\n", name, "text/plain; charset=utf-8")


@app.get("/api/status")
def server_status():
    return {"claude": has_claude(), "ocr": ocr_available(), "screen_ocr": SCREEN_OCR}


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")


if __name__ == "__main__":
    import socket
    import webbrowser

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    url = f"http://localhost:{port}"
    open_browser = os.getenv("OPEN_BROWSER") == "1"

    with socket.socket() as s:
        already_running = s.connect_ex(("127.0.0.1", port)) == 0
    if already_running:  # 이미 켜져 있으면 브라우저만 열기
        print(f"\n  이미 실행 중이에요 → {url}\n")
        if open_browser:
            webbrowser.open(url)
        raise SystemExit(0)

    print(f"\n  ▶ {url} 에서 열어주세요\n")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
