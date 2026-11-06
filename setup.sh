#!/bin/bash
# 인스타 텍스트 추출기 - macOS 자동 설치 & 실행
# run.command 를 더블클릭하면 이 스크립트가 실행됩니다. (터미널에서 bash setup.sh 로도 실행 가능)

cd "$(dirname "$0")" || exit 1

say()  { printf "\n\033[36m▶ %s\033[0m\n" "$1"; }
note() { printf "\n\033[33m▶ %s\033[0m\n" "$1"; }
fail() {
    printf "\n\033[31m오류: %s\033[0m\n" "$1"
    read -r -p "엔터를 누르면 창이 닫혀요..." _
    exit 1
}

if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" != "arm64" ]; then
    note "인텔 맥은 일부 AI 엔진(PyTorch)이 지원되지 않아 설치가 안 될 수 있어요. M1 이상 맥을 권장해요."
fi

# 1) Python 찾기: 3.10 ~ 3.14 만 사용
#    - 맥 기본 python3 는 3.9라서 너무 오래됨
#    - 최신 3.15 이상은 onnxruntime·scikit-image 등 필요한 부품이 아직 없어서 설치가 실패함
PY_OK='import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)'

find_python() {
    for c in python3.13 python3.12 python3.14 python3.11 python3.10 \
             /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
             /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
             /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
             /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
        if command -v "$c" >/dev/null 2>&1; then
            if "$c" -c "$PY_OK" 2>/dev/null; then
                command -v "$c"
                return 0
            fi
        fi
    done
    # uv 로 설치해 둔 Python
    if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
        UVBIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
        P="$("$UVBIN" python find 3.12 2>/dev/null)"
        if [ -n "$P" ] && "$P" -c "$PY_OK" 2>/dev/null; then echo "$P"; return 0; fi
    fi
    return 1
}

VENV=".venv"
PY="$VENV/bin/python"

# 예전에 지원하지 않는 Python(3.15 등)으로 만든 실행 환경이면 지우고 다시 만들기
if [ -x "$PY" ] && ! "$PY" -c "$PY_OK" 2>/dev/null; then
    note "지원하지 않는 Python 버전으로 만든 실행 환경이라 다시 만들어요..."
    rm -rf "$VENV"
fi

if [ ! -x "$PY" ]; then
    P="$(find_python)"
    if [ -z "$P" ]; then
        say "사용할 수 있는 Python(3.10~3.14)이 없어서 Python 3.12를 자동으로 설치합니다 (몇 분 걸려요)..."
        note "(이미 최신 Python 3.15 이상이 깔려 있어도 괜찮아요. 3.12를 따로 하나 더 설치해서 이 프로그램에만 써요)"
        if command -v brew >/dev/null 2>&1; then
            brew install python@3.12 && P="$(find_python)"
        fi
    fi
    if [ -z "$P" ]; then
        # Homebrew가 없으면 uv(파이썬 설치 도구, astral.sh)로 관리자 권한 없이 설치
        if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh || true
        fi
        UVBIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
        if [ -x "$UVBIN" ]; then
            "$UVBIN" python install 3.12 && P="$("$UVBIN" python find 3.12)"
        fi
    fi
    if [ -z "$P" ]; then
        note "Python 자동 설치에 실패했어요. 열리는 페이지에서 'Python 3.12' (최신 3.15 말고) 의 macOS installer 를 받아 설치한 뒤 다시 실행해 주세요."
        open "https://www.python.org/downloads/macos/"
        fail "Python 3.12 설치가 필요해요"
    fi
    say "실행 환경을 만듭니다... ($P)"
    "$P" -m venv "$VENV" || fail "가상환경(.venv) 생성 실패"
    "$PY" -m pip install --upgrade pip -q --disable-pip-version-check
fi

# 2) 글자 인식·화면 설명용 PyTorch (한 번만)
if [ ! -f "$VENV/.torch-installed" ]; then
    say "AI 엔진(PyTorch)을 설치합니다. 처음 한 번만 하며 몇 분 걸려요..."
    "$PY" -m pip install torch torchvision --disable-pip-version-check || fail "PyTorch 설치 실패. 인터넷 연결을 확인해 주세요."
    touch "$VENV/.torch-installed"
fi

# 3) 패키지 (requirements.txt 가 바뀌었을 때만 다시 설치)
HASH="$(shasum -a 256 requirements.txt | cut -d' ' -f1)"
if [ ! -f "$VENV/.requirements-hash" ] || [ "$(cat "$VENV/.requirements-hash")" != "$HASH" ]; then
    say "필요한 프로그램을 설치합니다. 처음 한 번만 하며 몇 분 걸려요..."
    "$PY" -m pip install -r requirements.txt --disable-pip-version-check || fail "패키지 설치 실패. 인터넷 연결을 확인해 주세요."
    echo "$HASH" > "$VENV/.requirements-hash"
fi

# 4) 인스타그램 변경에 대응하기 위해 다운로더는 항상 최신으로
say "다운로더(yt-dlp) 업데이트 확인..."
"$PY" -m pip install -U yt-dlp -q --disable-pip-version-check

# 5) 실행
say "프로그램을 시작합니다. 잠시 후 브라우저가 열려요."
say "다 쓰셨으면 이 창을 닫으면 됩니다."
export OPEN_BROWSER="${OPEN_BROWSER:-1}"
"$PY" app.py
