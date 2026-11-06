# 인스타 영상 텍스트 추출기 - 자동 설치 & 실행
# run.bat 을 더블클릭하면 이 스크립트가 실행됩니다.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "인스타 영상 텍스트 추출기 (이 창을 닫으면 프로그램이 꺼져요)"

function Say($msg, $color = "Cyan") { Write-Host ""; Write-Host "▶ $msg" -ForegroundColor $color }

# 지원하는 Python 버전: 3.10 ~ 3.14
# (3.15 이상은 onnxruntime·scikit-image 등 필요한 부품이 아직 나오지 않아서 설치가 실패함)
$PyCheck = "import sys; print(sys.executable) if (3, 10) <= sys.version_info[:2] <= (3, 14) else sys.exit(1)"

function Test-Py($exe, [string[]]$pre = @()) {
    try {
        $out = & $exe @pre -c $PyCheck 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ($out | Select-Object -Last 1).Trim() }
    } catch {}
    return $null
}

function Find-Python {
    # 여러 버전이 깔려 있으면 검증된 버전부터 (최신 3.15 등은 건너뜀)
    foreach ($v in @("3.13", "3.12", "3.14", "3.11", "3.10")) {
        $p = Test-Py "py" @("-$v"); if ($p) { return $p }
    }
    $p = Test-Py "python"; if ($p) { return $p }
    $roots = @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles", "${env:ProgramFiles(x86)}")
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $cands = Get-ChildItem $root -Directory -Filter "Python3*" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($c in $cands) {
            $exe = Join-Path $c.FullName "python.exe"
            if (Test-Path $exe) { $p = Test-Py $exe; if ($p) { return $p } }
        }
    }
    return $null
}

try {
    # Windows는 파일 경로가 260자를 넘으면 설치가 실패해요 (PyTorch 안에 아주 깊은 폴더가 있음).
    # 폴더 경로가 길면 실행 환경을 짧은 경로(%LOCALAPPDATA%\InstaText\...)에 만듭니다.
    $venvDir = Join-Path $PSScriptRoot ".venv"
    if ($venvDir.Length -gt 80) {
        $sha = [System.Security.Cryptography.SHA1]::Create()
        $id = -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($PSScriptRoot.ToLower()))[0..3] | ForEach-Object { $_.ToString("x2") })
        $venvDir = Join-Path $env:LOCALAPPDATA "InstaText\$id"
        Say "폴더 경로가 길어서 실행 환경은 $venvDir 에 만들어요." "DarkGray"
    }
    $venvPy = Join-Path $venvDir "Scripts\python.exe"

    # 예전에 지원하지 않는 Python(3.15 등)으로 만든 실행 환경이면 지우고 다시 만들기
    if ((Test-Path $venvPy) -and -not (Test-Py $venvPy)) {
        Say "지원하지 않는 Python 버전으로 만든 실행 환경이라 다시 만들어요..." "Yellow"
        Remove-Item -LiteralPath "\\?\$venvDir" -Recurse -Force -ErrorAction SilentlyContinue
    }

    # 1) Python
    if (-not (Test-Path $venvPy)) {
        $py = Find-Python
        if (-not $py) {
            Say "사용할 수 있는 Python(3.10~3.14)이 없어서 Python 3.12를 자동으로 설치합니다 (1~3분)..."
            Say "(이미 최신 Python 3.15 이상이 깔려 있어도 괜찮아요. 3.12를 따로 하나 더 설치해서 이 프로그램에만 써요)" "DarkGray"
            if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
                Say "자동 설치를 할 수 없는 PC예요. 열리는 페이지에서 'Python 3.12' 를 설치한 뒤 다시 실행해 주세요. (최신 3.15는 안 돼요)" "Yellow"
                Say "설치 화면 맨 아래 'Add python.exe to PATH' 를 꼭 체크하세요!" "Yellow"
                Start-Process "https://www.python.org/downloads/windows/"
                exit 1
            }
            winget install -e --id Python.Python.3.12 --scope user --silent `
                --accept-package-agreements --accept-source-agreements | Out-Host
            $py = Find-Python
            if (-not $py) { throw "Python 설치 후에도 찾을 수 없습니다. PC를 재시작한 뒤 다시 실행해 주세요." }
        }
        Say "실행 환경을 만듭니다... ($py)"
        & $py -m venv $venvDir
        if ($LASTEXITCODE -ne 0) { throw "가상환경(.venv) 생성 실패" }
        & $venvPy -m pip install --upgrade pip --disable-pip-version-check -q
    }

    $hasGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)

    # 2) 글자 인식(OCR)용 PyTorch - 그래픽카드 유무에 맞는 버전으로 한 번만 설치
    $torchFlag = Join-Path $venvDir ".torch-installed"
    if (-not (Test-Path $torchFlag)) {
        if ($hasGpu) {
            Say "글자 인식용 AI 엔진(GPU 버전, 약 2.5GB)을 설치합니다. 시간이 좀 걸려요..."
            & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check
        } else {
            Say "글자 인식용 AI 엔진(CPU 버전, 약 200MB)을 설치합니다..."
            & $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
        }
        if ($LASTEXITCODE -ne 0) { throw "PyTorch 설치 실패. 인터넷 연결을 확인하고 다시 실행해 주세요." }
        Set-Content $torchFlag "done"
    }

    # 3) 패키지 (requirements.txt 가 바뀌었을 때만 다시 설치)
    $hashFile = Join-Path $venvDir ".requirements-hash"
    $reqHash = (Get-FileHash requirements.txt -Algorithm SHA256).Hash
    if (-not (Test-Path $hashFile) -or (Get-Content $hashFile -Raw).Trim() -ne $reqHash) {
        Say "필요한 프로그램을 설치합니다. 처음 한 번만 하며 몇 분 걸려요..."
        & $venvPy -m pip install -r requirements.txt --disable-pip-version-check
        if ($LASTEXITCODE -ne 0) { throw "패키지 설치 실패. 인터넷 연결을 확인하고 다시 실행해 주세요." }
        Set-Content $hashFile $reqHash
    }

    # 4) NVIDIA 그래픽카드가 있으면 음성 인식용 GPU 가속 라이브러리 (약 1GB, 한 번만)
    $gpuFlag = Join-Path $venvDir ".gpu-checked"
    if (-not (Test-Path $gpuFlag)) {
        if ($hasGpu) {
            Say "NVIDIA 그래픽카드를 찾았어요. GPU 가속 라이브러리를 설치합니다 (약 1GB)..."
            & $venvPy -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" --disable-pip-version-check
            if ($LASTEXITCODE -ne 0) { Say "GPU 라이브러리 설치에 실패했어요. CPU 모드로 계속합니다." "Yellow" }
        }
        Set-Content $gpuFlag "done"
    }

    # 5) 인스타그램 변경에 대응하기 위해 다운로더는 항상 최신으로
    Say "다운로더(yt-dlp) 업데이트 확인..."
    & $venvPy -m pip install -U yt-dlp -q --disable-pip-version-check

    # 6) 실행
    Say "프로그램을 시작합니다. 잠시 후 브라우저가 열려요." "Green"
    Say "다 쓰셨으면 이 창을 닫으면 됩니다." "Green"
    if (-not $env:OPEN_BROWSER) { $env:OPEN_BROWSER = "1" }
    & $venvPy app.py
} catch {
    Write-Host ""
    Write-Host "오류: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
