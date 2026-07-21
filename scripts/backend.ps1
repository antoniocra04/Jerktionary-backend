param(
    [ValidateSet("run", "install", "test", "lint", "format")]
    [string]$Command = "run",
    [switch]$Dev,
    [switch]$PullLlm,
    [switch]$SkipOllamaInstall,
    [switch]$SkipModelSelect,
    [switch]$SkipProviderSelect,
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Curated catalogs for the interactive model picker (see Select-Models). Each entry:
#   name  = exact tag the backend downloads (ollama pull / faster-whisper HuggingFace id)
#   label = human-readable label shown in the menu
#   size  = approximate download footprint
#   note  = one-line description
$LlmModels = @(
    @{ name = "qwen3.5:9b"; label = "qwen3.5:9b"; size = "~6.6 GB"; note = "Рекомендуется: лучший баланс для русского" },
    @{ name = "qwen3:8b";   label = "qwen3:8b";   size = "~5.0 GB"; note = "Стабильный официальный Qwen3" },
    @{ name = "qwen3.5:4b"; label = "qwen3.5:4b"; size = "~2.5 GB"; note = "Быстрее, слабее" }
)

$WhisperModels = @(
    @{ name = "large-v3-turbo"; label = "large-v3-turbo"; size = "~1.6 GB"; note = "Рекомендуется: баланс скорости и качества" },
    @{ name = "large-v3";       label = "large-v3";       size = "~3.0 GB"; note = "Максимальное качество" },
    @{ name = "medium";         label = "medium";         size = "~1.5 GB"; note = "" },
    @{ name = "small";          label = "small";          size = "~480 MB"; note = "" },
    @{ name = "base";           label = "base";           size = "~145 MB"; note = "" },
    @{ name = "tiny";           label = "tiny";           size = "~75 MB";  note = "Самая быстрая" },
    @{ name = "nemotron-3.5-asr-streaming-0.6b"; label = "Nemotron 3.5 ASR Streaming 0.6B"; size = "~2.5 GB"; note = "Реальное время (NVIDIA NeMo): стриминг, пунктуация; ставит доп. зависимости" }
)

# LLM provider catalog — API providers only (mirrors app/core/providers.py
# LLM_PROVIDERS minus "ollama", which is handled by the "локально" branch of the
# launcher menu). Each entry:
#   key          = stored in .env as LLM_PROVIDER and routed by startup
#   label        = menu label
#   base_url     = used when LLM_API_BASE_URL is empty
#   default_model= used when LLM_API_MODEL is empty
# Load LLM providers from the canonical Python catalog (scripts/export_providers.py).
# Keep in sync with app/core/providers.py — add/edit/remove entries there only.
$_ExportScript = Join-Path $PSScriptRoot "export_providers.py"
$_ExportJson = & python $_ExportScript 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARN failed to run export_providers.py; providers menu will be unavailable" -ForegroundColor Yellow
    $_ExportJson = "[]"
}
$LlmProviders = $_ExportJson | ConvertFrom-Json

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PrimaryVenvDir = Join-Path $Root ".venv"
$FallbackVenvDir = Join-Path $env:LOCALAPPDATA "RealtimeTermsBackend\.venv"
$VenvDir = $PrimaryVenvDir
$Python = Join-Path $VenvDir "Scripts\python.exe"
$EnvFile = Join-Path $Root ".env"
$EnvExample = Join-Path $Root ".env.example"
$DataDir = Join-Path $Root "data"

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "OK  $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "WARN $Message" -ForegroundColor Yellow
}

function Use-Venv {
    param([string]$Path)
    $script:VenvDir = $Path
    $script:Python = Join-Path $script:VenvDir "Scripts\python.exe"
}

function Find-SystemPython {
    $Candidates = @(
        @("py", "-3.11"),
        @("py", "-3"),
        @("python", "")
    )

    foreach ($Candidate in $Candidates) {
        $Exe = $Candidate[0]
        $Arg = $Candidate[1]
        $CommandInfo = Get-Command $Exe -ErrorAction SilentlyContinue
        if (-not $CommandInfo) {
            continue
        }

        try {
            if ($Arg) {
                & $Exe $Arg -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
            } else {
                & $Exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
            }
            if ($LASTEXITCODE -eq 0) {
                return @{ Exe = $Exe; Arg = $Arg }
            }
        } catch {
            continue
        }
    }

    throw "Python 3.11+ not found. Install it first: winget install -e --id Python.Python.3.11 --scope user"
}

function Ensure-SystemPython {
    try {
        return Find-SystemPython
    } catch {
        $Winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $Winget) {
            throw
        }

        Write-Warn "Python 3.11+ not found"
        Write-Step "installing Python 3.11 via winget"
        Invoke-Checked "winget" @(
            "install",
            "-e",
            "--id",
            "Python.Python.3.11",
            "--scope",
            "user",
            "--accept-package-agreements",
            "--accept-source-agreements"
        )
        return Find-SystemPython
    }
}

function Invoke-SystemPython {
    param(
        [hashtable]$SystemPython,
        [string[]]$Arguments
    )

    if ($SystemPython.Arg) {
        & $SystemPython.Exe $SystemPython.Arg @Arguments
    } else {
        & $SystemPython.Exe @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Test-VenvPip {
    if (-not (Test-Path $Python)) {
        return $false
    }

    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $script:ErrorActionPreference = "Continue"
        & $Python -m pip --version *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        $script:ErrorActionPreference = $PreviousErrorActionPreference
    }
}

function Ensure-Venv {
    Use-Venv $PrimaryVenvDir
    if (Test-VenvPip) {
        Write-Ok "venv exists"
        return
    }

    if (Test-Path $Python) {
        Write-Warn ".venv exists but pip is broken; using $FallbackVenvDir"
        Use-Venv $FallbackVenvDir
        if (Test-VenvPip) {
            Write-Ok "fallback venv exists"
            return
        }
    }

    Write-Step "creating venv at $VenvDir"
    $SystemPython = Ensure-SystemPython
    Invoke-SystemPython $SystemPython @("-m", "venv", $VenvDir)
    Write-Ok "venv created"
}

function Ensure-ProjectFiles {
    if (-not (Test-Path $EnvFile)) {
        Copy-Item $EnvExample $EnvFile
        Write-Ok ".env created from .env.example"
    } else {
        Write-Ok ".env exists"
    }

    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir | Out-Null
        Write-Ok "data directory created"
    } else {
        Write-Ok "data directory exists"
    }
}

function Get-EnvValue {
    param(
        [string]$Name,
        [string]$Default = ""
    )

    if (-not (Test-Path $EnvFile)) {
        return $Default
    }

    $Pattern = "^\s*$([regex]::Escape($Name))\s*=\s*(.*)\s*$"
    foreach ($Line in Get-Content $EnvFile) {
        if ($Line -match $Pattern) {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }

    return $Default
}

function Set-EnvValue {
    param(
        [string]$Name,
        [string]$Value
    )

    $Pattern = "^\s*$([regex]::Escape($Name))\s*="
    $Lines = [System.Collections.Generic.List[string]]::new()
    if (Test-Path $EnvFile) {
        $Lines.AddRange([string[]](Get-Content $EnvFile))
    }

    $Updated = $false
    for ($Index = 0; $Index -lt $Lines.Count; $Index++) {
        if ($Lines[$Index] -match $Pattern) {
            $Lines[$Index] = "$Name=$Value"
            $Updated = $true
            break
        }
    }

    if (-not $Updated) {
        $Lines.Add("$Name=$Value")
    }

    Set-Content -Path $EnvFile -Value $Lines -Encoding UTF8
}

function Test-EnvEnabled {
    param([string]$Name)

    $Value = (Get-EnvValue $Name "true").ToLowerInvariant()
    return -not ($Value -in @("0", "false", "no", "off"))
}

function Ensure-GpuSettings {
    $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    $HasGpu = $false
    if ($NvidiaSmi) {
        & $NvidiaSmi *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "NVIDIA GPU detected"
            $HasGpu = $true
        } else {
            Write-Warn "nvidia-smi is present but returned an error"
        }
    } else {
        Write-Warn "nvidia-smi not found; Whisper will run on CPU"
    }

    if ($HasGpu) {
        if ((Get-EnvValue "WHISPER_DEVICE" "cuda") -ne "cuda") {
            Set-EnvValue "WHISPER_DEVICE" "cuda"
            Write-Ok "WHISPER_DEVICE set to cuda"
        } else {
            Write-Ok "WHISPER_DEVICE=cuda"
        }
        if ((Get-EnvValue "WHISPER_COMPUTE_TYPE" "float16") -ne "float16") {
            Set-EnvValue "WHISPER_COMPUTE_TYPE" "float16"
            Write-Ok "WHISPER_COMPUTE_TYPE set to float16"
        } else {
            Write-Ok "WHISPER_COMPUTE_TYPE=float16"
        }
    } else {
        if ((Get-EnvValue "WHISPER_DEVICE" "cuda") -ne "cpu") {
            Set-EnvValue "WHISPER_DEVICE" "cpu"
            Write-Ok "WHISPER_DEVICE set to cpu"
        } else {
            Write-Ok "WHISPER_DEVICE=cpu"
        }
        if ((Get-EnvValue "WHISPER_COMPUTE_TYPE" "int8") -ne "int8") {
            Set-EnvValue "WHISPER_COMPUTE_TYPE" "int8"
            Write-Ok "WHISPER_COMPUTE_TYPE set to int8"
        } else {
            Write-Ok "WHISPER_COMPUTE_TYPE=int8"
        }
    }
}

function Test-Import {
    param([string]$Module)

    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $script:ErrorActionPreference = "Continue"
        & $Python -c "import $Module" *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        $script:ErrorActionPreference = $PreviousErrorActionPreference
    }
}

function Find-Ollama {
    $CommandInfo = Get-Command ollama -ErrorAction SilentlyContinue
    if ($CommandInfo) {
        return $CommandInfo.Source
    }

    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) {
            return $Candidate
        }
    }

    return $null
}

function Install-Dependencies {
    param([switch]$WithDev)

    Write-Step "upgrading pip"
    Invoke-Checked $Python @("-m", "pip", "install", "--quiet", "--upgrade", "pip")

    # Force setuptools<81 explicitly: pymorphy2 (via natasha) imports pkg_resources,
    # which was removed in setuptools 82 (Feb 2026). A bare `pip install -e .`
    # won't downgrade an already-installed setuptools 83 because the editable
    # install reuses the resolved environment, so pin it here first.
    Write-Step "pinning setuptools < 81 (pymorphy2 needs pkg_resources)"
    Invoke-Checked $Python @("-m", "pip", "install", "--quiet", "setuptools<81")

    Write-Step "installing backend dependencies"
    # No --quiet: faster-whisper/numpy/natasha pull large wheels, and a silent
    # multi-minute install looks like a hang. pip's own progress is the clearest
    # signal that something is actually downloading (parity with backend.sh).
    if ($WithDev) {
        Invoke-Checked $Python @("-m", "pip", "install", "-e", ".[dev]")
    } else {
        Invoke-Checked $Python @("-m", "pip", "install", "-e", ".")
    }

    Write-Ok "dependencies installed"
}

function Install-CudaDeps {
    param([switch]$Force)
    if ($IsMacOS -or $IsLinux) { return }
    if (-not $Force) {
        $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $NvidiaSmi) { return }
        & $NvidiaSmi *> $null
        if ($LASTEXITCODE -ne 0) { return }
    }
    if (Test-Import "nvidia") {
        Write-Ok "CUDA runtime DLLs already available (nvidia package found)"
        return
    }
    Write-Step "installing CUDA runtime DLLs (cuBLAS + cudart)"
    try {
        # No --quiet: these are multi-hundred-MB wheels; show download progress.
        & $Python -m pip install -e ".[cuda]"
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "CUDA runtime DLLs installed"
        } else {
            Write-Warn "pip install .[cuda] returned exit code $LASTEXITCODE — GPU startup may fail"
        }
    } catch {
        Write-Warn "Failed to install CUDA runtime: $_ — GPU startup may fail"
    }
}

function Ensure-YandexDeps {
    # gRPC client + SpeechKit stubs are an optional extra (pyproject [yandex]) so
    # users who never pick the Yandex provider don't pay for them.
    if (Test-Import "yandex.cloud.ai.stt.v3.stt_service_pb2_grpc") {
        Write-Ok "Yandex SpeechKit deps ready"
        return
    }
    Write-Step "installing Yandex SpeechKit dependencies"
    Invoke-Checked $Python @("-m", "pip", "install", "-e", ".[yandex]")
    Write-Ok "Yandex SpeechKit deps installed"
}

function Ensure-SpacyDeps {
    # spaCy itself is an optional extra (pyproject [spacy]); the two pipelines are
    # separate packages that pip cannot resolve, so they are fetched afterwards with
    # `spacy download`. Each is probed independently — the extra may be installed
    # while a model is still missing.
    if (-not (Test-Import "spacy")) {
        Write-Step "installing spaCy"
        Invoke-Checked $Python @("-m", "pip", "install", "-e", ".[spacy]")
    }
    foreach ($Model in @((Get-EnvValue "SPACY_MODEL_RU" "ru_core_news_md"),
                         (Get-EnvValue "SPACY_MODEL_EN" "en_core_web_md"))) {
        if (Test-Import $Model) {
            continue
        }
        Write-Step "downloading spaCy model $Model"
        Invoke-Checked $Python @("-m", "spacy", "download", $Model)
    }
    Write-Ok "spaCy deps ready"
}

function Ensure-NemotronDeps {
    # NeMo toolkit is an optional extra (pyproject [nemotron]) — multi-GB, so only
    # installed when the Nemotron streaming model is actually selected. It pulls
    # from git main (see pyproject [nemotron]): the nemotron-3.5 checkpoints need
    # classes/APIs not in any published PyPI release.
    #
    # Probe the specific module the checkpoint references, not just `nemo.collections.asr`:
    # a stale PyPI release imports fine but is missing rnnt_bpe_models_prompt, so the model
    # later fails with "Can't instantiate abstract class ASRModel". A missing module means
    # either not installed at all, or the wrong (PyPI) release — both need the git install.
    if (Test-Import "nemo.collections.asr.models.rnnt_bpe_models_prompt") {
        Write-Ok "Nemotron (NeMo) deps ready"
        return
    }
    # A bare `pip install torch` on Windows gives the CPU-only build; when the
    # backend is configured for CUDA, install the CUDA wheel explicitly first so
    # NeMo's dependency resolution keeps it.
    if ((Get-EnvValue "WHISPER_DEVICE" "cuda") -eq "cuda") {
        # Pick the CUDA wheel index by GPU compute capability: NVIDIA Blackwell GPUs
        # (RTX 50-series, compute capability >= 12.0 / sm_120) need the cu128 wheels —
        # the cu126 wheels only build kernels up to sm_90 and crash at runtime with
        # "no kernel image is available for execution on the device". Older GPUs run
        # fine on cu126. Default to cu126 when the capability can't be queried.
        $CuIndex = "cu126"
        $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($NvidiaSmi) {
            $CcRaw = (& $NvidiaSmi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
            $CcVal = 0.0
            # Parse with the invariant culture: nvidia-smi always emits "12.0" with a dot,
            # but on a non-en_US host [double]::TryParse uses the locale decimal separator
            # (a comma) and rejects the dot, silently forcing the wrong cu126 wheel.
            if ($CcRaw -and [double]::TryParse($CcRaw, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$CcVal) -and $CcVal -ge 12.0) {
                $CuIndex = "cu128"
            }
        }
        Write-Step "installing CUDA build of PyTorch (~2.5 GB; download progress below)"
        Write-Host "    torch CUDA wheel: $CuIndex"
        # No --quiet: the CUDA torch wheel is ~2.5 GB; without pip's progress bar a
        # multi-minute download looks like a hang.
        Invoke-Checked $Python @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/$CuIndex")
    }
    Write-Step "installing NeMo toolkit from git main (this can take several minutes; download progress below)"
    # No --quiet: NeMo pulls torch/numpy/etc. multi-GB wheels; show download progress.
    # The [nemotron] extra pins nemo_toolkit to NVIDIA/NeMo@main in pyproject.
    Invoke-Checked $Python @("-m", "pip", "install", "-e", ".[nemotron]")
    # NeMo pins protobuf ~=5.29 but runs fine on 6.x, while the yandexcloud stubs
    # REQUIRE >=6.31 — restore the newer runtime so both providers keep working.
    if (Test-Import "yandexcloud") {
        Invoke-Checked $Python @("-m", "pip", "install", "--quiet", "protobuf>=6.31,<7")
    }
    Write-Ok "Nemotron (NeMo) deps installed"
}

function Ensure-Dependencies {
    # pkg_resources (setuptools) is included because natasha → pymorphy2 imports
    # it, and on Python ≥3.12 setuptools is no longer bundled with the
    # interpreter. Without it an existing venv passes the other checks but Natasha
    # crashes at startup with ModuleNotFoundError: pkg_resources.
    $Needed = @("fastapi", "uvicorn", "pydantic_settings", "aiosqlite", "pkg_resources")
    foreach ($Module in $Needed) {
        if (-not (Test-Import $Module)) {
            Install-Dependencies -WithDev:$Dev
            return
        }
    }

    # The project package itself must be importable. An editable install
    # (pip install -e .) records the package's *path* at install time, so if the
    # project was later moved/copied — or the venv was created elsewhere and this
    # dir was never installed editable — the third-party libs above are present but
    # `app` is not. Reinstall editable to (re)point it at the current location.
    # Without this, preload_models.py (run as `python scripts/...`) fails with
    # "No module named 'app'" and crashes the startup right after the model pick.
    if (-not (Test-Import "app")) {
        Write-Warn "project package 'app' is not importable in this venv; reinstalling editable"
        Install-Dependencies -WithDev:$Dev
        return
    }
    Write-Ok "dependencies ready"
}

function Test-OllamaModel {
    param([string]$Model)

    $BaseUrl = Get-EnvValue "OLLAMA_BASE_URL" "http://127.0.0.1:11434"
    Write-Step "checking Ollama models at $BaseUrl"
    $Response = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/tags" -TimeoutSec 5
    foreach ($Item in $Response.models) {
        if ($Item.name -eq $Model -or $Item.model -eq $Model) {
            return $true
        }
    }

    return $false
}

function Wait-OllamaServer {
    param([string]$Ollama)

    $BaseUrl = Get-EnvValue "OLLAMA_BASE_URL" "http://127.0.0.1:11434"
    Write-Step "checking Ollama server $BaseUrl"
    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        try {
            Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/tags" -TimeoutSec 5 | Out-Null
            Write-Ok "Ollama server responding"
            return
        } catch {
            if ($Attempt -eq 1) {
                Write-Warn "Ollama server is not responding; starting ollama serve"
                Start-Process -FilePath $Ollama -ArgumentList "serve" -WindowStyle Hidden
            } else {
                Write-Warn "waiting for Ollama server ($Attempt/3)"
            }
            Start-Sleep -Seconds 3
        }
    }

    throw "Ollama server did not respond at $BaseUrl. Open Ollama or run: ollama serve"
}

function Warm-OllamaModel {
    param([string]$Model)

    $BaseUrl = Get-EnvValue "OLLAMA_BASE_URL" "http://127.0.0.1:11434"
    $KeepAlive = Get-EnvValue "OLLAMA_KEEP_ALIVE" "30m"
    $NumCtx = [int](Get-EnvValue "OLLAMA_NUM_CTX" "1024")
    $NumBatch = [int](Get-EnvValue "OLLAMA_NUM_BATCH" "1024")
    $NumGpu = [int](Get-EnvValue "OLLAMA_NUM_GPU" "999")
    $MainGpu = [int](Get-EnvValue "OLLAMA_MAIN_GPU" "0")
    $Think = (Get-EnvValue "OLLAMA_THINK" "false").ToLowerInvariant() -in @("1", "true", "yes", "on")
    $Body = @{
        model = $Model
        prompt = "/no_think`nready"
        stream = $false
        keep_alive = $KeepAlive
        think = $Think
        options = @{
            num_predict = 1
            temperature = 0
            num_ctx = $NumCtx
            num_batch = $NumBatch
            num_gpu = $NumGpu
            main_gpu = $MainGpu
        }
    } | ConvertTo-Json -Depth 5

    Write-Step "warming Ollama model $Model on GPU; this can take a while"
    Invoke-RestMethod `
        -Method Post `
        -Uri "$BaseUrl/api/generate" `
        -ContentType "application/json" `
        -Body $Body `
        -TimeoutSec 600 | Out-Null
    Write-Ok "Ollama model warmed; keep_alive=$KeepAlive"
}

function Check-Ollama {
    if (-not (Test-EnvEnabled "LLM_ENABLED")) {
        Write-Warn "LLM is disabled in .env"
        return
    }

    $Ollama = Find-Ollama
    if (-not $Ollama) {
        if (-not $SkipOllamaInstall) {
            $Winget = Get-Command winget -ErrorAction SilentlyContinue
            if (-not $Winget) {
                Write-Warn "winget not found; cannot install Ollama automatically"
                return
            }

            Write-Step "installing Ollama via winget"
            Invoke-Checked "winget" @(
                "install",
                "-e",
                "--id",
                "Ollama.Ollama",
                "--scope",
                "user",
                "--accept-package-agreements",
                "--accept-source-agreements"
            )
            $Ollama = Find-Ollama
            if (-not $Ollama) {
                Write-Warn "Ollama installed but not visible in current shell. Restart terminal."
                return
            }
        } else {
            Write-Warn "Ollama not found. Backend will run; explain endpoint may return LLM_UNAVAILABLE."
            return
        }
    }

    Write-Ok "Ollama found"
    Wait-OllamaServer $Ollama
    $Model = Get-EnvValue "OLLAMA_MODEL" "qwen3:8b"
    if ((Test-OllamaModel $Model) -and -not $PullLlm) {
        Write-Ok "Ollama model exists: $Model"
    } else {
        Write-Step "pulling Ollama model $Model; download progress is shown below"
        Invoke-Checked $Ollama @("pull", $Model)
        Write-Ok "Ollama model ready: $Model"
    }

    Warm-OllamaModel $Model
}

function Ensure-WhisperModel {
    Write-Step "preloading Whisper model"
    Invoke-Checked $Python @("scripts\preload_models.py")
}

function Show-ModelMenu {
    # Renders a numbered menu of model options, reads the user's choice in a loop
    # (re-asking on invalid input), and returns the chosen model name. If the value
    # already in .env ($CurrentValue) is not one of the listed options, an extra
    # "[0] Оставить текущую" entry is appended so existing config is never silently
    # clobbered. If $CurrentValue matches a listed option, that option is the default
    # (shown in the prompt and selected on empty input); otherwise the first option is.
    param(
        [string]$Title,
        [array]$Options,
        [string]$CurrentValue = ""
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Доступные модели:"

    # Widest label drives the column alignment.
    $LabelWidth = ($Options | ForEach-Object { $_.label.Length } | Measure-Object -Maximum).Maximum
    if (-not $LabelWidth) { $LabelWidth = 0 }

    $CurrentNorm = $CurrentValue.Trim().ToLowerInvariant()
    $DefaultIndex = 0
    $HasKeepCurrent = $false

    # If the current value is outside the curated list, offer to keep it.
    $InList = $false
    for ($I = 0; $I -lt $Options.Count; $I++) {
        if ($Options[$I].name.Trim().ToLowerInvariant() -eq $CurrentNorm) {
            $InList = $true
            $DefaultIndex = $I
            break
        }
    }
    if ($CurrentNorm -and -not $InList) {
        $HasKeepCurrent = $true
    }

    for ($I = 0; $I -lt $Options.Count; $I++) {
        $Opt = $Options[$I]
        $Padded = $Opt.label.PadRight($LabelWidth)
        $Line = "  [{0}] {1}   {2}" -f ($I + 1), $Padded, $Opt.size
        if ($Opt.note) { $Line += "   • {0}" -f $Opt.note }
        Write-Host $Line
    }
    if ($HasKeepCurrent) {
        Write-Host ("  [0] Оставить текущую: {0}" -f $CurrentValue)
    }

    # Default selection shown in the prompt: 1 for a listed option, 0 for "keep current".
    $DefaultChoice = if ($HasKeepCurrent) { 0 } else { $DefaultIndex + 1 }
    $MaxChoice = $Options.Count

    while ($true) {
        $Prompt = "Введите номер [{0}]: " -f $DefaultChoice
        $Input = Read-Host $Prompt
        $Trimmed = "$Input".Trim()

        # Empty input accepts the default.
        if (-not $Trimmed) {
            if ($HasKeepCurrent -and $DefaultChoice -eq 0) {
                return $CurrentValue
            }
            return $Options[$DefaultIndex].name
        }

        if ($Trimmed -notmatch '^\d+$') {
            Write-Warn "Нужно число от 0 до $MaxChoice"
            continue
        }

        $Num = [int]$Trimmed
        if ($Num -eq 0 -and $HasKeepCurrent) {
            return $CurrentValue
        }
        if ($Num -ge 1 -and $Num -le $MaxChoice) {
            return $Options[$Num - 1].name
        }
        Write-Warn "Нужно число от 0 до $MaxChoice"
    }
}

function Read-SecretWithKeep {
    # Prompt for an API key. If $Current is non-empty, offer to keep it (Enter) or
    # re-enter. Never echoes the stored value — only a masked hint. Returns the
    # chosen value (the existing one on keep, or the typed one).
    param([string]$Prompt, [string]$Current = "")

    if ($Current) {
        $Mask = $Current.Substring(0, [Math]::Min(4, $Current.Length)) + ("*" * [Math]::Max(0, $Current.Length - 4))
        $Entered = Read-Host "$Prompt [Enter = оставить текущий ($Mask)]"
        if (-not $Entered) {
            return $Current
        }
        return $Entered
    }
    return Read-Host $Prompt
}

function Show-LlmApiProviderMenu {
    # Returns the chosen $LlmProviders entry (hashtable). Default index is the
    # entry whose key matches $CurrentValue, else 1.
    param([string]$CurrentValue = "")
    Write-Host "LLM-провайдер:"
    $LabelWidth = ($LlmProviders | ForEach-Object { $_.label.Length } | Measure-Object -Maximum).Maximum
    $DefaultIndex = 0
    for ($I = 0; $I -lt $LlmProviders.Count; $I++) {
        $P = $LlmProviders[$I]
        $Padded = $P.label.PadRight($LabelWidth)
        Write-Host ("  [{0}] {1}   {2}" -f ($I + 1), $Padded, $P.default_model)
        if ($P.key -eq $CurrentValue) { $DefaultIndex = $I }
    }
    while ($true) {
        $Input = Read-Host ("LLM-провайдер [$($DefaultIndex + 1)]")
        $Trimmed = "$Input".Trim()

        if (-not $Trimmed) {
            return $LlmProviders[$DefaultIndex]
        }

        if ($Trimmed -notmatch '^\d+$') {
            Write-Warn "Нужно число от 1 до $($LlmProviders.Count)"
            continue
        }

        $Num = [int]$Trimmed
        if ($Num -ge 1 -and $Num -le $LlmProviders.Count) {
            return $LlmProviders[$Num - 1]
        }

        Write-Warn "Нужно число от 1 до $($LlmProviders.Count)"
    }
}

function Select-Models {
    # Interactive model/mode picker run before the backend starts.
    #
    # For each family (LLM, Whisper) the user is FIRST asked local-vs-API, then the
    # relevant submenu right away (no second pass):
    #   LLM     local → qwen model menu   | api → provider menu + key + model
    #   Whisper local → whisper size menu | api → key prompt
    # Writes all choices into .env in one place. Skipped with -SkipModelSelect.
    if ($SkipModelSelect) {
        Write-Ok "model selection skipped (-SkipModelSelect)"
        return
    }

    Write-Host ""
    Write-Host "==> Выбор моделей" -ForegroundColor Cyan

    if (Test-EnvEnabled "LLM_ENABLED") {
        $CurrentLlmProvider = Get-EnvValue "LLM_PROVIDER" "ollama"
        Write-Host ""
        Write-Host "LLM — как запускать:"
        Write-Host "  [1] Локально (Ollama)"
        Write-Host "  [2] Через API (ключ)"
        $LlmModeDefault = if ($CurrentLlmProvider -ne "ollama" -and $CurrentLlmProvider -ne "") { 2 } else { 1 }
        while ($true) {
            $Input = Read-Host ("Режим LLM [$LlmModeDefault]")
            $Trimmed = "$Input".Trim()
            $LlmModeNum = if (-not $Trimmed) { $LlmModeDefault } else { 0 }
            if (-not $Trimmed -or ($Trimmed -match '^[12]$')) {
                if ($Trimmed) { $LlmModeNum = [int]$Trimmed }
                break
            }
            Write-Warn "Нужно 1 или 2"
        }

        if ($LlmModeNum -eq 1) {
            Set-EnvValue "LLM_PROVIDER" "ollama"
            $CurrentLlm = Get-EnvValue "OLLAMA_MODEL" ""
            $Choice = Show-ModelMenu -Title "Выбор LLM-модели (Ollama)" -Options $LlmModels -CurrentValue $CurrentLlm
            if ($Choice -and $Choice.Trim().ToLowerInvariant() -ne $CurrentLlm.Trim().ToLowerInvariant()) {
                Set-EnvValue "OLLAMA_MODEL" $Choice
                Write-Ok "OLLAMA_MODEL=$Choice"
            } else {
                Write-Ok "LLM-модель без изменений: $CurrentLlm"
            }
        } else {
            $Chosen = Show-LlmApiProviderMenu -CurrentValue $CurrentLlmProvider
            Set-EnvValue "LLM_PROVIDER" $Chosen.key
            Write-Ok "LLM_PROVIDER=$($Chosen.key)"
            $Key = Read-SecretWithKeep -Prompt "LLM API key" -Current (Get-EnvValue "LLM_API_KEY" "")
            Set-EnvValue "LLM_API_KEY" $Key
            $CurModel = Get-EnvValue "LLM_API_MODEL" ""
            $ModelInput = Read-Host "Модель [Enter = $($Chosen.default_model)]"
            $ModelVal = if ($ModelInput) { $ModelInput } else { $CurModel }
            Set-EnvValue "LLM_API_MODEL" $ModelVal
            Write-Ok "LLM_API_MODEL=$ModelVal"
        }
    } else {
        Write-Warn "LLM disabled in .env — пропуск выбора LLM-модели"
    }

    if (Test-EnvEnabled "WHISPER_ENABLED") {
        $CurrentWhisperProvider = Get-EnvValue "WHISPER_PROVIDER" "local"
        Write-Host ""
        Write-Host "Whisper — как запускать:"
        Write-Host "  [1] Локально (faster-whisper)"
        Write-Host "  [2] Через API (ключ)"
        Write-Host "  [3] Yandex SpeechKit (реальное время)"
        $WhisperModeDefault = switch ($CurrentWhisperProvider) {
            "api" { 2 }
            "yandex" { 3 }
            default { 1 }
        }
        while ($true) {
            $Input = Read-Host ("Режим Whisper [$WhisperModeDefault]")
            $Trimmed = "$Input".Trim()
            $WhisperModeNum = if (-not $Trimmed) { $WhisperModeDefault } else { 0 }
            if (-not $Trimmed -or ($Trimmed -match '^[123]$')) {
                if ($Trimmed) { $WhisperModeNum = [int]$Trimmed }
                break
            }
            Write-Warn "Нужно 1, 2 или 3"
        }

        if ($WhisperModeNum -eq 1) {
            Set-EnvValue "WHISPER_PROVIDER" "local"
            $CurrentWhisper = Get-EnvValue "WHISPER_MODEL" ""
            $Choice = Show-ModelMenu -Title "Выбор локальной модели распознавания" -Options $WhisperModels -CurrentValue $CurrentWhisper
            if ($Choice -and $Choice.Trim().ToLowerInvariant() -ne $CurrentWhisper.Trim().ToLowerInvariant()) {
                Set-EnvValue "WHISPER_MODEL" $Choice
                Write-Ok "WHISPER_MODEL=$Choice"
            } else {
                Write-Ok "Модель распознавания без изменений: $CurrentWhisper"
            }
            if ((Get-EnvValue "WHISPER_MODEL" "").Trim().ToLowerInvariant().StartsWith("nemotron")) {
                Ensure-NemotronDeps
            }
        } elseif ($WhisperModeNum -eq 2) {
            Set-EnvValue "WHISPER_PROVIDER" "api"
            Write-Ok "WHISPER_PROVIDER=api"
            $Key = Read-SecretWithKeep -Prompt "Whisper API key" -Current (Get-EnvValue "WHISPER_API_KEY" "")
            Set-EnvValue "WHISPER_API_KEY" $Key
        } else {
            Set-EnvValue "WHISPER_PROVIDER" "yandex"
            Write-Ok "WHISPER_PROVIDER=yandex"
            $Key = Read-SecretWithKeep -Prompt "Yandex SpeechKit API key" -Current (Get-EnvValue "YANDEX_STT_API_KEY" "")
            Set-EnvValue "YANDEX_STT_API_KEY" $Key
            Ensure-YandexDeps
        }

        # ASR_LANGUAGE steers every provider (Whisper forces it, Yandex restricts to
        # it, Nemotron prompts it), so ask once here regardless of the mode above.
        $CurrentAsrLanguage = (Get-EnvValue "ASR_LANGUAGE" "ru").Trim().ToLowerInvariant()
        Write-Host ""
        Write-Host "Язык распознавания речи:"
        Write-Host "  [1] Русский"
        Write-Host "  [2] English"
        $LangDefault = if ($CurrentAsrLanguage -eq "en") { 2 } else { 1 }
        while ($true) {
            $Input = Read-Host ("Язык распознавания [$LangDefault]")
            $Trimmed = "$Input".Trim()
            $LangNum = if (-not $Trimmed) { $LangDefault } else { 0 }
            if (-not $Trimmed -or ($Trimmed -match '^[12]$')) {
                if ($Trimmed) { $LangNum = [int]$Trimmed }
                break
            }
            Write-Warn "Нужно 1 или 2"
        }
        $ChosenLang = if ($LangNum -eq 2) { "en" } else { "ru" }
        Set-EnvValue "ASR_LANGUAGE" $ChosenLang
        Write-Ok "ASR_LANGUAGE=$ChosenLang"
    } else {
        Write-Warn "WHISPER disabled in .env — пропуск выбора Whisper-модели"
    }

    # Morphology backend for term extraction. Independent of the ASR choice: it runs
    # on the recognized text, so it is asked even when Whisper is disabled.
    $CurrentNlpBackend = (Get-EnvValue "NLP_BACKEND" "natasha").Trim().ToLowerInvariant()
    Write-Host ""
    Write-Host "Выделение терминов — движок морфологии:"
    Write-Host "  [1] Natasha (только русский, быстрее: ~1.5 мс)"
    Write-Host "  [2] spaCy (русский + английский, точнее: ~4 мс)"
    $NlpDefault = if ($CurrentNlpBackend -eq "spacy") { 2 } else { 1 }
    while ($true) {
        $Input = Read-Host ("Движок терминов [$NlpDefault]")
        $Trimmed = "$Input".Trim()
        $NlpNum = if (-not $Trimmed) { $NlpDefault } else { 0 }
        if (-not $Trimmed -or ($Trimmed -match '^[12]$')) {
            if ($Trimmed) { $NlpNum = [int]$Trimmed }
            break
        }
        Write-Warn "Нужно 1 или 2"
    }
    if ($NlpNum -eq 2) {
        Set-EnvValue "NLP_BACKEND" "spacy"
        Write-Ok "NLP_BACKEND=spacy"
        Ensure-SpacyDeps
    } else {
        Set-EnvValue "NLP_BACKEND" "natasha"
        Write-Ok "NLP_BACKEND=natasha"
    }

    Write-Host ""
}

function Run-Backend {
    Ensure-Venv
    Ensure-ProjectFiles
    Ensure-GpuSettings
    Install-CudaDeps
    Ensure-Dependencies
    Select-Models
    $LlmProvider = Get-EnvValue "LLM_PROVIDER"
    if ($LlmProvider -eq "ollama" -or -not $LlmProvider) {
        Check-Ollama
    }
    Ensure-WhisperModel

    Write-Step "starting FastAPI http://$HostAddress`:$Port"
    Invoke-Checked $Python @(
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        $HostAddress,
        "--port",
        "$Port",
        "--reload"
    )
}

function Run-Tests {
    Ensure-Venv
    Ensure-ProjectFiles
    Ensure-Dependencies
    Invoke-Checked $Python @("-m", "pytest")
}

function Run-Lint {
    Ensure-Venv
    Ensure-ProjectFiles
    Ensure-Dependencies
    Invoke-Checked $Python @("-m", "ruff", "check", ".")
    Invoke-Checked $Python @("-m", "mypy", "app")
}

function Run-Format {
    Ensure-Venv
    Ensure-ProjectFiles
    Ensure-Dependencies
    Invoke-Checked $Python @("-m", "black", ".")
}

Set-Location $Root

switch ($Command) {
    "install" {
        Ensure-Venv
        Ensure-ProjectFiles
        Install-Dependencies -WithDev:$true
        Check-Ollama
    }
    "run" { Run-Backend }
    "test" { Run-Tests }
    "lint" { Run-Lint }
    "format" { Run-Format }
}
