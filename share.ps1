# Share the app over the internet from this PC through a free Cloudflare quick tunnel.
#
#   powershell -ExecutionPolicy Bypass -File share.ps1
#
# Prints a public https://....trycloudflare.com link. The link works only while
# this window stays open, and changes every time you start the script.
# Press Ctrl+C to stop sharing.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$port = 8000

# The link is public, so refuse to run without a password.
$password = $env:APP_PASSWORD
if (-not $password -and (Test-Path .env)) {
    $line = Get-Content .env | Where-Object { $_ -match '^\s*APP_PASSWORD\s*=' } | Select-Object -First 1
    if ($line) { $password = ($line -replace '^\s*APP_PASSWORD\s*=', '').Trim() }
}
if (-not $password) {
    Write-Host "APP_PASSWORD is empty. Add a line like  APP_PASSWORD=your-password  to .env first:" -ForegroundColor Red
    Write-Host "anyone with the link could otherwise read your PDFs and use your Gemini key." -ForegroundColor Red
    exit 1
}

$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) { $cloudflared = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe" }
if (-not (Test-Path $cloudflared)) {
    Write-Host "cloudflared not found. Install it with:  winget install --id Cloudflare.cloudflared" -ForegroundColor Red
    exit 1
}

$log = Join-Path $env:TEMP "doc-qa-tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue

Write-Host "Starting the app on http://127.0.0.1:$port ..."
$server = Start-Process -FilePath ".venv\Scripts\python.exe" -NoNewWindow -PassThru `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$port"
$tunnel = $null
try {
    $ready = $false
    for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
        Start-Sleep -Seconds 1
        if ($server.HasExited) { throw "The app failed to start (is port $port already in use?)." }
        try { $ready = (Invoke-WebRequest "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
    }
    if (-not $ready) { throw "The app did not start within 60 seconds." }

    Write-Host "Opening the Cloudflare tunnel ..."
    $tunnel = Start-Process -FilePath $cloudflared -NoNewWindow -PassThru -RedirectStandardError $log `
        -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port"

    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        if ($tunnel.HasExited) { throw "The tunnel stopped. Details: $log" }
        $match = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) { $url = $match.Matches[0].Value }
    }
    if (-not $url) { throw "No link received from Cloudflare. Details: $log" }

    Write-Host ""
    Write-Host "  Your app is online at:  $url" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Open it on your phone or laptop, log in with APP_PASSWORD, then tap 'Install app'."
    Write-Host "  The link can take up to a minute to start working. Keep this window open; Ctrl+C stops sharing."
    while (-not $tunnel.HasExited -and -not $server.HasExited) { Start-Sleep -Seconds 2 }
    if ($server.HasExited) { Write-Host "The app stopped." -ForegroundColor Red }
    if ($tunnel.HasExited) { Write-Host "The tunnel stopped. Details: $log" -ForegroundColor Red }
}
finally {
    # /T also stops the real Python process started by the venv launcher.
    foreach ($p in @($tunnel, $server)) {
        if ($p -and -not $p.HasExited) { taskkill /PID $p.Id /T /F *> $null }
    }
    Write-Host "Stopped sharing."
}
