# run_live.ps1 — 带日志的启动脚本
# 用法: 修改下面的 Key 后, 在 PowerShell 中执行:  .\run_live.ps1
# 可加参数: .\run_live.ps1 -Limit 2000 -Concurrency 40
param(
    [int]$Limit = 2000,
    [int]$Concurrency = 100
)

# ── 填入你的 Key (注意: 不要有多余的反引号/空格) ──
$env:DEEPSEEK_API_KEY_RELATION="sk-24c7da9e1f8849f699cf8b0c65fb4ed0"
$env:DEEPSEEK_API_KEY_EXTRACTION="sk-d4174217b79d44faa3ed02a9b89edc6a"
$env:DEEPSEEK_API_KEY_CLASSIFICATION="sk-b03764ef95de41ac8dd11d5f6b3fa3b7"
$env:DEEPSEEK_API_KEY_REVIEWER="sk-32409b82093e4659a9c6ac2295f1429a"
$env:DEEPSEEK_API_KEY_VERIFICATION="sk-4fb9596e9f8241069dcf6ae3abfbaf1e"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:LLM_MODEL="deepseek-chat"

Set-Location $PSScriptRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "output\run_$ts.log"
Write-Host "日志: $log  |  Limit=$Limit Concurrency=$Concurrency"
Write-Host "Key检查: RELATION=$([bool]$env:DEEPSEEK_API_KEY_RELATION) EXTRACTION=$([bool]$env:DEEPSEEK_API_KEY_EXTRACTION) CLASSIFICATION=$([bool]$env:DEEPSEEK_API_KEY_CLASSIFICATION) REVIEWER=$([bool]$env:DEEPSEEK_API_KEY_REVIEWER) VERIFICATION=$([bool]$env:DEEPSEEK_API_KEY_VERIFICATION)"

python -u run.py --live --limit $Limit --concurrency $Concurrency 2>&1 | Tee-Object -FilePath $log
Write-Host "退出码: $LASTEXITCODE (日志已保存到 $log)"
