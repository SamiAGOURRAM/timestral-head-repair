# Test B (block size): wait for the GPU, build lags 16/24, then score lags 16, 24, 32 and the
# published model with blocks of 16 and 24 future tokens on the lag sweep's seven configurations.
param([string]$WaitFor)
if ($WaitFor) { while (-not (Select-String -Path $WaitFor -Pattern 'exit:' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 30 } }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
.venv\Scripts\python.exe -u experiments\edit_5m\block_size_test.py *> results\edit_5m\block_size_build.log
"build exit: $LASTEXITCODE"
$only = 'solar/H/long,solar/H/medium,electricity/H/long,m_dense/H/long,ett1/H/long,loop_seattle/H/long,us_births/D/short'
foreach ($chunk in 16, 24) {
    .venv\Scripts\python.exe evaluation\gift_eval_run.py --size 5M --chunk $chunk --only $only *> results\edit_5m\block_base_chunk$chunk.log
    "base chunk $chunk exit: $LASTEXITCODE at $(Get-Date -Format HH:mm)"
    foreach ($lag in 16, 24, 32) {
        .venv\Scripts\python.exe evaluation\gift_eval_run.py --checkpoint checkpoints\lagsweep_$lag --chunk $chunk --name 5M-lag$lag-chunk$chunk --only $only --compare-with Timestral-5M *> results\edit_5m\block_lag${lag}_chunk$chunk.log
        "lag $lag chunk $chunk exit: $LASTEXITCODE at $(Get-Date -Format HH:mm)"
    }
}
"block test done"
