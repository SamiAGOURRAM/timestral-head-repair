# Lag sweep on head 5.1: build the checkpoints, then GIFT-Eval each on seven hourly and
# daily configurations that blind runs 3-5 damage. Run from the project root.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
.venv\Scripts\python.exe -u experiments\edit_5m\lag_sweep.py *> results\edit_5m\lag_sweep.log
"build exit: $LASTEXITCODE"
$only = 'solar/H/long,solar/H/medium,electricity/H/long,m_dense/H/long,ett1/H/long,loop_seattle/H/long,us_births/D/short'
foreach ($lag in 11, 21, 28, 30, 31, 32, 33, 34, 36, 40, 42, 43) {
    .venv\Scripts\python.exe evaluation\gift_eval_run.py --checkpoint checkpoints\lagsweep_$lag --name 5M-lagsweep-$lag --only $only --compare-with Timestral-5M *> results\edit_5m\lagsweep_gift_$lag.log
    "lag $lag gift exit: $LASTEXITCODE at $(Get-Date -Format HH:mm)"
}
