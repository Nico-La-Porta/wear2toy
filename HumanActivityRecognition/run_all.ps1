Set-Location -Path $PSScriptRoot

Write-Host "Trovati questi file Python nella cartella:"

# Prende tutti i file .py che matchano la struttura indicata
$files = Get-ChildItem -Filter "*.py" | Where-Object {
    $_.Name -match '^T[^_]+_PTN[^_]+_PTA[^_]+_PTWS[^_]+_FTN[^_]+_FTA[^_]+_FTWS[^_]+_S[^_]+\.py$'
}

if ($files.Count -eq 0) {
    Write-Host "Nessun file corrisponde al pattern specificato."
    exit
}

foreach ($file in $files) {
    Write-Host "`n Eseguendo: $($file.Name)"
    $output = python $file.FullName 2>&1
    Write-Host $output
    Write-Host "Exit code: $LASTEXITCODE"
}
