$delay = 60

Write-Host "Now entering monitoring phase for Baseline V2..."
while ($true) {
    Start-Sleep -Seconds $delay
    Write-Host "Fetching log tail..."
    $log = gcloud compute ssh instance-20260823-164347 --zone=us-east1-c --command="cat /home/DCCS5/cerebellum_project/showdown_N30_metrics.out | tail -n 40" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "--- LOG OUTPUT ---\n"
        Write-Host $log
        if ($log -match "Showdown complete!") {
            Write-Host "Showdown complete!"
            break
        }
    } else {
        Write-Host "API timed out or failed. Retrying..."
    }
}
