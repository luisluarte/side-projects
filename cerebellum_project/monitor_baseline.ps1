$delay = 180

Write-Host "Starting continuous API polling loop for SCP..."

while ($true) {
    Write-Host "Attempting to SCP script to VM..."
    $out = gcloud compute scp run_baseline_v2_n30.R instance-20260823-164347:/home/DCCS5/cerebellum_project/run_baseline_v2_n30.R --zone=us-east1-c 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully uploaded the script!"
        break
    } else {
        Write-Host "API timed out. Retrying in 3 minutes..."
        Start-Sleep -Seconds $delay
    }
}

while ($true) {
    Write-Host "Attempting to execute script on VM..."
    $out = gcloud compute ssh instance-20260823-164347 --zone=us-east1-c --command="nohup Rscript /home/DCCS5/cerebellum_project/run_baseline_v2_n30.R > /home/DCCS5/cerebellum_project/showdown_N30_metrics.out 2>&1 &" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully launched the metrics script!"
        break
    } else {
        Write-Host "API timed out. Retrying in 3 minutes..."
        Start-Sleep -Seconds $delay
    }
}

Write-Host "Now entering monitoring phase..."
while ($true) {
    Start-Sleep -Seconds $delay
    Write-Host "Fetching log tail..."
    $log = gcloud compute ssh instance-20260823-164347 --zone=us-east1-c --command="cat /home/DCCS5/cerebellum_project/showdown_N30_metrics.out | tail -n 30" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "--- LOG OUTPUT ---\n"
        Write-Host $log
        if ($log -match "Showdown complete!") {
            Write-Host "Showdown complete!"
            break
        }
    }
}
