$delay = 180

Write-Host "Starting continuous API polling loop for SCP..."

while ($true) {
    Write-Host "Attempting to SCP script to VM..."
    $out = gcloud compute scp launch_showdown.sh instance-20260823-164347:/home/DCCS5/cerebellum_project/launch_showdown.sh --zone=us-east1-c 2>&1
    
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
    $out = gcloud compute ssh instance-20260823-164347 --zone=us-east1-c --command="chmod +x /home/DCCS5/cerebellum_project/launch_showdown.sh && nohup /home/DCCS5/cerebellum_project/launch_showdown.sh > /dev/null 2>&1 &" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully launched the showdown script!"
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
    $log = gcloud compute ssh instance-20260823-164347 --zone=us-east1-c --command="cat /home/DCCS5/cerebellum_project/showdown_N30_final.out | tail -n 20" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "--- LOG OUTPUT ---\n"
        Write-Host $log
        if ($log -match "Done!") {
            Write-Host "Showdown complete!"
            break
        }
    }
}
