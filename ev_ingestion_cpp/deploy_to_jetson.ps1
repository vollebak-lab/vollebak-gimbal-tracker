param (
    [Parameter(Mandatory=$true)]
    [string]$JetsonIp,

    [Parameter(Mandatory=$true)]
    [string]$JetsonUser,

    [Parameter(Mandatory=$false)]
    [string]$RemoteDir = "~/evpropnet_workspace"
)

$LocalDir = $PSScriptRoot

Write-Host "Deploying C++ Ingestion project to $JetsonUser@$JetsonIp : $RemoteDir" -ForegroundColor Cyan

# 1. Ensure remote directory exists
ssh "${JetsonUser}@${JetsonIp}" "mkdir -p $RemoteDir/ev_ingestion_cpp"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to connect via SSH or create directory." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Secure Copy the files to the Jetson Nano
Write-Host "Copying files..."
scp -r "$LocalDir\main.cpp" "$LocalDir\CMakeLists.txt" "${JetsonUser}@${JetsonIp}:${RemoteDir}/ev_ingestion_cpp/"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment Successful." -ForegroundColor Green
    Write-Host ""
    Write-Host "To compile on the Jetson, SSH into it and run:" -ForegroundColor Yellow
    Write-Host "  cd $RemoteDir/ev_ingestion_cpp"
    Write-Host "  mkdir build && cd build"
    Write-Host "  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .."
    Write-Host "  make -j8"
} else {
    Write-Host "File transfer failed." -ForegroundColor Red
}
