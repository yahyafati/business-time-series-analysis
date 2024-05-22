# Check if Chocolatey is installed
Write-Host "Checking for Chocolatey..."
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force; 
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; 
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
} else {
    Write-Host "Chocolatey is already installed."
}

# Install Python using Chocolatey
Write-Host "Installing Python..."
choco install -y python

# Create a virtual environment in the current directory
Write-Host "Creating virtual environment..."
python -m venv .

# Activate the virtual environment
Write-Host "Activating virtual environment..."
& ./Scripts/Activate.ps1

# Install requirements from requirements.txt
if (Test-Path -Path "./requirements.txt") {
    Write-Host "Installing dependencies..."
    pip install -r requirements.txt
} else {
    Write-Host "requirements.txt not found."
    exit 1
}

# Keep the PowerShell window open
Read-Host "Press Enter to continue..."




