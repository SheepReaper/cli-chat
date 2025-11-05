# Create a zip archive of the source code, respecting .gitignore

$SourceDir = Get-Location
$OutputDir = Join-Path $SourceDir "out"
$ArchiveName = "cli-chat-source.zip"
$ArchivePath = Join-Path $OutputDir $ArchiveName

# Ensure the output directory exists
if (!(Test-Path -Path $OutputDir -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputDir
}

# Use 7-Zip to create the archive, respecting .gitignore
& "C:\Program Files\7-Zip\7z.exe" a -tzip -xr!"*bin/" -xr!"obj/" -xr!"out/" -xr!"publish/" $ArchivePath $SourceDir

Write-Host "Source archive created at: $ArchivePath"
