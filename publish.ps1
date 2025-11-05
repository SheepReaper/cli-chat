#!/usr/bin/env pwsh

$rids = @("linux-x64", "win-x64")

foreach ($rid in $rids) {
    dotnet publish ./ChatConsole/ChatConsole.csproj -c Release -r $rid -o "publish/$rid" -p:DebugSymbols=false
}

foreach ($rid in $rids) {
    Compress-Archive -Path "publish/$rid/*", "README.md" -DestinationPath "publish/chatconsole-$rid.zip" -Force
}
