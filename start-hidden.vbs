' Blink Video Downloader - Hidden Startup Script
' Double-click to run the server in the background

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
strPath = FSO.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = strPath

' Kill any existing instances first (silently)
WshShell.Run "net stop BlinkDownloader", 0, True
WshShell.Run "taskkill /F /IM pythonw.exe", 0, True

' Kill anything on port 8080
WshShell.Run "cmd /c for /f ""tokens=5"" %a in ('netstat -ano ^| findstr "":8080""') do taskkill /F /PID %a", 0, True

' Wait for port to be released
WScript.Sleep 2000

' Check for venv
If FSO.FolderExists(strPath & "\venv") Then
    pythonPath = strPath & "\venv\Scripts\pythonw.exe"
Else
    pythonPath = "pythonw"
End If

' Run app.py hidden (pythonw has no console window)
WshShell.Run Chr(34) & pythonPath & Chr(34) & " -m uvicorn app:app --host 0.0.0.0 --port 8080", 0, False
