$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Extract-DocxXml($path) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
    try {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq 'word/document.xml' }
        $reader = New-Object System.IO.StreamReader($entry.Open(), [Text.Encoding]::UTF8)
        $xml = $reader.ReadToEnd()
        $reader.Close()
        return $xml
    } finally {
        $zip.Dispose()
    }
}

function XmlToText($xml) {
    $text = $xml -replace '</w:p>', "`n"
    $text = $text -replace '<[^>]+>', ''
    $text = $text -replace '&amp;', '&' -replace '&lt;', '<' -replace '&gt;', '>' -replace '&quot;', '"' -replace '&apos;', "'"
    return $text
}

$files = Get-ChildItem -Path '.' | Where-Object { $_.Extension -eq '.docx' }
foreach ($f in $files) {
    Write-Output ("===== FILE: " + $f.Name + " =====")
    $xml = Extract-DocxXml $f.FullName
    Write-Output (XmlToText $xml)
    Write-Output ""
}

# .doc files via Word COM
$docFiles = Get-ChildItem -Path '.' | Where-Object { $_.Extension -eq '.doc' }
foreach ($f in $docFiles) {
    Write-Output ("===== FILE: " + $f.Name + " =====")
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Open($f.FullName, $false, $true)
    Write-Output $doc.Content.Text
    $doc.Close($false)
    $word.Quit()
    Write-Output ""
}
