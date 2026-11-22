param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$stage = 'load WinRT types'

$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]

$script:AsTaskGeneric = ([System.WindowsRuntimeSystemExtensions]).GetMethods() |
    Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
        $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
if ($null -eq $script:AsTaskGeneric) {
    throw 'Could not find the Windows Runtime AsTask bridge.'
}

function Wait-WinRT([object]$Operation, [type]$ResultType) {
    $method = $script:AsTaskGeneric.MakeGenericMethod($ResultType)
    $task = $method.Invoke($null, @($Operation))
    return $task.GetAwaiter().GetResult()
}

$path = [System.IO.Path]::GetFullPath($ImagePath)
$stage = 'open image stream'
Write-Verbose $stage
$netStream = [System.IO.File]::OpenRead($path)
$stream = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($netStream)
$stage = 'decode image'
Write-Verbose $stage
$decoder = Wait-WinRT ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$stage = 'convert software bitmap'
Write-Verbose $stage
$bitmap = Wait-WinRT ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$stage = 'create OCR engine'
Write-Verbose $stage
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    throw 'Windows OCR has no recognizer for the current user languages.'
}
$stage = 'recognize image text'
Write-Verbose $stage
$result = Wait-WinRT ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$lines = @(
    foreach ($line in $result.Lines) {
        $left = [double]::PositiveInfinity
        $top = [double]::PositiveInfinity
        $right = [double]::NegativeInfinity
        $bottom = [double]::NegativeInfinity
        foreach ($word in $line.Words) {
            $rect = $word.BoundingRect
            $x = [double]$rect.X
            $y = [double]$rect.Y
            $wordRight = $x + [double]$rect.Width
            $wordBottom = $y + [double]$rect.Height
            if ($x -lt $left) { $left = $x }
            if ($y -lt $top) { $top = $y }
            if ($wordRight -gt $right) { $right = $wordRight }
            if ($wordBottom -gt $bottom) { $bottom = $wordBottom }
        }
        [pscustomobject]@{
            text = $line.Text
            x = if ([double]::IsInfinity($left)) { 0.0 } else { $left }
            y = if ([double]::IsInfinity($top)) { 0.0 } else { $top }
            width = if ([double]::IsInfinity($right)) { 0.0 } else { $right - $left }
            height = if ([double]::IsInfinity($bottom)) { 0.0 } else { $bottom - $top }
        }
    }
)
[pscustomobject]@{
    width = [double]$bitmap.PixelWidth
    height = [double]$bitmap.PixelHeight
    text = [string]$result.Text
    lines = $lines
} | ConvertTo-Json -Depth 6 -Compress
$stream.Dispose()
$netStream.Dispose()
