# Download and verify the research assets

Use [Zenodo assets v1.0.0](https://doi.org/10.5281/zenodo.23235197), published on 8 October 2026 under CC BY 4.0. The record contains synthetic data, the forward operator, 27 registered checkpoints, and supporting arrays. Source code remains under MIT. Raw third-party Vinton observations are excluded.

## Files to download

Download every part of these three archives from the record:

| Archive prefix | Parts | Final archive |
|---|---:|---|
| `TARF-GDC-assets-v1.0.0-dataset.zip.part-` | 001–011 | `TARF-GDC-assets-v1.0.0-dataset.zip` |
| `TARF-GDC-assets-v1.0.0-checkpoints.zip.part-` | 001–005 | `TARF-GDC-assets-v1.0.0-checkpoints.zip` |
| `TARF-GDC-assets-v1.0.0-supporting.zip.part-` | 001–002 | `TARF-GDC-assets-v1.0.0-supporting.zip` |

Also download `ZENODO_PARTS_SHA256SUMS.txt`, `ARCHIVE_SHA256SUMS.txt`, `PAYLOAD_SHA256SUMS.txt`, `DOWNLOAD_AND_REASSEMBLE.md`, `LICENSE_DATA.md`, and the sanitization notes/report. All listed record files total 4,445,556,555 bytes (approximately 4.45 GB, decimal). Reconstructed ZIPs and extracted files need additional space. A single part is not a complete ZIP.

The record DOI above is authoritative if the bundled README still contains DOI placeholders. Asset version 1.0.0 is not a claim that the source repository has a matching tag.

## Linux or macOS

In a directory containing all downloaded parts, verify them before reassembly. These commands refuse to overwrite an existing reconstructed ZIP:

```bash
sha256sum -c ZENODO_PARTS_SHA256SUMS.txt || exit 1
for group in dataset checkpoints supporting; do
  archive="TARF-GDC-assets-v1.0.0-${group}.zip"
  if [ -e "$archive" ]; then
    echo "Already exists: $archive" >&2
    exit 1
  fi
  cat "${archive}.part-"* > "$archive"
done
sha256sum -c ARCHIVE_SHA256SUMS.txt || exit 1
unzip TARF-GDC-assets-v1.0.0-dataset.zip
unzip TARF-GDC-assets-v1.0.0-checkpoints.zip
unzip TARF-GDC-assets-v1.0.0-supporting.zip
cd TARF-GDC-assets-v1.0.0
sha256sum -c ../PAYLOAD_SHA256SUMS.txt
```

Stop if any checksum check fails. On macOS, use `shasum -a 256 -c` in place of `sha256sum -c`, or the GNU coreutils equivalent.

## Windows PowerShell

Run from the directory containing the downloaded files. The function below verifies file names and SHA-256 values from each manifest:

```powershell
function Test-AssetHashes([string]$ManifestPath) {
  foreach ($entry in Get-Content -LiteralPath $ManifestPath) {
    if ([string]::IsNullOrWhiteSpace($entry)) { continue }
    if ($entry -notmatch '^([a-fA-F0-9]{64})\s+[*]?(.+)$') {
      throw "Invalid checksum entry: $entry"
    }
    $expectedDigest = $Matches[1]
    $assetRelativePath = $Matches[2]
    $actualDigest = (Get-FileHash -LiteralPath $assetRelativePath -Algorithm SHA256).Hash
    if ($actualDigest -ne $expectedDigest) { throw "Checksum mismatch: $assetRelativePath" }
  }
}
Test-AssetHashes 'ZENODO_PARTS_SHA256SUMS.txt'
$partCounts = @{dataset=11; checkpoints=5; supporting=2}
foreach ($assetGroup in 'dataset', 'checkpoints', 'supporting') {
  $archiveName = "TARF-GDC-assets-v1.0.0-$assetGroup.zip"
  $archivePath = Join-Path $PWD $archiveName
  $archiveParts = @(Get-ChildItem -File "$archiveName.part-*" | Sort-Object Name)
  if ($archiveParts.Count -ne $partCounts[$assetGroup]) { throw "Incomplete parts: $assetGroup" }
  $archiveStream = [System.IO.File]::Open($archivePath, [System.IO.FileMode]::CreateNew)
  try {
    foreach ($archivePart in $archiveParts) {
      $partStream = [System.IO.File]::OpenRead($archivePart.FullName)
      try { $partStream.CopyTo($archiveStream) } finally { $partStream.Dispose() }
    }
  } finally { $archiveStream.Dispose() }
}
Test-AssetHashes 'ARCHIVE_SHA256SUMS.txt'
foreach ($assetGroup in 'dataset', 'checkpoints', 'supporting') {
  Expand-Archive -LiteralPath "TARF-GDC-assets-v1.0.0-$assetGroup.zip" -DestinationPath .
}
Push-Location 'TARF-GDC-assets-v1.0.0'
try { Test-AssetHashes '../PAYLOAD_SHA256SUMS.txt' } finally { Pop-Location }
```

Use a fresh extraction location. The three archives share the same `TARF-GDC-assets-v1.0.0` root. After verification, pass that directory to `--assets`, or rename it to `TARF-GDC-assets` beside the repository to match the examples in the [user guide](USER_GUIDE.md). The asset root must directly contain `dataset/`, `operator/`, and `checkpoints/`.

The supplied `SANITIZATION_NOTES.md` and `CHECKPOINT_SANITIZATION_REPORT.json` describe removal of machine-specific metadata and preservation of numerical content. Keep registered filenames and public hashes unchanged. Do not substitute earlier checkpoints with the same display name.
