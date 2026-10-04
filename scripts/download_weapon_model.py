"""Download the pinned upstream model over verified HTTPS; validate before install."""
import hashlib
import json
import os
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download():
    manifest = json.loads((ROOT / 'models' / 'weapons.json').read_text())
    expected = manifest['sha256']
    destination = ROOT / 'models' / 'weapons.onnx'
    if destination.is_file():
        with destination.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() == expected:
                print('Verified weapon model already installed.')
                return
        raise RuntimeError('Existing model checksum mismatch; preserve and inspect it before replacement.')
    url = ('https://raw.githubusercontent.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8/'
           + manifest['upstream_commit'] + '/' + manifest['upstream_path'])
    # Honor the cloud's configured CA bundle while keeping TLS verification enabled.
    context = ssl.create_default_context(cafile=os.environ.get('SSL_CERT_FILE') or None)
    temporary = destination.with_suffix('.part')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, context=context, timeout=60) as response, temporary.open('wb') as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError('Downloaded model checksum mismatch; model was not installed.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print('Installed verified gun-and-knife model. Read models/NOTICE.md for license information.')


if __name__ == '__main__':
    download()
