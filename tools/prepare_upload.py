"""Write the raw Track-2 descriptor, then let the OFFICIAL toolkit seal and pack it.

## Executive summary (read this first)

An upload is a zip holding `submission.json` (the sealed descriptor) and `team-claim.json`
(the team proof). The organizers publish `qfbench2 submission pack`, which derives your
`team_id`, seals `descriptor_digest`, computes the HMAC proof and writes the zip. This script
does NOT reimplement any of that cryptography: it only writes the raw descriptor from
`submission/image.json`, validates the field values the schema is strict about, and then
invokes the official command.

The Team Key is read by the official command from a hidden prompt or from a mode-600 file
that this script creates, uses and deletes. It never appears on a command line, in a log, or
in the zip.

    python tools/prepare_upload.py --team-number YOUR_TEAM_NUMBER --phase dev              # dev upload
    python tools/prepare_upload.py --team-number YOUR_TEAM_NUMBER --phase dev --descriptor-only

Official command it delegates to, for the record:

    python -m qfbench2_common.cli submission pack \
        --team-number YOUR_TEAM_NUMBER --descriptor submission/out/submission.json --out submission/out/submission.zip
"""
from pathlib import Path
import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = {'dev': 'agenthon2026-forecasting-dev', 'final': 'agenthon2026-forecasting-final'}
DIGEST_RE = re.compile(r'^sha256:[0-9a-f]{64}$')
REPOSITORY_RE = re.compile(r'^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$')
REGISTRY_RE = re.compile(r'^[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?(?::[0-9]{1,5})?$')

#: The pinned house model from the organizers' HOUSE-MODEL documentation. MODEL_NAME overrides
#: it at run time; this is the declaration the descriptor requires.
DEFAULT_MODELS = [{'name': 'nvidia/nemotron-3-super-120b-a12b', 'version': 'rl-030326-fp8',
                   'training_cutoff': 'unpublished', 'access': 'api', 'revision': 'rl-030326-fp8'}]


def fail(message):
    raise SystemExit(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, default=ROOT / 'submission/image.json')
    p.add_argument('--phase', choices=['dev', 'final'], default='dev')
    p.add_argument('--team-number', type=int, required=True, help='your agenthon.net team number')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'submission/out')
    p.add_argument('--license', default='Apache-2.0')
    p.add_argument('--descriptor-only', action='store_true',
                   help='write submission.json and stop; do not read the Team Key or pack')
    p.add_argument('--linked-descriptor', type=Path,
                   help='prior scored descriptor or ZIP from the same linked account and competition; pack without a new claim')
    a = p.parse_args()

    if not a.image.exists():
        fail(f'{a.image} not found; fill in registry/repository/digest after pushing the image')
    image = json.loads(a.image.read_text())
    if not DIGEST_RE.match(str(image.get('digest', ''))):
        fail('image.digest must be sha256: + 64 lowercase hex (a tag is rejected before the pull)')
    if not REGISTRY_RE.match(str(image.get('registry', ''))):
        fail('image.registry must be a hostname[:port], e.g. docker.io or ghcr.io')
    if not REPOSITORY_RE.match(str(image.get('repository', ''))):
        fail('image.repository must be a lowercase path, e.g. yourname/qfb2-forecaster')

    models_file = ROOT / 'submission/models.json'
    models = json.loads(models_file.read_text()) if models_file.exists() else DEFAULT_MODELS
    if 'REPLACE' in json.dumps(image):
        fail('submission/image.json still holds the placeholder text; fill in the real digest first')

    a.out_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = a.out_dir / 'submission.json'
    descriptor = {'schema_version': '1.0.0', 'interface_version': '2.0',
                  'competition_id': COMPETITION[a.phase], 'track': 'forecasting', 'phase': a.phase,
                  'category': 'api',
                  'image': {'registry': image['registry'], 'repository': image['repository'],
                            'digest': image['digest']},
                  'image_access': 'public', 'models': models, 'license': a.license}
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + '\n')
    print(f'wrote {descriptor_path}')
    print(json.dumps(descriptor['image'], indent=2))

    if a.descriptor_only:
        print('\nDescriptor written. Review it, then rerun without --descriptor-only to pack.',
              file=sys.stderr)
        return 0

    zip_path = a.out_dir / 'submission.zip'
    if zip_path.exists():
        fail(f'{zip_path} exists; move it aside so an older upload cannot be mistaken for this one')

    if a.linked_descriptor:
        from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest
        if zipfile.is_zipfile(a.linked_descriptor):
            with zipfile.ZipFile(a.linked_descriptor) as previous_zip:
                previous = json.loads(previous_zip.read('submission.json'))
        else:
            previous = json.loads(a.linked_descriptor.read_text())
        SubmissionDescriptor.from_mapping(previous)
        if previous['competition_id'] != descriptor['competition_id']:
            fail('linked descriptor must be from the same competition')
        descriptor['team_id'] = previous['team_id']
        sealed = seal_descriptor_digest(descriptor)
        SubmissionDescriptor.from_mapping(sealed)
        payload = (json.dumps(sealed, indent=2, sort_keys=True) + '\n').encode()
        descriptor_path.write_bytes(payload)
        fd = os.open(zip_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            with zipfile.ZipFile(handle, 'w', zipfile.ZIP_STORED) as archive:
                archive.writestr('submission.json', payload)
        print(f'zip: {zip_path} (linked account only; no team claim)')
        return 0

    key = os.environ.get('AGENTHON_TEAM_KEY') or getpass.getpass('Team Key (hidden, never saved): ')
    if not key:
        fail('no Team Key given')
    fd, key_file = tempfile.mkstemp(prefix='agenthon-key-')
    try:
        os.write(fd, key.encode())
        os.close(fd)
        os.chmod(key_file, 0o600)
        cmd = [sys.executable, '-m', 'qfbench2_common.cli', 'submission', 'pack',
               '--team-number', str(a.team_number), '--team-key-file', key_file,
               '--descriptor', str(descriptor_path), '--out', str(zip_path)]
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode:
            fail(f'official pack failed with exit {proc.returncode}')
    finally:
        # Remove the key file on every path, including a failure.
        try:
            os.unlink(key_file)
        except OSError:
            pass
    if not zip_path.exists():
        fail('pack reported success but wrote no zip')
    os.chmod(zip_path, 0o600)
    print(f'\nzip: {zip_path}  ({zip_path.stat().st_size} bytes, mode 0600)')
    print('Upload it as it is: the claim is bound to these exact descriptor bytes.', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
