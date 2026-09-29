"""Prepare only the supported offline CLI; default mode prints a download plan.

The plan and verification use the standard library. Downloads require --install;
Apple's system model additionally requires --prepare-apple. Existing model files
are hash checked and never overwritten, including on an interrupted install.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ('qwen17', 'qwen-aligner', 'silero-vad32', 'fluid-model')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def within(base, name):
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe relative path: {name}')
    path = base / relative
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'Path escapes its directory: {name}')
    return path


def manifests(root=ROOT):
    result = []
    for name in MANIFESTS:
        data = json.loads((root / 'config/models' / f'{name}-manifest.json').read_text())
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', data['repo']):
            raise ValueError(f'Invalid model repository: {name}')
        if not re.fullmatch(r'[a-f0-9]{40}', data['revision']):
            raise ValueError(f'Model revision must be a fixed commit: {name}')
        directory = within(root, data['directory'])
        paths = set()
        for entry in data['files']:
            within(directory, entry['path'])
            if entry['path'] in paths:
                raise ValueError(f'Duplicate model path: {entry["path"]}')
            paths.add(entry['path'])
            if not re.fullmatch(r'[a-f0-9]{64}', entry['sha256']) or entry['bytes'] < 0:
                raise ValueError(f'Invalid file digest or size: {entry["path"]}')
        if data['total_bytes'] != sum(entry['bytes'] for entry in data['files']):
            raise ValueError(f'Model size mismatch: {name}')
        result.append(data)
    return result


def verify_file(path, entry):
    if not path.is_file():
        raise FileNotFoundError(f'Missing model asset: {path}')
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise ValueError(f'Model asset differs; preserved without replacement: {path}')


def verify_cache_revision(manifest, directory):
    if manifest['repo'] == 'FluidInference/speaker-diarization-coreml':
        marker = within(directory, '.fluidaudio-revision')
        if marker.read_text().strip() != manifest['revision']:
            raise ValueError(f'FluidAudio cache revision differs: {marker}')


def fetch_model(manifest, root=ROOT, opener=urllib.request.urlopen):
    directory = within(root, manifest['directory'])
    for entry in manifest['files']:
        destination = within(directory, entry['path'])
        if destination.exists():
            verify_file(destination, entry)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (f'https://huggingface.co/{manifest["repo"]}/resolve/'
               f'{manifest["revision"]}/{urllib.parse.quote(entry["path"], safe="/")}')
        print(f'Download: {manifest["repo"]}/{entry["path"]} ({entry["bytes"]} bytes)',
              file=sys.stderr, flush=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix='.' + destination.name + '.',
                                                       suffix='.download', dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'wb') as output, opener(url, timeout=120) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            verify_file(temporary, entry)
            # Atomic and no replacement if another installer created the destination.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                verify_file(destination, entry)
        finally:
            temporary.unlink(missing_ok=True)
    if manifest['repo'] == 'FluidInference/speaker-diarization-coreml':
        # FluidAudio requires this local marker even when all model bytes exist.
        # Publish it only after every pinned asset has passed verification.
        marker = within(directory, '.fluidaudio-revision')
        try:
            with marker.open('x') as stream:
                stream.write(manifest['revision'] + '\n')
        except FileExistsError:
            pass
        verify_cache_revision(manifest, directory)


def command(arguments, root=ROOT, capture=False):
    return subprocess.run([str(arg) for arg in arguments], cwd=root, text=True,
                          check=True, capture_output=capture)


def runtime_config(root=ROOT):
    return json.loads((root / 'config/runtime.json').read_text())


def plan(root=ROOT):
    models = []
    for item in manifests(root):
        absent = sum(entry['bytes'] for entry in item['files']
                     if not (root / item['directory'] / entry['path']).is_file())
        models.append({key: item[key] for key in
                       ('repo', 'revision', 'license', 'purpose', 'directory', 'total_bytes')}
                      | {'missing_file_bytes': absent})
    runtime = runtime_config(root)
    return {'action': 'plan_only_no_downloads', 'models': models,
            'total_model_bytes': sum(item['total_bytes'] for item in models),
            'missing_model_bytes': sum(item['missing_file_bytes'] for item in models),
            'existing_assets_are_not_hash_checked_in_plan': True,
            'runtime': runtime,
            'runtime_size_note': 'Python installed_file_bytes are measured disk bytes, not wheel transfer sizes. '
                                 'Allow additional space for Python, Swift builds, downloads and outputs. '
                                 'Apple download size is not exposed by this API.',
            'next': 'scripts/setup.sh --install [--prepare-apple]'}


def runtime_versions(root=ROOT):
    executable = root / '.venv-asr/bin/python'
    packages = runtime_config(root)['python_packages']
    names = [item['name'] for item in packages]
    code = ('import importlib.metadata as m,json,sys; '
            'print(json.dumps({"python":sys.version.split()[0],"packages":'
            '{name:m.version(name) for name in json.loads(sys.argv[1])}}))')
    result = command([executable, '-c', code, json.dumps(names)], root, capture=True)
    actual = json.loads(result.stdout)
    mismatched = [f'{item["name"]}: expected {item["version"]}, got {actual["packages"][item["name"]]}'
                  for item in packages if actual['packages'][item['name']] != item['version']]
    if mismatched:
        raise ValueError('Python dependency mismatch: ' + '; '.join(mismatched))
    return actual


def verify_vendor(root=ROOT):
    fluid = runtime_config(root)['fluid_audio']
    directory = root / 'vendor/FluidAudio'
    revision = command(['git', '-C', directory, 'rev-parse', 'HEAD'], root, capture=True).stdout.strip()
    if revision != fluid['revision']:
        raise ValueError('Existing FluidAudio checkout differs; no reset was performed')
    patch = root / fluid['patch']
    if digest(patch) != fluid['patch_sha256']:
        raise ValueError('FluidAudio patch digest mismatch')
    changes = command(['git', '-C', directory, 'diff'], root, capture=True).stdout
    staged = command(['git', '-C', directory, 'diff', '--cached'], root, capture=True).stdout
    untracked = command(['git', '-C', directory, 'ls-files', '--others', '--exclude-standard'], root, capture=True).stdout
    if changes != patch.read_text() or staged or untracked:
        raise ValueError('FluidAudio changes differ from the recorded patch; preserved without reset')
    return {'revision': revision, 'patch_sha256': fluid['patch_sha256']}


def verify(root=ROOT):
    report = {'action': 'verify_no_downloads', 'errors': [], 'models': [],
              'audio_uploaded': False}
    for item in manifests(root):
        try:
            directory = within(root, item['directory'])
            for entry in item['files']:
                verify_file(within(directory, entry['path']), entry)
            verify_cache_revision(item, directory)
            report['models'].append({'repo': item['repo'], 'status': 'verified',
                                     'bytes': item['total_bytes']})
        except (OSError, ValueError) as error:
            report['errors'].append(str(error))
    for name, check in [('python', runtime_versions), ('fluid_audio', verify_vendor)]:
        try:
            report[name] = check(root)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            report['errors'].append(f'{name}: {error}')
    for product in runtime_config(root)['swift_products']:
        path = root / '.build/release' / product
        if not path.is_file() or not os.access(path, os.X_OK):
            report['errors'].append(f'Missing executable: {path}')
    apple = root / '.build/release/apple-transcribe'
    if apple.is_file():
        result = subprocess.run([str(apple), '--check-model'], text=True, capture_output=True)
        try:
            report['apple_speech'] = json.loads(result.stdout)
        except ValueError:
            report['apple_speech'] = {'status': 'check_failed', 'message': result.stderr.strip()}
        status = report['apple_speech']
        if (result.returncode or not isinstance(status, dict) or
                status.get('status') != 'ready' or status.get('installed') is not True):
            report['errors'].append('Apple Japanese model is not ready; see apple_speech')
    report['ready'] = not report['errors']
    return report


def prepare_interpreter(python, root=ROOT):
    if python.exists() and os.environ.get('PYTHON'):
        actual = command([python, '-c',
                          'import sys; print(sys._base_executable)'], root, capture=True).stdout.strip()
        if Path(actual).resolve() != Path(sys._base_executable).resolve():
            command([sys.executable, '-m', 'venv', '--clear', root / '.venv-asr'], root)


def install(root=ROOT, prepare_apple=False):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('Installation requires an Apple silicon Mac')
    if int(platform.mac_ver()[0].split('.')[0]) < 26:
        raise ValueError('SpeechTranscriber requires macOS 26 or newer')
    fluid = runtime_config(root)['fluid_audio']
    directory = root / 'vendor/FluidAudio'
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        command(['git', 'clone', '--no-checkout', fluid['repo'], directory], root)
        command(['git', '-C', directory, 'checkout', '--detach', fluid['revision']], root)
    revision = command(['git', '-C', directory, 'rev-parse', 'HEAD'], root, capture=True).stdout.strip()
    if revision != fluid['revision']:
        raise ValueError('Existing FluidAudio revision differs; no checkout/reset was performed')
    changes = command(['git', '-C', directory, 'diff'], root, capture=True).stdout
    if not changes:
        command(['git', '-C', directory, 'apply', root / fluid['patch']], root)
    verify_vendor(root)
    python = root / '.venv-asr/bin/python'
    prepare_interpreter(python, root)
    if not python.exists():
        command([sys.executable, '-m', 'venv', root / '.venv-asr'], root)
    try:
        runtime_versions(root)
    except (OSError, ValueError, subprocess.CalledProcessError):
        command([python, '-m', 'pip', 'install', '--only-binary=:all:',
                 '-r', root / 'requirements/asr.txt'], root)
    runtime_versions(root)
    for item in manifests(root):
        fetch_model(item, root)
    for product in runtime_config(root)['swift_products']:
        command(['swift', 'build', '-c', 'release', '--product', product], root)
    if prepare_apple:
        command([root / '.build/release/apple-transcribe', '--prepare-model'], root)
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Show fixed sources, licenses and sizes; default')
    mode.add_argument('--verify', action='store_true', help='Hash existing assets and check the runtime; no downloads')
    mode.add_argument('--install', action='store_true', help='Download missing assets and install/build the supported CLI')
    parser.add_argument('--prepare-apple', action='store_true', help='With --install, explicitly prepare the Apple ja-JP model')
    args = parser.parse_args()
    if args.prepare_apple and not args.install:
        parser.error('--prepare-apple requires --install')
    if sys.version_info < (3, 12):
        parser.error('Python 3.12+ is required by pinned NumPy/SciPy; validated environment uses Python 3.14.7')
    try:
        report = install(prepare_apple=args.prepare_apple) if args.install else verify() if args.verify else plan()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get('ready', True) else 1
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
