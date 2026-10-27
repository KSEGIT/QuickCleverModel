#!/usr/bin/env python3
"""Install an explicitly selected local Responses provider without changing defaults."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SKILL = 'quickclevermodel-inference'


class SetupError(Exception):
    pass


def fail(message):
    raise SetupError(message)


def dotenv(path):
    """Read assignments as data, never as shell code."""
    result = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].lstrip()
            key, sep, value = line.partition('=')
            if sep and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key.strip()):
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                result[key.strip()] = value
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, key, payload=None, optional=False, timeout=15):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    def expired(signum, frame):
        fail('Authenticated request exceeded its total deadline.')

    # The macOS/Linux CLI runs on the main thread. Interrupt even a body whose
    # bytes keep arriving within the socket inactivity timeout.
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                fail('Server response exceeds size limit.')
            return json.loads(raw)
    except Exception:
        if optional:
            return {}
        fail('Authenticated request failed; check the address, API support, and key.')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def positive(value):
    # bool is an int in Python, but is never a useful context window.
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
        return number if str(number) == str(value) and number > 0 else None
    except (ValueError, TypeError):
        return None


def context(data):
    """Return the smallest known per-request limit, never a total KV allocation."""
    if not isinstance(data, dict):
        return None
    limits = []
    flags = {}
    args = data.get('args')
    if isinstance(args, list):
        for i, arg in enumerate(args):
            if not isinstance(arg, str):
                continue
            name, sep, value = arg.partition('=')
            if name in ('--ctx-size', '-c', '--parallel', '-np', '--kv-unified-per-slot'):
                flags[name] = positive(value if sep else args[i + 1] if i + 1 < len(args) else None)
    total = flags.get('--ctx-size') or flags.get('-c') or positive(data.get('ctx-size')) or positive(data.get('--ctx-size'))
    slots = flags.get('--parallel') or flags.get('-np') or positive(data.get('parallel')) or positive(data.get('total_slots'))
    cap = flags.get('--kv-unified-per-slot') or positive(data.get('kv-unified-per-slot'))
    # Dividing is conservative even with unified KV, where one slot can borrow
    # unused space. Missing/auto slot counts cannot establish a per-request size.
    if total and slots and total // slots > 0:
        limits.append(total // slots)
    if cap:
        limits.append(cap)
    for key in ('n_ctx', 'n_ctx_slot', 'slot_n_ctx', 'context_window', 'context_length'):
        found = positive(data.get(key))
        if found:
            limits.append(found)
    for key in ('status', 'preset', 'params', 'default_generation_settings', 'meta', 'metadata'):
        found = context(data.get(key))
        if found:
            limits.append(found)
    return min(limits) if limits else None


def model_state(model):
    status = model.get('status')
    return status.get('value') if isinstance(status, dict) else status


def smoke_test(url, key, selected):
    # Cold local loads and thinking models need more than a short network probe.
    # Generation remains bounded, with one total deadline and no retries.
    smoke = request(url + '/responses', key, {'model': selected, 'input': 'Reply OK.',
                    'max_output_tokens': 512, 'stream': False}, timeout=120)
    if not isinstance(smoke, dict) or smoke.get('status') in ('failed', 'cancelled', 'incomplete') or 'error' in smoke:
        fail('Responses smoke test did not complete.')
    if not any(
        isinstance(part, dict) and part.get('type') == 'output_text' and isinstance(part.get('text'), str) and part['text'].strip()
        for output in smoke.get('output', []) if isinstance(output, dict) and output.get('type') == 'message'
        for part in output.get('content', []) if isinstance(output.get('content'), list)):
        fail('Responses smoke test returned no final text; reasoning alone does not verify inference.')


def catalog(models, contexts):
    try:
        proc = subprocess.run(['codex', 'debug', 'models', '--bundled'],
                              capture_output=True, text=True, timeout=20, check=True)
        bundled = json.loads(proc.stdout)
        templates = bundled['models']
        template = next(m for m in templates if isinstance(m, dict)
                        and m.get('use_responses_lite') is False
                        and m.get('tool_mode') is None and 'context_window' in m)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, StopIteration, TypeError):
        fail('Installed Codex bundled catalog has no supported Responses template; update or inspect Codex.')
    entries = []
    for index, model in enumerate(models):
        item = copy.deepcopy(template)
        # Preserve required schema fields; replace model-specific instructions/capabilities.
        for key in ('base_instructions', 'model_messages', 'upgrade', 'availability_nux'):
            item.pop(key, None)
        item.update(slug=model['id'], display_name=model['id'],
                    description='Local Responses model', context_window=contexts[model['id']],
                    base_instructions='You are a helpful coding assistant. Follow the user instructions and verify your work.',
                    max_context_window=contexts[model['id']], input_modalities=['text'],
                    supported_reasoning_levels=[], default_reasoning_level=None,
                    visibility='list', supported_in_api=True, priority=index,
                    additional_speed_tiers=[], service_tiers=[],
                    supports_search_tool=False, supports_image_detail_original=False,
                    supports_experimental_context=False, support_verbosity=False,
                    experimental_supported_tools=[], use_responses_lite=False)
        entries.append(item)
    return {'models': entries}


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_dir(path):
    if path.is_symlink():
        fail('Refusing symlink at managed directory.')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def write(path, data, backup=True):
    if path.is_symlink():
        fail('Refusing symlink at managed file.')
    if path.exists() and path.read_bytes() == data:
        path.chmod(0o600)
        return
    if path.exists() and backup:
        backup = path.with_name(path.name + '.bak-' + str(time.time_ns()))
        shutil.copy2(path, backup)
        backup.chmod(0o600)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--name', default='qcm')
    parser.add_argument('--model')
    parser.add_argument('--context-window', type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--skills-dir', type=Path, default=Path.home() / '.agents/skills')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_-]*', args.name) or args.name in ('openai', 'ollama', 'lmstudio'):
        fail('Use a non-reserved provider name containing letters, numbers, underscores or hyphens.')
    if args.context_window is not None and args.context_window <= 0:
        fail('Context window must be positive.')
    settings = dotenv(Path(os.environ.get('BONSAI_ENV_FILE', ROOT / '.env')))
    settings.update(os.environ)
    url = (args.base_url or settings.get('BONSAI_SERVER_URL') or 'http://127.0.0.1:8080').rstrip('/')
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        fail('Use an HTTP(S) base URL without credentials, query or fragment.')
    if parsed.hostname in ('0.0.0.0', '::'):
        fail('Use a connect address, not a wildcard bind address.')
    if not url.endswith('/v1'):
        url += '/v1'
    key = args.key_file.read_text(encoding='utf-8').strip() if args.key_file else settings.get('BONSAI_SERVER_KEY') or settings.get('BONSAI_API_KEY', '')
    if not key or '\n' in key or '\r' in key:
        fail('Provide a key file or BONSAI_SERVER_KEY/BONSAI_API_KEY.')
    discovered = request(url + '/models', key)
    models = discovered.get('data') if isinstance(discovered, dict) else None
    if not isinstance(models, list) or not models or any(not isinstance(m, dict) or not isinstance(m.get('id'), str) or not m['id'] for m in models):
        fail('Server returned no valid model list.')
    ids = [m['id'] for m in models]
    if key in url or any(key in model for model in ids):
        fail('Refusing credential-bearing server metadata or address.')
    if len(set(ids)) != len(ids):
        fail('Server returned duplicate model IDs.')
    selected = args.model or next((m['id'] for m in models if model_state(m) == 'loaded'), ids[0])
    if selected not in ids:
        fail('Selected model is not in the authenticated model list.')
    contexts = {}
    for model in models:
        props = {}
        # Router /props can auto-load an unrelated model and evict the selected
        # one. Never probe explicitly cold models; prevent autoload even if a
        # loaded/unknown model changes state between discovery and this request.
        if model_state(model) not in ('unloaded', 'sleeping', 'cached', 'downloading', 'incomplete'):
            props = request(url[:-3] + '/props?model=' + urllib.parse.quote(model['id'], safe='') + '&autoload=false', key, optional=True)
        limits = [n for n in (args.context_window, context(model), context(props)) if n]
        value = min(limits) if limits else None
        if not value:
            fail('A discovered model has no live context size; provide --context-window explicitly.')
        contexts[model['id']] = value
    smoke_test(url, key, selected)
    generated_catalog = catalog(models, contexts)
    home = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')).expanduser().resolve()
    managed = home / 'quickclevermodel' / args.name
    config_path = home / 'config.toml'
    for path in (home / 'quickclevermodel', managed, config_path):
        if path.is_symlink():
            fail('Refusing symlink at managed configuration path.')
    text = config_path.read_text(encoding='utf-8') if config_path.exists() else ''
    cfg = tomllib.loads(text)
    manifest_path = managed / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else None
    begin = '# BEGIN quickclevermodel ' + args.name
    end = '# END quickclevermodel ' + args.name
    old_block = ''
    if manifest:
        old_block = manifest.get('config_block', '')
        if not old_block or text.count(old_block) != 1:
            fail('Managed configuration was changed; restore it or choose another name.')
        for name, checksum in manifest.get('files', {}).items():
            if name not in ('api-key', 'client.json', 'models.json'):
                fail('Invalid managed client manifest.')
            path = managed / name
            if path.is_symlink() or not path.is_file() or digest(path) != checksum:
                fail('Managed client files were changed; restore them or choose another name.')
        for name, checksum in manifest.get('profiles', {}).items():
            if not re.fullmatch(r'[A-Za-z0-9_-]+\.config\.toml', name):
                fail('Invalid managed profile manifest.')
            path = home / name
            if path.is_symlink() or not path.is_file() or digest(path) != checksum:
                fail('Managed profile files were changed; restore them or choose another name.')
        remaining = text.replace(old_block, '', 1)
    else:
        if managed.exists() or begin in text or end in text:
            fail('Client path or configuration marker is not managed by this setup.')
        remaining = text
    profiles = [args.name] + [args.name + '-' + re.sub(r'[^A-Za-z0-9_-]+', '-', model).strip('-').lower() for model in ids]
    if len(set(profiles)) != len(profiles) or any(p.endswith('-') for p in profiles):
        fail('Model IDs collide after profile name sanitization.')
    unmanaged = tomllib.loads(remaining)
    if args.name in unmanaged.get('model_providers', {}) or any(p in unmanaged.get('profiles', {}) for p in profiles):
        fail('A provider or profile name conflicts with unmanaged configuration.')
    old_profiles = manifest.get('profiles', {}) if manifest else {}
    for name in profiles:
        path = home / (name + '.config.toml')
        if path.is_symlink() or (path.exists() and path.name not in old_profiles):
            fail('A profile file conflicts with unmanaged configuration.')
    lines = [begin, '[model_providers.' + args.name + ']', 'name = ' + encoded('QuickCleverModel ' + args.name),
             'base_url = ' + encoded(url), 'wire_api = "responses"',
             '[model_providers.' + args.name + '.auth]', 'command = "cat"',
             'args = [' + encoded(str(managed / 'api-key')) + ']']
    profile_data = {}
    for name, model in zip(profiles, [selected] + ids):
        profile_data[name + '.config.toml'] = ('\n'.join([
            '# Managed by quickclevermodel setup', 'model_provider = ' + encoded(args.name),
            'model = ' + encoded(model), 'model_catalog_json = ' + encoded(str(managed / 'models.json'))]) + '\n').encode()
    lines.append(end)
    block = '\n'.join(lines) + '\n'
    if old_block:
        new_text = text.replace(old_block, block, 1)
    else:
        new_text = text + ('\n' if text and not text.endswith('\n\n') else '') + block
    tomllib.loads(new_text)
    source = ROOT / '.agents/skills' / SKILL
    sources = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if 'SKILL.md' not in sources or 'scripts/inference.py' not in sources:
        fail('Repository inference skill package is missing.')
    skill = args.skills_dir.expanduser().resolve() / SKILL
    skill_manifest = skill / '.quickclevermodel-managed.json'
    if skill.is_symlink():
        fail('Refusing symlink at skill path.')
    if skill.exists():
        if not skill_manifest.is_file():
            fail('Global skill path exists without setup ownership.')
        owned = json.loads(skill_manifest.read_text(encoding='utf-8'))
        actual = {str(p.relative_to(skill)): digest(p) for p in skill.rglob('*') if p.is_file() and p != skill_manifest and '__pycache__' not in p.parts}
        if owned != actual:
            fail('Global skill was changed; restore it before updating.')
        if any((skill / rel).is_symlink() for rel in actual):
            fail('Refusing symlink inside skill package.')
    sources['client-path'] = (str(managed / 'client.json') + '\n').encode()
    if args.check:
        print('Client checks passed; no files changed.')
        return
    private_dir(home)
    private_dir(home / 'quickclevermodel')
    private_dir(managed)
    private_dir(skill)
    payloads = {'api-key': key.encode() + b'\n',
                'client.json': (json.dumps({'base_url': url, 'key_file': str(managed / 'api-key'), 'default_model': selected, 'models': ids}, indent=2) + '\n').encode(),
                'models.json': (json.dumps(generated_catalog, indent=2) + '\n').encode()}
    for name, content in payloads.items():
        write(managed / name, content)
    for rel, content in sources.items():
        private_dir((skill / rel).parent)
        write(skill / rel, content, backup=False)
    # Skill files are replaced without backups; snapshot the installed package.
    checksums = {str(p.relative_to(skill)): digest(p) for p in skill.rglob('*') if p.is_file() and p != skill_manifest and '__pycache__' not in p.parts}
    write(skill_manifest, (json.dumps(checksums, sort_keys=True) + '\n').encode(), backup=False)
    write(config_path, new_text.encode())
    for name, content in profile_data.items():
        write(home / name, content)
    for name in old_profiles.keys() - profile_data.keys():
        path = home / name
        path.rename(path.with_name(path.name + '.bak-' + str(time.time_ns())))
    write(manifest_path, (json.dumps({'config_block': block,
          'files': {n: digest(managed / n) for n in payloads},
          'profiles': {n: digest(home / n) for n in profile_data}}, indent=2) + '\n').encode(), backup=False)
    print('Client configured. Start Codex with --profile ' + args.name + '.')
    for name in profiles[1:]:
        print('codex --profile ' + name)


if __name__ == '__main__':
    try:
        run()
    except (SetupError, OSError, ValueError, KeyError, TypeError):
        # Never print remote bodies, subprocess output, configuration or credentials.
        error = sys.exc_info()[1]
        print(str(error) if isinstance(error, SetupError) else 'Client setup failed; check input files and configuration.', file=sys.stderr)
        sys.exit(1)
