#!/usr/bin/env python3
"""Send one chosen task to a configured Responses-compatible local endpoint."""
import argparse
import json
import math
import os
import signal
from pathlib import Path
import sys
import urllib.request
from urllib.parse import urlsplit

MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class SafeError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise SafeError('Invalid arguments; use --help for usage.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SafeError('Redirect refused.')


def config_path(explicit):
    if explicit:
        return Path(explicit).expanduser()
    marker = Path(__file__).resolve().parents[1] / 'client-path'
    if marker.exists():
        return Path(marker.read_text().strip()).expanduser()
    return Path(os.environ.get('CODEX_HOME', '~/.codex')).expanduser() / 'quickclevermodel/qcm/client.json'


def request(config, endpoint, timeout, body=None):
    base = config['base_url'].rstrip('/')
    url = urlsplit(base)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment:
        raise SafeError('Invalid endpoint configuration.')
    key = Path(config['key_file']).expanduser().read_text().strip()
    if not key or '\n' in key or '\r' in key:
        raise SafeError('Invalid credential file.')
    req = urllib.request.Request(base + endpoint, data=None if body is None else json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # Disable proxy environment routing and every redirect: credentials stay on the chosen endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def deadline_expired(signum, frame):
        raise SafeError('Inference timed out.')

    # This macOS/Linux CLI runs on the main thread. A process timer interrupts
    # connection, headers, and body reads even when a peer keeps dripping bytes.
    previous_handler = signal.signal(signal.SIGALRM, deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with opener.open(req, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except TimeoutError:
        raise SafeError('Inference timed out.') from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    if len(data) > MAX_RESPONSE_BYTES:
        raise SafeError('Response exceeded the size limit.')
    result = json.loads(data)
    if not isinstance(result, dict) or 'error' in result:
        raise SafeError('Invalid API response.')
    return result


def main():
    parser = Parser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--prompt-file')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_output_tokens <= 0:
        raise SafeError('Timeout and output limit must be positive and finite.')
    config = json.loads(config_path(args.config).read_text())
    if args.list:
        result = request(config, '/models', args.timeout)
        models = result.get('data')
        if not isinstance(models, list) or not models or any(not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or any(c in item['id'] for c in '\r\n') for item in models):
            raise SafeError('Invalid model list.')
        print('\n'.join(item['id'] for item in models))
        return
    model = args.model or config['default_model']
    if not isinstance(model, str) or not model or model not in config['models']:
        raise SafeError('Model is not in the configured catalog; run setup again to refresh it.')
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else sys.stdin.read()
    if not prompt.strip():
        raise SafeError('Provide a nonempty prompt file or stdin.')
    result = request(config, '/responses', args.timeout, {'model': model, 'input': prompt, 'max_output_tokens': args.max_output_tokens})
    if result.get('status') in ('failed', 'cancelled', 'incomplete'):
        raise SafeError('Inference did not complete.')
    texts = []
    for item in result.get('output', []):
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            for part in item.get('content', []):
                if part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                    texts.append(part['text'])
    if not texts or not ''.join(texts).strip():
        raise SafeError('Response contained no final text.')
    print('\n'.join(texts))


if __name__ == '__main__':
    try:
        main()
    except SafeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception:
        # Exception strings may contain keys, URLs, prompts, or an untrusted server body.
        print('Inference failed; check the client configuration and endpoint.', file=sys.stderr)
        sys.exit(1)
