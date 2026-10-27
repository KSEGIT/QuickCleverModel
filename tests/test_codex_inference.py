"""Offline HTTP contract tests for the installed inference helper."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / '.agents/skills/quickclevermodel-inference'
SPEC = importlib.util.spec_from_file_location('qcm_inference', SKILL / 'scripts/inference.py')
INFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFERENCE)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.calls = []
        self.status = 200
        self.delay = 0
        self.drip_interval = 0
        self.response = {'output': [{'type': 'reasoning', 'summary': []}, {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'checked'}]}]}
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.reply()
            def do_POST(self):
                self.reply()
            def reply(self):
                body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                owner.calls.append((self.path, self.headers.get('Authorization'), json.loads(body) if body else None))
                time.sleep(owner.delay)
                self.send_response(owner.status)
                self.send_header('Location', '/stolen')
                self.end_headers()
                try:
                    data = json.dumps(owner.response).encode()
                    if owner.drip_interval:
                        for byte in data:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(owner.drip_interval)
                    else:
                        self.wfile.write(data)
                except BrokenPipeError:
                    pass
            def log_message(self, *_):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.key = self.home / 'api-key'
        self.key.write_text('test-secret-token')
        self.config = self.home / 'quickclevermodel/qcm/client.json'
        self.config.parent.mkdir(parents=True)
        self.config.write_text(json.dumps({'base_url': f'http://127.0.0.1:{self.server.server_port}/v1', 'key_file': str(self.key), 'default_model': 'default', 'models': ['default', 'other']}))

    def run_helper(self, *args, script=None, prompt='chosen context'):
        return subprocess.run([sys.executable, str(script or SKILL / 'scripts/inference.py'), *args], input=prompt, text=True, capture_output=True, env={**os.environ, 'CODEX_HOME': str(self.home)}, timeout=10)

    def test_default_request_and_text_only(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'checked\n')
        path, auth, body = self.calls[0]
        self.assertEqual(path, '/v1/responses')
        self.assertEqual(auth, 'Bearer test-secret-token')
        self.assertEqual(body, {'model': 'default', 'input': 'chosen context', 'max_output_tokens': 1024})

    def test_selected_model_prompt_file_and_output_bound(self):
        prompt = self.home / 'prompt'
        prompt.write_text('review only this')
        result = self.run_helper('--config', str(self.config), '--model', 'other', '--prompt-file', str(prompt), '--max-output-tokens', '77', '--timeout', '2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls[0][2], {'model': 'other', 'input': 'review only this', 'max_output_tokens': 77})

    def test_installed_marker_works_without_checkout(self):
        installed = self.home / 'installed'
        shutil.copytree(SKILL, installed)
        (installed / 'client-path').write_text(str(self.config))
        alternate = self.home / 'selected-client.json'
        self.config.rename(alternate)
        (installed / 'client-path').write_text(str(alternate))
        result = self.run_helper(script=installed / 'scripts/inference.py')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_list_outputs_only_model_ids(self):
        self.response = {'data': [{'id': 'first', 'args': 'test-secret-token'}, {'id': 'second'}]}
        result = self.run_helper('--list')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'first\nsecond\n')
        self.assertEqual(self.calls[0][0], '/v1/models')

    def test_redirects_and_errors_are_safe_without_retries(self):
        for status in (302, 307, 401, 500):
            with self.subTest(status=status):
                self.calls.clear()
                self.status = status
                self.response = {'error': 'test-secret-token'}
                result = self.run_helper()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertNotIn('test-secret-token', result.stderr)
                self.assertEqual(len(self.calls), 1)

    def test_rejects_bad_input_without_network(self):
        for args in (('--timeout', 'nan'), ('--timeout', '0'), ('--max-output-tokens', '-1'), ('--model', 'unknown'), ('--unknown-secret-token',)):
            with self.subTest(args=args):
                result = self.run_helper(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertNotIn('unknown-secret-token', result.stderr)
        self.assertEqual(self.calls, [])

    def test_returned_commands_are_never_executed(self):
        target = self.home / 'must-not-exist'
        text = f'touch {target}'
        self.response['output'][1]['content'][0]['text'] = text
        result = self.run_helper()
        self.assertEqual(result.stdout, text + '\n')
        self.assertFalse(target.exists())

    def test_explicit_config_overrides_installed_marker(self):
        installed = self.home / 'installed'
        shutil.copytree(SKILL, installed)
        (installed / 'client-path').write_text(str(self.home / 'missing'))
        result = self.run_helper('--config', str(self.config), script=installed / 'scripts/inference.py')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_timeout_is_finite(self):
        self.delay = 0.3
        start = time.monotonic()
        result = self.run_helper('--timeout', '0.05')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertLess(time.monotonic() - start, 2)

    def test_drip_response_obeys_whole_request_deadline(self):
        self.drip_interval = 0.01
        start = time.monotonic()
        result = self.run_helper('--timeout', '0.15')
        elapsed = time.monotonic() - start
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, 'Inference timed out.\n')
        self.assertLess(elapsed, 0.8)
        self.assertEqual(len(self.calls), 1)

    def test_connection_failure_is_safe(self):
        self.server.shutdown()
        self.server.server_close()
        result = self.run_helper('--timeout', '0.1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertNotIn('test-secret-token', result.stderr)

    def test_url_credentials_are_rejected_before_network(self):
        config = json.loads(self.config.read_text())
        config['base_url'] = 'http://test-secret-token:password@127.0.0.1/v1'
        self.config.write_text(json.dumps(config))
        result = self.run_helper()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('test-secret-token', result.stderr)
        self.assertEqual(self.calls, [])

    def test_http_transport_warning_allows_remote_endpoints(self):
        config = json.loads(self.config.read_text())
        for base, warn in (
                ('http://127.0.0.1/v1', False),
                ('http://127.0.0.2/v1', False),
                ('http://[::1]/v1', False),
                ('http://localhost/v1', False),
                ('http://192.168.1.10/v1', True),
                ('http://100.64.0.1/v1', True),
                ('http://inference.internal/v1', True),
                ('https://inference.internal/v1', False)):
            with self.subTest(base=base), \
                    mock.patch.object(INFERENCE.urllib.request.OpenerDirector, 'open', return_value=io.BytesIO(b'{}')) as opened, \
                    mock.patch('sys.stderr', new_callable=io.StringIO) as stderr:
                self.assertEqual(INFERENCE.request({**config, 'base_url': base}, '/models', 1), {})
                self.assertEqual(opened.call_count, 1)
                self.assertEqual(opened.call_args.args[0].full_url, base + '/models')
                if warn:
                    self.assertIn('Warning: non-loopback HTTP', stderr.getvalue())
                    self.assertIn('HTTPS or a protected tunnel/network', stderr.getvalue())
                    self.assertNotIn('test-secret-token', stderr.getvalue())
                    self.assertNotIn(base, stderr.getvalue())
                else:
                    self.assertEqual(stderr.getvalue(), '')

    def test_missing_text_and_malformed_responses_fail_safely(self):
        for payload in ({'output': []}, {'output': None}, {'error': 'test-secret-token'}):
            self.response = payload
            result = self.run_helper()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, '')
            self.assertNotIn('test-secret-token', result.stderr)


if __name__ == '__main__':
    unittest.main()
