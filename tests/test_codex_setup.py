"""Offline client setup integration tests; no personal configuration is used."""
import http.server
import importlib.util
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import tomllib
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'setup-codex.sh'
TOKEN = 'fixture-private-token'
SPEC = importlib.util.spec_from_file_location('codex_setup', ROOT / 'codex-client/setup.py')
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class SetupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.home = self.root / 'codex'
        self.home.mkdir()
        self.config = self.home / 'config.toml'
        self.original = 'model = "existing-gpt"\n[mcp_servers.keep]\ncommand = "keep"\n'
        self.config.write_text(self.original)
        self.key = self.root / 'key'
        self.key.write_text(TOKEN)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.template = {'slug': 'fixture', 'context_window': 100000,
                         'max_context_window': 100000, 'use_responses_lite': False,
                         'input_modalities': ['text', 'image'],
                         'supported_reasoning_levels': [], 'default_reasoning_level': None}
        self.codex = self.bin / 'codex'
        self.codex.write_text('#!/usr/bin/env python3\nimport json\nprint(' + repr(json.dumps({'models': [self.template]})) + ')\n')
        self.codex.chmod(0o755)
        self.models = [{'id': 'small', 'context_window': 8192},
                       {'id': 'large/model', 'meta': {'context_window': 32768}}]
        self.props = {}
        self.status = 200
        self.response = {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'OK'}]}]}
        self.requests = []
        self.drip = False
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.requests.append((self.path, self.headers.get('Authorization'), None))
                body = {'data': owner.models} if self.path == '/v1/models' else owner.props
                self.send_response(owner.status)
                self.end_headers()
                raw = json.dumps(body).encode()
                try:
                    if owner.drip:
                        for byte in raw:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.03)
                    else:
                        self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append((self.path, self.headers.get('Authorization'), body))
                self.send_response(owner.status)
                self.end_headers()
                self.wfile.write(json.dumps(owner.response).encode())

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def run_setup(self, *args, extra_env=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('BONSAI_', 'QCM_'))}
        env.update(CODEX_HOME=str(self.home), HOME=str(self.root),
                   PATH=str(self.bin) + os.pathsep + env['PATH'], BONSAI_ENV_FILE=str(self.root / 'absent'))
        env.update(extra_env or {})
        return subprocess.run(['bash', str(SCRIPT), '--base-url', self.url,
                               '--key-file', str(self.key), '--skills-dir', str(self.root / 'skills'),
                               *args], env=env, capture_output=True, text=True)

    def installed(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = tomllib.loads(self.config.read_text())
        self.assertNotIn('profiles', cfg)
        cfg['profiles'] = {p.name.removesuffix('.config.toml'): tomllib.loads(p.read_text())
                           for p in self.home.glob('*.config.toml')}
        return cfg

    def test_preserves_defaults_and_uses_command_auth_and_model_contexts(self):
        cfg = self.installed()
        self.assertEqual(cfg['model'], 'existing-gpt')
        self.assertEqual(cfg['mcp_servers']['keep']['command'], 'keep')
        profile = cfg['profiles']['qcm']
        self.assertNotIn('model_context_window', profile)
        provider = cfg['model_providers'][profile['model_provider']]
        self.assertEqual(provider['wire_api'], 'responses')
        self.assertEqual(provider['auth']['command'], 'cat')
        self.assertEqual(provider['auth']['args'], [str(self.home / 'quickclevermodel/qcm/api-key')])
        catalog = json.loads(Path(profile['model_catalog_json']).read_text())
        self.assertEqual([m['context_window'] for m in catalog['models']], [8192, 32768])
        self.assertIn('qcm-large-model', cfg['profiles'])
        self.assertTrue(all(r[1] == 'Bearer ' + TOKEN for r in self.requests))
        smoke = next(r[2] for r in self.requests if r[0] == '/v1/responses')
        self.assertLessEqual(smoke['max_output_tokens'], 1024)

    def test_permissions_skill_marker_backup_and_no_secret_output(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        private = self.home / 'quickclevermodel/qcm'
        self.assertEqual(private.stat().st_mode & 0o777, 0o700)
        self.assertEqual((private / 'api-key').stat().st_mode & 0o777, 0o600)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        self.assertNotIn(TOKEN, (private / 'models.json').read_text() + self.config.read_text())
        skill = self.root / 'skills/quickclevermodel-inference'
        self.assertTrue((skill / 'scripts/inference.py').is_file())
        self.assertEqual((skill / 'client-path').read_text().strip(), str(private / 'client.json'))
        self.assertTrue(any(p.read_text() == self.original for p in self.home.glob('config.toml.bak*')))

    def test_rerun_is_idempotent_and_updates_models(self):
        self.installed()
        before = self.config.read_text()
        self.installed()
        self.assertEqual(before, self.config.read_text())
        self.models = [{'id': 'new', 'context_window': 4096}]
        cfg = self.installed()
        self.assertEqual(cfg['profiles']['qcm']['model'], 'new')
        self.assertNotIn('qcm-small', cfg['profiles'])

    def test_utf8_files_under_ascii_locale(self):
        self.original += '# Configuration café\n'
        self.config.write_text(self.original, encoding='utf-8')
        self.key.write_text('\u2003' + TOKEN + '\u2003', encoding='utf-8')
        dotenv = self.root / 'env'
        dotenv.write_text('# Réglages\n', encoding='utf-8')
        env = {'LC_ALL': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0',
               'BONSAI_ENV_FILE': str(dotenv)}
        for _ in range(2):
            result = self.run_setup(extra_env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(self.config.read_text(encoding='utf-8').startswith(self.original))

    def test_check_does_not_write(self):
        result = self.run_setup('--check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.config.read_text(), self.original)
        self.assertFalse((self.home / 'quickclevermodel').exists())
        self.assertFalse((self.root / 'skills').exists())

    def test_auth_failure_is_safe_and_does_not_write(self):
        self.status = 401
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        self.assertEqual(self.config.read_text(), self.original)

    def test_smoke_requires_final_text(self):
        self.response = {'output': []}
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.assertEqual(self.config.read_text(), self.original)

    def test_missing_context_fails_or_accepts_explicit_override(self):
        self.models = [{'id': 'unknown'}]
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.assertEqual(self.run_setup('--context-window', '16384').returncode, 0)

    def test_live_preset_context(self):
        self.models = [{'id': 'preset', 'status': {'args': ['--ctx-size', '12288', '--parallel', '1']}}]
        cfg = self.installed()
        catalog = json.loads(Path(cfg['profiles']['qcm']['model_catalog_json']).read_text())
        self.assertEqual(catalog['models'][0]['context_window'], 12288)

    def test_training_limit_is_not_a_served_context(self):
        self.models = [{'id': 'model', 'meta': {'n_ctx_train': 999999}}]
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.props = {'default_generation_settings': {'n_ctx': 4096}}
        cfg = self.installed()
        catalog = json.loads(Path(cfg['profiles']['qcm']['model_catalog_json']).read_text())
        self.assertEqual(catalog['models'][0]['context_window'], 4096)

    def test_live_preset_beats_metadata(self):
        self.models = [{'id': 'model', 'context_window': 999999, 'status': {'args': ['--ctx-size', '4096', '--parallel', '1']}}]
        cfg = self.installed()
        catalog = json.loads(Path(cfg['profiles']['qcm']['model_catalog_json']).read_text())
        self.assertEqual(catalog['models'][0]['context_window'], 4096)

    def test_live_per_request_limit_and_props_cap_preset_allocation(self):
        self.models = [{'id': 'model', 'status': {'value': 'loaded', 'args': ['--ctx-size', '8192', '--parallel', '4']},
                        'default_generation_settings': {'n_ctx': 2048}}]
        self.props = {'default_generation_settings': {'n_ctx': 1024}}
        cfg = self.installed()
        catalog = json.loads(Path(cfg['profiles']['qcm']['model_catalog_json']).read_text())
        self.assertEqual(catalog['models'][0]['context_window'], 1024)
        self.assertTrue(any(path.startswith('/props?model=') for path, _, _ in self.requests))

    def test_unloaded_presets_do_not_trigger_runtime_property_autoload(self):
        self.models = [{'id': 'cold', 'status': {'value': 'unloaded', 'args': ['--ctx-size', '8192', '--parallel', '2']}}]
        cfg = self.installed()
        self.assertFalse(any(path.startswith('/props') for path, _, _ in self.requests))
        catalog = json.loads(Path(cfg['profiles']['qcm']['model_catalog_json']).read_text())
        self.assertEqual(catalog['models'][0]['context_window'], 4096)

    def test_loaded_properties_disable_autoload_and_remain_authoritative(self):
        self.models = [{'id': 'ready', 'status': {'value': 'loaded', 'args': ['--ctx-size', '8192', '--parallel', '1']}}]
        self.props = {'default_generation_settings': {'n_ctx': 2048}}
        cfg = self.installed()
        paths = [path for path, _, _ in self.requests if path.startswith('/props')]
        self.assertEqual(len(paths), 1)
        self.assertIn('autoload=false', paths[0])
        catalog = json.loads(Path(cfg['profiles']['qcm']['model_catalog_json']).read_text())
        self.assertEqual(catalog['models'][0]['context_window'], 2048)

    def test_default_prefers_loaded_model_but_explicit_model_wins(self):
        self.models[0]['status'] = {'value': 'unloaded'}
        self.models[1]['status'] = {'value': 'loaded'}
        cfg = self.installed()
        self.assertEqual(cfg['profiles']['qcm']['model'], 'large/model')
        self.assertEqual(self.run_setup('--model', 'small').returncode, 0)
        self.assertEqual(tomllib.loads((self.home / 'qcm.config.toml').read_text())['model'], 'small')

    def test_cold_smoke_has_finite_generation_budget_and_checks_completion(self):
        with mock.patch.object(SETUP, 'request', return_value=self.response) as probe:
            SETUP.smoke_test(self.url + '/v1', TOKEN, 'cold')
        call = probe.call_args
        self.assertEqual(call.kwargs['timeout'], 120)
        self.assertGreaterEqual(call.args[2]['max_output_tokens'], 256)
        self.assertLessEqual(call.args[2]['max_output_tokens'], 1024)
        for result in ({'output': [{'type': 'reasoning', 'content': [{'type': 'output_text', 'text': 'thinking'}]}]},
                       {**self.response, 'status': 'incomplete'}, {**self.response, 'status': 'failed'}):
            with mock.patch.object(SETUP, 'request', return_value=result), self.assertRaises(SETUP.SetupError):
                SETUP.smoke_test(self.url + '/v1', TOKEN, 'cold')

    def test_parallel_allocation_and_per_slot_cap(self):
        self.assertEqual(SETUP.context({'args': ['--ctx-size', '8192', '--parallel', '4']}), 2048)
        self.assertEqual(SETUP.context({'args': ['--ctx-size=8192', '--parallel=4', '--kv-unified-per-slot=1024']}), 1024)
        self.assertEqual(SETUP.context({'args': ['--ctx-size', '8192', '--parallel', '0']}), None)

    def test_setup_bypasses_proxy_environment(self):
        seen = []
        def opened(handler, req):
            seen.append(req.host)
            response = io.BytesIO(b'{}')
            response.code = 200
            response.msg = 'OK'
            response.info = lambda: {}
            return response
        with mock.patch.dict(os.environ, {'http_proxy': 'http://proxy.invalid:8080', 'HTTP_PROXY': 'http://proxy.invalid:8080', 'no_proxy': '', 'NO_PROXY': ''}), \
                mock.patch('urllib.request.HTTPHandler.http_open', opened):
            SETUP.request('http://chosen.invalid/v1/models', TOKEN)
        self.assertEqual(seen, ['chosen.invalid'])

    def test_drip_response_has_total_deadline_and_safe_failure(self):
        self.drip = True
        started = time.monotonic()
        with self.assertRaises(SETUP.SetupError) as caught:
            SETUP.request(self.url + '/v1/models', TOKEN, timeout=0.15)
        self.assertLess(time.monotonic() - started, 0.6)
        self.assertNotIn(TOKEN, str(caught.exception))

    def test_prints_generated_profile_commands(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('codex --profile qcm-small', result.stdout)
        self.assertIn('codex --profile qcm-large-model', result.stdout)

    def test_unmanaged_profile_provider_skill_and_data_conflicts(self):
        for text in ['[profiles.qcm]\nmodel="mine"\n', '[model_providers.qcm]\nname="mine"\n']:
            self.config.write_text(self.original + text)
            self.assertNotEqual(self.run_setup().returncode, 0)
            self.assertEqual(self.config.read_text(), self.original + text)
        self.config.write_text(self.original)
        skill = self.root / 'skills/quickclevermodel-inference'
        skill.mkdir(parents=True)
        (skill / 'SKILL.md').write_text('mine')
        self.assertNotEqual(self.run_setup().returncode, 0)

    def test_modified_managed_config_is_not_overwritten(self):
        self.installed()
        profile = self.home / 'qcm.config.toml'
        profile.write_text(profile.read_text().replace('model = "small"', 'model = "mine"'))
        self.assertNotEqual(self.run_setup().returncode, 0)

    def test_unmanaged_profile_file_is_preserved(self):
        profile = self.home / 'qcm.config.toml'
        profile.write_text('model = "mine"\n')
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.assertEqual(profile.read_text(), 'model = "mine"\n')

    @unittest.skipUnless(shutil.which('codex'), 'Codex executable unavailable')
    def test_installed_codex_accepts_profile_and_catalog(self):
        executable = shutil.which('codex')
        self.codex.unlink()
        self.codex.symlink_to(executable)
        self.installed()
        env = {**os.environ, 'CODEX_HOME': str(self.home)}
        for command in ([executable, '--profile', 'qcm', 'mcp', 'list'],
                        [executable, 'debug', 'models', '-c', 'model_catalog_json=' + json.dumps(str(self.home / 'quickclevermodel/qcm/models.json'))]):
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_sanitization_collision_rejected(self):
        self.models = [{'id': 'a/b', 'context_window': 8192}, {'id': 'a-b', 'context_window': 8192}]
        self.assertNotEqual(self.run_setup().returncode, 0)

    def test_literal_dotenv_and_no_bind_address_fallback(self):
        env = self.root / 'env'
        env.write_text('BONSAI_SERVER_URL=' + self.url + '\nBONSAI_API_KEY=' + TOKEN + '\nBONSAI_HOST=0.0.0.0\nUNUSED=$(touch should-not-exist)\n')
        child_env = {k: v for k, v in os.environ.items() if not k.startswith(('BONSAI_', 'QCM_'))}
        child_env.update(CODEX_HOME=str(self.home), BONSAI_ENV_FILE=str(env),
                         PATH=str(self.bin) + os.pathsep + child_env['PATH'])
        result = subprocess.run(['bash', str(SCRIPT), '--check', '--skills-dir', str(self.root / 'skills')],
                                capture_output=True, text=True, env=child_env, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'should-not-exist').exists())

    def test_skill_update_and_multiple_clients_keep_latest_default(self):
        self.installed()
        for name in ('other', 'qcm', 'other'):
            result = self.run_setup('--name', name)
            self.assertEqual(result.returncode, 0, result.stderr)
        marker = self.root / 'skills/quickclevermodel-inference/client-path'
        self.assertIn('/other/client.json', marker.read_text())
        self.assertFalse(list(marker.parent.rglob('*.bak-*')))

    def test_redirect_never_receives_credentials(self):
        # Redirects are errors, including when a gateway points at another host.
        self.status = 302
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.requests), 1)

    def test_unsupported_bundled_schema_fails_without_changes(self):
        self.codex.write_text('#!/bin/sh\necho \'{"models":[{"slug":"future","tool_mode":"code_mode_only","context_window":100000,"use_responses_lite":true}]}\'\n')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('supported Responses template', result.stderr)
        self.assertEqual(self.config.read_text(), self.original)

    def test_unmanaged_data_directory_and_symlink_rejected(self):
        data = self.home / 'quickclevermodel/qcm'
        data.mkdir(parents=True)
        (data / 'client.json').write_text('mine')
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.assertEqual((data / 'client.json').read_text(), 'mine')

    def test_key_echo_in_model_id_is_not_written_or_printed(self):
        self.models = [{'id': TOKEN, 'context_window': 8192}]
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        self.assertEqual(self.config.read_text(), self.original)

    def test_managed_skill_edit_is_preserved(self):
        self.installed()
        skill = self.root / 'skills/quickclevermodel-inference/SKILL.md'
        skill.write_text('user edit')
        self.assertNotEqual(self.run_setup().returncode, 0)
        self.assertEqual(skill.read_text(), 'user edit')


if __name__ == '__main__':
    unittest.main()
