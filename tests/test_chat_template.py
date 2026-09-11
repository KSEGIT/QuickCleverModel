"""Tests for bonsai-chat-template.jinja: the multi-system-message fix.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

Run: python3 -m unittest discover -s tests -v

Background, because it is the whole point of these tests: the chat template
baked into the Bonsai GGUFs renders only messages[0] as the system block and
calls raise_exception('System message must be at the beginning.') for any
system message after it.

Codex sends two. One comes from the Responses API `instructions` field, the
other from a `developer` input item it emits on every request; llama-server
maps both to system messages. The result was a 400 on every single Codex
request -- "Unable to generate parser for this template" -- while the same
server answered opencode and plain curl fine. Sending either message alone
returns 200, which is what pinned the cause to the pair, not to tools or to
the Responses endpoint.

bonsai-chat-template.jinja is a copy of the GGUF template with that one raise
replaced by the same render the user branch uses, and models.ini.in points
llama-server at it with chat-template-file. These tests pin both halves: the
wiring (stdlib, always runs) and the rendering (needs jinja2).

The render tests use Python jinja2. llama.cpp renders with minja, a C++ subset,
so a green run here is necessary but not sufficient -- it proves the template
logic, not minja parity. Parity is checked against the live server after deploy.
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "bonsai-chat-template.jinja")
MODELS_INI = os.path.join(ROOT, "models.ini.in")
DOCKERFILE = os.path.join(ROOT, "docker", "Dockerfile")

# The exact call the GGUF template makes. Its presence anywhere in our copy
# means the fix has been reverted or a template refresh overwrote it.
SYSTEM_POSITION_RAISE = "System message must be at the beginning."

try:
    import jinja2
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    jinja2 = None


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class Wiring(unittest.TestCase):
    """Stdlib-only. These run everywhere, including a bare CI runner."""

    def test_template_file_ships(self):
        self.assertTrue(
            os.path.isfile(TEMPLATE),
            "bonsai-chat-template.jinja is missing; models.ini.in points at it "
            "and llama-server fails to start without it",
        )

    def test_models_ini_points_at_the_template(self):
        # @ROOT@ is start-server.sh's own directory: /app in the container,
        # the repo root for a native run. One path covers both.
        self.assertIn(
            "chat-template-file = @ROOT@/bonsai-chat-template.jinja",
            read(MODELS_INI),
            "models.ini.in must point llama-server at the patched template, "
            "otherwise --jinja silently falls back to the GGUF one",
        )

    def test_template_is_in_the_image(self):
        # A bind mount will not do: /app/models is read-only and holds weights.
        self.assertIn(
            "bonsai-chat-template.jinja",
            read(DOCKERFILE),
            "the Dockerfile must COPY the template into /app, or the container "
            "starts with chat-template-file pointing at nothing",
        )

    def test_the_raise_is_gone(self):
        # assertNotIn would dump all 7 KB of template into the failure report.
        self.assertFalse(
            SYSTEM_POSITION_RAISE in read(TEMPLATE),
            "the system-position raise is back -- Codex will 400 again",
        )


@unittest.skipIf(jinja2 is None, "jinja2 not installed")
class Rendering(unittest.TestCase):
    """Needs jinja2. CI installs it so these do not silently skip."""

    def setUp(self):
        env = jinja2.Environment(extensions=["jinja2.ext.do"])

        def raise_exception(message):
            raise RuntimeError(message)

        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(read(TEMPLATE))
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def render(self, messages, tools=None):
        return self.template.render(
            messages=messages, tools=tools, add_generation_prompt=True
        )

    def test_two_system_messages_render(self):
        """The exact shape Codex sends. This raised before the fix."""
        out = self.render(
            [
                {"role": "system", "content": "FROM-INSTRUCTIONS"},
                {"role": "system", "content": "FROM-DEVELOPER-ITEM"},
                {"role": "user", "content": "hi"},
            ],
            self.tools,
        )
        self.assertIn("FROM-INSTRUCTIONS", out)
        self.assertIn("FROM-DEVELOPER-ITEM", out)
        self.assertEqual(
            2,
            out.count("<|im_start|>system"),
            "both system messages must reach the model, not just messages[0]",
        )

    def test_escapes_are_real_newlines(self):
        r"""The patched line must emit newlines, not a literal \n.

        Writing the replacement through a shell heredoc once turned the \n
        escapes into literal backslash-n inside the Jinja string, which renders
        a template that looks right and tokenises wrong.
        """
        out = self.render(
            [
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertNotIn("\\n", out)
        self.assertIn("<|im_start|>system\nB<|im_end|>\n", out)

    def test_single_system_message_is_untouched(self):
        """The common case must not gain a second block from the new branch."""
        for tools in (None, self.tools):
            with self.subTest(tools=bool(tools)):
                out = self.render(
                    [
                        {"role": "system", "content": "S"},
                        {"role": "user", "content": "hi"},
                    ],
                    tools,
                )
                self.assertEqual(1, out.count("<|im_start|>system"))
                self.assertIn("S", out)

    def test_no_system_message_still_renders(self):
        out = self.render([{"role": "user", "content": "hi"}])
        self.assertIn("<|im_start|>user", out)

    def test_multi_turn_conversation_renders(self):
        out = self.render(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            self.tools,
        )
        for text in ("S", "one", "two", "three"):
            self.assertIn(text, out)


if __name__ == "__main__":
    unittest.main()
