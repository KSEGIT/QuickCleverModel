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
import hashlib
import os
import re
import struct
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "bonsai-chat-template.jinja")
MODELS_INI = os.path.join(ROOT, "models.ini.in")
DOCKERFILE = os.path.join(ROOT, "docker", "Dockerfile")

# The unmodified template, read out of the GGUF metadata key
# tokenizer.chat_template. Kept so the parity tests below can prove the claim
# models.ini.in makes: our copy changes multi-system handling and nothing else.
# Refresh it if the weights ship a new template.
GGUF_TEMPLATE = os.path.join(ROOT, "tests", "fixtures", "gguf-chat-template.jinja")

# The exact call the GGUF template makes. Its presence anywhere in our copy
# means the fix has been reverted or a template refresh overwrote it.
SYSTEM_POSITION_RAISE = "System message must be at the beginning."
QWEN_TEMPLATE = os.path.join(ROOT, "qwen3.5-chat-template.jinja")
QWEN_NATIVE_SHA256 = "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67"
QWEN_PATCHED_SHA256 = "25f597df3371a86717074a45f77aba1cc90ec83917a28c495df9c6b00baf91e9"

try:
    import jinja2
except ImportError:  # pragma: no cover - exercised only on a bare interpreter
    jinja2 = None

# Locally a missing jinja2 is a skip. On CI it is a failure: the workflow
# installs jinja2 precisely so the render tests run, and a skip there would
# report green while testing nothing. The two steps are independent, so this
# is what actually closes the gap.
ON_CI = os.environ.get("CI") == "true"


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def parse_models_ini(text):
    """Section name -> {key: value}, comments dropped.

    Splitting on the literal "[*]" is not good enough: the file's own header
    comment mentions "[*]" and "[section]", so a naive split lands inside the
    header and inspects a comment instead of the globals block. That is not
    hypothetical -- it silently made this file's first draft pass.
    """
    sections, current = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue
        header = re.fullmatch(r"\[(.+)\]", stripped)
        if header:
            current = header.group(1)
            sections.setdefault(current, {})
        elif current is not None and "=" in stripped:
            key, _, value = stripped.partition("=")
            sections[current][key.strip()] = value.strip()
    return sections


def compile_template(source):
    """Compile the way the server does.

    The fork hardcodes lstrip_blocks and trim_blocks to true for chat templates
    (common/jinja/lexer.cpp); jinja2 defaults both to false. Every tag in this
    template carries an explicit '-', so the two agree today -- but a future
    edit that drops one would only show up if the test matches production.
    """
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise RuntimeError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


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
        # Match the COPY itself: a bare mention would also match a comment,
        # leaving chat-template-file pointing at a path that does not exist.
        self.assertRegex(
            read(DOCKERFILE),
            r"(?m)^COPY .*\bbonsai-chat-template\.jinja\b.*\s/app/\s*$",
            "the Dockerfile must COPY the template into /app, or the container "
            "starts with chat-template-file pointing at nothing",
        )

    def test_every_bonsai_section_sets_the_template(self):
        """Not [*]: that would cascade Bonsai's ChatML onto a future model."""
        sections = parse_models_ini(read(MODELS_INI))
        models = {name: keys for name, keys in sections.items() if name.startswith("bonsai-")}
        self.assertEqual(
            3, len(models), f"expected three model sections, found {sorted(models)}"
        )
        for name, keys in models.items():
            self.assertEqual(
                "@ROOT@/bonsai-chat-template.jinja",
                keys.get("chat-template-file"),
                f"[{name}] must set chat-template-file itself",
            )

    def test_the_globals_block_does_not_set_the_template(self):
        self.assertNotIn(
            "chat-template-file",
            parse_models_ini(read(MODELS_INI)).get("*", {}),
            "chat-template-file in [*] cascades onto every future section, "
            "overriding that model's own template with Bonsai's ChatML",
        )

    def test_text_preset_has_its_own_context_knob(self):
        """The text alias keeps a separate context knob after the runtime upgrade."""
        sections = parse_models_ini(read(MODELS_INI))
        self.assertEqual(
            "@CTX_TEXT@",
            sections["bonsai-27b-ternary-text"].get("c"),
            "the text preset must take its context from BONSAI_CTX_TEXT",
        )
        for vision in ("bonsai-27b-ternary", "bonsai-27b-1bit"):
            self.assertEqual(
                "@CTX@",
                sections[vision].get("c"),
                f"[{vision}] must retain BONSAI_CTX without inheriting the text "
                "preset's context",
            )

    def test_start_server_substitutes_the_text_context(self):
        """A placeholder with no sed rule would reach llama-server verbatim."""
        script = read(os.path.join(ROOT, "start-server.sh"))
        self.assertIn("@CTX_TEXT@|$CTX_TEXT", script)
        self.assertIn('CTX_TEXT="${BONSAI_CTX_TEXT:-$CTX}"', script)

    def test_the_raise_is_gone(self):
        # assertNotIn would dump all 7 KB of template into the failure report.
        self.assertFalse(
            SYSTEM_POSITION_RAISE in read(TEMPLATE),
            "the system-position raise is back -- Codex will 400 again",
        )


class Qwen35Wiring(unittest.TestCase):
    """The Qwen override changes one guard, not its tool or thinking grammar."""

    def test_template_is_exactly_pinned_and_shipped(self):
        source = read(QWEN_TEMPLATE)
        self.assertEqual(QWEN_PATCHED_SHA256, hashlib.sha256(source.encode()).hexdigest())
        self.assertNotIn(SYSTEM_POSITION_RAISE, source)
        self.assertIn("tool_call.arguments is mapping", source)
        self.assertIn("enable_thinking is defined and enable_thinking is true", source)
        self.assertRegex(read(DOCKERFILE),
            r"(?m)^COPY .*\bqwen3\.5-chat-template\.jinja\b.*\s/app/\s*$")

    def test_native_gguf_diff_is_only_late_system_guard_when_weights_available(self):
        paths = (
            os.path.join(ROOT, "models", "Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
            os.path.join(ROOT, "models", "Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
        )
        available = [path for path in paths if os.path.isfile(path)]
        if not available:
            self.skipTest("Qwen3.5 GGUF weights are not downloaded in this checkout")
        old = "            {{- raise_exception('System message must be at the beginning.') }}"
        new = ("            {%- set content = render_content(message.content, false, true)|trim %}\n"
               "            {%- if content %}\n"
               "                {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
               "            {%- endif %}")
        for path in available:
            with self.subTest(path=path):
                native = gguf_chat_template(path)
                self.assertEqual(QWEN_NATIVE_SHA256, hashlib.sha256(native.encode()).hexdigest())
                self.assertEqual(1, native.count(old))
                self.assertEqual(read(QWEN_TEMPLATE), native.replace(old, new))


@unittest.skipIf(jinja2 is None and not ON_CI, "jinja2 not installed")
class Qwen35Rendering(unittest.TestCase):
    def test_multiple_system_messages_render(self):
        rendered = compile_template(read(QWEN_TEMPLATE)).render(
            messages=[{"role": "system", "content": "First instruction"},
                      {"role": "system", "content": "Second instruction"},
                      {"role": "user", "content": "Hello"}],
            tools=[], add_generation_prompt=True, enable_thinking=False)
        self.assertIn("<|im_start|>system\nFirst instruction<|im_end|>", rendered)
        self.assertIn("<|im_start|>system\nSecond instruction<|im_end|>", rendered)
        self.assertIn("<|im_start|>user\nHello<|im_end|>", rendered)


@unittest.skipIf(jinja2 is None and not ON_CI, "jinja2 not installed")
class Rendering(unittest.TestCase):
    """Needs jinja2. CI installs it so these do not silently skip."""

    def setUp(self):
        self.template = compile_template(read(TEMPLATE))
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
                        {"role": "system", "content": "ONLY-SYSTEM-MARKER"},
                        {"role": "user", "content": "hi"},
                    ],
                    tools,
                )
                self.assertEqual(1, out.count("<|im_start|>system"))
                self.assertIn("ONLY-SYSTEM-MARKER", out)

    def test_no_system_message_still_renders(self):
        out = self.render([{"role": "user", "content": "hi"}])
        self.assertIn("<|im_start|>user", out)

    def test_multi_turn_conversation_renders(self):
        out = self.render(
            [
                {"role": "system", "content": "SYSTEM-MARKER"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
            self.tools,
        )
        for text in ("SYSTEM-MARKER", "one", "two", "three"):
            self.assertIn(text, out)


@unittest.skipIf(jinja2 is None and not ON_CI, "jinja2 not installed")
class ParityWithTheGgufTemplate(unittest.TestCase):
    """Our copy must render byte-identically except for multiple system messages.

    Substring assertions would miss a lost newline or a dropped control token in
    the tool_calls and tool branches, which the fix does not touch but a future
    edit could. Comparing whole renders against the GGUF original catches that.
    """

    def setUp(self):
        self.ours = compile_template(read(TEMPLATE))
        self.gguf = compile_template(read(GGUF_TEMPLATE))

    # Every branch of the template that the fix does not touch.
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                },
            },
        }
    ]
    SHAPES = {
        "system and user": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "hi"},
        ],
        "no system message": [{"role": "user", "content": "hi"}],
        "multi turn": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
        "assistant makes a tool call": [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "exec_command",
                            "arguments": {"cmd": "ls -la"},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "a.txt\nb.txt"},
            {"role": "user", "content": "thanks"},
        ],
        "two tool responses in a row": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {"function": {"name": "exec_command", "arguments": {"cmd": "a"}}},
                    {"function": {"name": "exec_command", "arguments": {"cmd": "b"}}},
                ],
            },
            {"role": "tool", "content": "out a"},
            {"role": "tool", "content": "out b"},
            {"role": "user", "content": "done?"},
        ],
        "assistant with reasoning content": [
            {"role": "user", "content": "think"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "some thinking",
            },
            {"role": "user", "content": "again"},
        ],
    }

    def test_renders_are_byte_identical(self):
        for name, messages in self.SHAPES.items():
            for tools in (None, self.TOOLS):
                with self.subTest(shape=name, tools=bool(tools)):
                    kwargs = dict(
                        messages=messages, tools=tools, add_generation_prompt=True
                    )
                    self.assertEqual(
                        self.gguf.render(**kwargs),
                        self.ours.render(**kwargs),
                        f"{name} renders differently from the GGUF template",
                    )

    def test_the_fixture_is_the_unpatched_template(self):
        """A stale fixture would make the parity test meaningless."""
        self.assertIn(SYSTEM_POSITION_RAISE, read(GGUF_TEMPLATE))

    def test_only_multi_system_differs(self):
        """The one shape that must NOT match: the GGUF template raises."""
        messages = [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "hi"},
        ]
        with self.assertRaises(RuntimeError):
            self.gguf.render(messages=messages, add_generation_prompt=True)
        self.ours.render(messages=messages, add_generation_prompt=True)


@unittest.skipIf(jinja2 is None and not ON_CI, "jinja2 not installed")
class SystemBranchGuards(unittest.TestCase):
    """The new branch must keep the guards the first system message has.

    The fix renders later system messages instead of raising. It must render
    them the way messages[0] is rendered -- render_content(..., false, true) --
    or it quietly drops the media guard and the vision counter goes wrong.
    """

    def setUp(self):
        self.template = compile_template(read(TEMPLATE))

    def render(self, messages):
        return self.template.render(messages=messages, add_generation_prompt=True)

    def test_images_are_still_refused_in_a_later_system_message(self):
        with self.assertRaises(RuntimeError) as caught:
            self.render(
                [
                    {"role": "system", "content": "first"},
                    {
                        "role": "system",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    },
                    {"role": "user", "content": "hi"},
                ]
            )
        self.assertIn("System message cannot contain images", str(caught.exception))

    def test_videos_are_still_refused_in_a_later_system_message(self):
        with self.assertRaises(RuntimeError):
            self.render(
                [
                    {"role": "system", "content": "first"},
                    {"role": "system", "content": [{"type": "video", "video": "x"}]},
                    {"role": "user", "content": "hi"},
                ]
            )

    def test_an_empty_later_system_message_emits_nothing(self):
        """Codex sends its developer item every turn; empty must not add a block.

        The tools header guards content the same way.
        """
        out = self.render(
            [
                {"role": "system", "content": "first"},
                {"role": "system", "content": "   "},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(1, out.count("<|im_start|>system"))


@unittest.skipIf(jinja2 is None and not ON_CI, "jinja2 not installed")
class RaisePaths(unittest.TestCase):
    """The template's raises must still fire.

    A raise on the wrong input is what broke Codex, so a template edit that
    turns a raise into silent empty output is exactly the regression to catch.
    """

    def setUp(self):
        self.template = compile_template(read(TEMPLATE))

    def test_no_messages(self):
        with self.assertRaises(RuntimeError):
            self.template.render(messages=[], add_generation_prompt=True)

    def test_unexpected_role(self):
        with self.assertRaises(RuntimeError):
            self.template.render(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "wizard", "content": "?"},
                ],
                add_generation_prompt=True,
            )

    def test_no_user_query(self):
        with self.assertRaises(RuntimeError):
            self.template.render(
                messages=[{"role": "user", "content": "<tool_response>x</tool_response>"}],
                add_generation_prompt=True,
            )

    def test_without_a_generation_prompt(self):
        out = self.template.render(
            messages=[{"role": "user", "content": "hi"}], add_generation_prompt=False
        )
        self.assertNotIn("<|im_start|>assistant", out)


def gguf_chat_template(path):
    """Read tokenizer.chat_template out of a GGUF, stdlib only.

    Metadata sits at the head of the file, so this reads a few MB at most.
    """
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as fh:

        def take(n):
            data = fh.read(n)
            if len(data) != n:
                raise EOFError("truncated GGUF")
            return data

        def u32():
            return struct.unpack("<I", take(4))[0]

        def u64():
            return struct.unpack("<Q", take(8))[0]

        def skip(kind):
            if kind == 8:
                fh.seek(u64(), 1)
            elif kind == 9:
                element, count = u32(), u64()
                if element in (8, 9):
                    for _ in range(count):
                        skip(element)
                else:
                    fh.seek(sizes[element] * count, 1)
            else:
                fh.seek(sizes[kind], 1)

        if take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        u32(), u64()
        for _ in range(u64()):
            key = take(u64()).decode("utf-8", "replace")
            kind = u32()
            if key == "tokenizer.chat_template":
                return take(u64()).decode("utf-8", "replace")
            skip(kind)
    return None


# Only runs where the weights are: a dev box or the inference host, not CI.
# models/ is gitignored, so a git worktree does not have it -- point
# BONSAI_MODELS_DIR at the main checkout's models/ to run this from one.
TERNARY_MODELS_DIR = os.path.join(
    os.environ.get("BONSAI_MODELS_DIR") or os.path.join(ROOT, "models"),
    "Ternary-Bonsai-27B-gguf",
)
PQ2_WEIGHTS = os.path.join(TERNARY_MODELS_DIR, "Ternary-Bonsai-27B-PQ2_0.gguf")
LEGACY_WEIGHTS = os.path.join(TERNARY_MODELS_DIR, "Ternary-Bonsai-27B-Q2_0.gguf")
WEIGHTS = PQ2_WEIGHTS if os.path.isfile(PQ2_WEIGHTS) else LEGACY_WEIGHTS


@unittest.skipUnless(os.path.isfile(WEIGHTS), "weights not present")
class FixtureMatchesTheWeights(unittest.TestCase):
    """Catch the vendored copy drifting away from the model it came from.

    chat-template-file always beats the GGUF's own metadata, so if prism-ml
    ships a corrected template and fetch-models.sh pulls it, our frozen copy
    keeps winning -- silently, forever, with a green suite. This is the only
    test that would notice.
    """

    def test_fixture_still_matches_the_gguf(self):
        baked = gguf_chat_template(WEIGHTS)
        self.assertIsNotNone(baked, "GGUF has no tokenizer.chat_template")
        self.assertEqual(
            baked.rstrip("\n"),
            read(GGUF_TEMPLATE).rstrip("\n"),
            "the weights ship a different template than tests/fixtures records; "
            "re-extract it and rebase bonsai-chat-template.jinja on the new one",
        )

    def test_only_the_system_branch_differs(self):
        """Everything outside the system branch must be the GGUF's text verbatim."""
        start = '{%- if message.role == "system" %}'
        end = '{%- elif message.role == "user" %}'
        for label, text in (
            ("gguf", gguf_chat_template(WEIGHTS)),
            ("ours", read(TEMPLATE)),
        ):
            self.assertEqual(1, text.count(start), f"{label}: system branch not found")
            self.assertEqual(1, text.count(end), f"{label}: user branch not found")

        baked, ours = gguf_chat_template(WEIGHTS), read(TEMPLATE)
        self.assertEqual(
            baked.split(start)[0],
            ours.split(start)[0],
            "our copy diverges from the weights BEFORE the system branch",
        )
        self.assertEqual(
            baked.split(end, 1)[1].rstrip("\n"),
            ours.split(end, 1)[1].rstrip("\n"),
            "our copy diverges from the weights AFTER the system branch",
        )
        self.assertIn(SYSTEM_POSITION_RAISE, baked)
        self.assertNotIn(SYSTEM_POSITION_RAISE, ours)


if __name__ == "__main__":
    unittest.main()
