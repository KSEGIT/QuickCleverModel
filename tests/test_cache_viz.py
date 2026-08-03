"""Tests for cache-viz.py's log parser.

stdlib unittest, not pytest — cache-viz.py promises "Stdlib only; no install
step" and its tests should not break that.

Run: python3 -m unittest discover -s tests -v

Every fixture below is a VERBATIM line from run/logs/llama.log captured while
the router had both 27B quantizations loaded. The `[port]` prefix and the
`task 0` collision between the two children are real, not invented.
"""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cache_viz():
    """cache-viz.py has a dash in its name, so it is not directly importable."""
    spec = importlib.util.spec_from_file_location(
        "cache_viz", os.path.join(ROOT, "cache-viz.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.cv = load_cache_viz()
        self.cv._requests.clear()
        self.cv._pending.clear()
        self.cv._ports.clear()

    def feed(self, *lines):
        for line in lines:
            self.cv._parse_line(line)

    def test_learns_port_to_model_mapping(self):
        self.feed(
            "0.00.229.322 I srv          load: spawning server instance with name=bonsai-27b-ternary on port 53204",
            "19.01.821.918 I srv          load: spawning server instance with name=bonsai-27b-1bit on port 55152",
        )
        self.assertEqual(self.cv._ports.get("53204"), "bonsai-27b-ternary")
        self.assertEqual(self.cv._ports.get("55152"), "bonsai-27b-1bit")

    def test_two_children_reusing_task_0_do_not_merge(self):
        """The real collision: both children numbered their first request task 0."""
        self.feed(
            "0.00.229.322 I srv          load: spawning server instance with name=bonsai-27b-ternary on port 53204",
            "19.01.821.918 I srv          load: spawning server instance with name=bonsai-27b-1bit on port 55152",
            "[53204] 18.58.572.485 I slot print_timing: id  0 | task 0 | prompt eval time =   13345.37 ms /    16 tokens (  834.09 ms per token,     1.20 tokens per second)",
            "[53204] 18.58.572.490 I slot print_timing: id  0 | task 0 |        eval time =  107418.84 ms /   300 tokens (  358.06 ms per token,     2.79 tokens per second)",
            "[55152] 1.28.167.508 I slot print_timing: id  0 | task 0 | prompt eval time =    1545.78 ms /    16 tokens (   96.61 ms per token,    10.35 tokens per second)",
            "[55152] 1.28.167.514 I slot print_timing: id  0 | task 0 |        eval time =   75038.52 ms /   300 tokens (  250.13 ms per token,     4.00 tokens per second)",
        )
        self.assertEqual(len(self.cv._requests), 2)
        by_model = {r["model"]: r for r in self.cv._requests}
        self.assertEqual(set(by_model), {"bonsai-27b-ternary", "bonsai-27b-1bit"})
        self.assertEqual(by_model["bonsai-27b-ternary"]["gen_tok_s"], 2.79)
        self.assertEqual(by_model["bonsai-27b-1bit"]["gen_tok_s"], 4.00)
        self.assertEqual(by_model["bonsai-27b-ternary"]["new"], 16)
        self.assertEqual(by_model["bonsai-27b-1bit"]["new"], 16)

    def test_prompt_eval_line_does_not_complete_the_record(self):
        """'prompt eval time' also ends in 'tokens per second' and must not be
        mistaken for the generation timing that closes a request."""
        self.feed(
            "0.00.229.322 I srv          load: spawning server instance with name=bonsai-27b-ternary on port 53204",
            "[53204] 18.58.572.485 I slot print_timing: id  0 | task 0 | prompt eval time =   13345.37 ms /    16 tokens (  834.09 ms per token,     1.20 tokens per second)",
        )
        self.assertEqual(len(self.cv._requests), 0)
        self.assertIn(("53204", "0"), self.cv._pending)

    def test_reused_tokens_attributed_to_the_right_child(self):
        self.feed(
            "0.00.229.322 I srv          load: spawning server instance with name=bonsai-27b-ternary on port 53204",
            "19.01.821.918 I srv          load: spawning server instance with name=bonsai-27b-1bit on port 55152",
            "[53204] 3.01.421.640 W slot update_slots: id  0 | task 0 | restored context checkpoint (pos_min = 484, pos_max = 484, n_tokens = 485, n_past = 485, size = 149.626 MiB)",
            "[53204] 3.01.421.700 I slot print_timing: id  0 | task 0 | prompt eval time =      10.00 ms /     3 tokens (    3.33 ms per token,   300.00 tokens per second)",
            "[53204] 3.01.421.800 I slot print_timing: id  0 | task 0 |        eval time =     100.00 ms /     5 tokens (   20.00 ms per token,     5.00 tokens per second)",
            "[55152] 3.02.000.000 I slot print_timing: id  0 | task 0 | prompt eval time =      90.00 ms /    40 tokens (    2.25 ms per token,   444.00 tokens per second)",
            "[55152] 3.02.000.100 I slot print_timing: id  0 | task 0 |        eval time =     100.00 ms /     5 tokens (   20.00 ms per token,     5.00 tokens per second)",
        )
        by_model = {r["model"]: r for r in self.cv._requests}
        self.assertEqual(by_model["bonsai-27b-ternary"]["reused"], 485)
        self.assertEqual(by_model["bonsai-27b-1bit"]["reused"], 0)

    def test_slot_init_lines_with_task_minus_one_are_ignored(self):
        """'task -1' marks slot initialisation, not a request. It must not
        create a record that lingers in _pending forever."""
        self.feed(
            "[55152] 0.11.382.279 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 32768",
        )
        self.assertEqual(len(self.cv._pending), 0)
        self.assertEqual(len(self.cv._requests), 0)

    def test_unprefixed_lines_still_parse(self):
        """Single-model logs and pre-router history must keep working."""
        self.feed(
            "0.01.0 I slot print_timing: id  0 | task 7 | prompt eval time =     500.00 ms /   100 tokens (    5.00 ms per token,   200.00 tokens per second)",
            "0.01.1 I slot print_timing: id  0 | task 7 |        eval time =    1000.00 ms /    10 tokens (  100.00 ms per token,    12.90 tokens per second)",
        )
        self.assertEqual(len(self.cv._requests), 1)
        self.assertEqual(self.cv._requests[0]["new"], 100)
        self.assertEqual(self.cv._requests[0]["gen_tok_s"], 12.9)
        self.assertEqual(self.cv._requests[0]["model"], "single")

    def test_unknown_port_falls_back_to_the_port_number(self):
        """A child whose spawn line scrolled out of the log is still distinct."""
        self.feed(
            "[41007] 0.01.0 I slot print_timing: id  0 | task 0 | prompt eval time =     500.00 ms /   100 tokens (    5.00 ms per token,   200.00 tokens per second)",
            "[41007] 0.01.1 I slot print_timing: id  0 | task 0 |        eval time =     100.00 ms /     5 tokens (   20.00 ms per token,     5.00 tokens per second)",
        )
        self.assertEqual(self.cv._requests[0]["model"], ":41007")

    def test_four_digit_port_is_right_aligned_by_the_router(self):
        """LOG("[%5d] ...") pads a 4-digit port with a leading space."""
        self.feed(
            "0.00.1 I srv          load: spawning server instance with name=tiny on port 8081",
            "[ 8081] 0.01.0 I slot print_timing: id  0 | task 0 | prompt eval time =     500.00 ms /   100 tokens (    5.00 ms per token,   200.00 tokens per second)",
            "[ 8081] 0.01.1 I slot print_timing: id  0 | task 0 |        eval time =     100.00 ms /     5 tokens (   20.00 ms per token,     5.00 tokens per second)",
        )
        self.assertEqual(self.cv._requests[0]["model"], "tiny")


if __name__ == "__main__":
    unittest.main()
