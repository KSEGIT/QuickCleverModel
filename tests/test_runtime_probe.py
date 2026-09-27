"""Offline contract tests for the opt-in live runtime probe."""
import importlib.util
import argparse
import pathlib
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "runtime_probe", pathlib.Path(__file__).with_name("runtime_probe.py"))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class StreamTest(unittest.TestCase):
    def test_chat_tool_fragments_keep_text_and_merge_arguments(self):
        events = [
            {"choices": [{"delta": {"role": "assistant", "content": "Checking now.", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {
                    "name": "lookup_code", "arguments": '{"key":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"fixture"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        result = probe.chat_stream(events)
        self.assertEqual(result["content"], "Checking now.")
        self.assertEqual(probe.tool_arguments(result)["key"], "fixture")

    def test_chat_stream_rejects_missing_or_wrong_role(self):
        for delta in ({"content": "Hello"}, {"role": "user", "content": "Hello"}):
            with self.subTest(delta=delta), self.assertRaisesRegex(AssertionError, "role"):
                probe.chat_stream([{"choices": [{"delta": delta, "finish_reason": "stop"}]}])

    def test_chat_stream_rejects_missing_or_wrong_tool_type(self):
        for kind in (None, "custom"):
            call = {"index": 0, "id": "call1", "function": {
                "name": "lookup_code", "arguments": '{"key":"fixture"}'}}
            if kind is not None:
                call["type"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(AssertionError, "type"):
                probe.chat_stream([{"choices": [{"delta": {
                    "role": "assistant", "tool_calls": [call]}, "finish_reason": "tool_calls"}]}])

    def test_invalid_tool_arguments_fail(self):
        with self.assertRaises((ValueError, AssertionError)):
            probe.tool_arguments({"tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "lookup_code", "arguments": "not JSON"}}]})

    def test_wrong_function_is_not_a_success(self):
        with self.assertRaises(AssertionError):
            probe.tool_arguments({"tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "distractor_1", "arguments": '{"key":"fixture"}'}}]})

    def test_sse_multiline_and_done(self):
        events = list(probe.sse_events([
            b': keepalive\n', b'event: message\n', b'data: {"x":\n',
            b'data: 1}\n', b'\n', b'data: [DONE]\n', b'\n']))
        self.assertEqual(events, [{"x": 1}, {"_done": True}])

    def test_unterminated_sse_fails(self):
        with self.assertRaises(AssertionError):
            list(probe.sse_events([b'data: {"x":1}\n']))

    def test_api_error_event_fails(self):
        with self.assertRaises(AssertionError):
            probe.chat_stream([{"error": {"message": "parser failed"}}])

    def test_tool_matrix_includes_auto_and_required(self):
        cases = probe.tool_cases()
        for count in (1, 5, 24):
            for choice in ("auto", "required"):
                self.assertTrue(any(c["count"] == count and c["choice"] == choice
                                    and c["long"] for c in cases))
        self.assertTrue(any(c["text_first"] for c in cases))

    def test_focused_matrix_selects_exact_cases(self):
        cases = probe.tool_cases()
        self.assertEqual(probe.matrix_subset("0,20,47"), [cases[0], cases[20], cases[47]])
        for invalid in ("x", "1,1", "-1", "48"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probe.matrix_subset(invalid)


class APIContractTest(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(base_url="http://127.0.0.1:18080", model="fixture",
            key_env="UNSET_QCM_UNIT_KEY", label="unit", runtime="unit", context=8192,
            reasoning="default", reasoning_budget=128, max_tokens=512)
        self.probe = probe.Probe(self.args)

    def test_reasoning_budget_matches_pinned_server_field(self):
        self.args.reasoning = "bounded"
        body = self.probe.generation()
        self.assertEqual(body["reasoning_budget_tokens"], 128)
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.args.reasoning = "off"
        self.assertFalse(self.probe.generation()["chat_template_kwargs"]["enable_thinking"])

    def test_reasoning_mode_applies_to_all_api_adapters(self):
        self.args.reasoning = "bounded"
        responses = self.probe.request_body("/v1/responses", {"model": "fixture"})
        messages = self.probe.request_body("/v1/messages", {"model": "fixture"})
        self.assertEqual(responses["reasoning_budget_tokens"], 128)
        self.assertEqual(messages["thinking"], {"type": "enabled", "budget_tokens": 128})
        self.assertTrue(messages["chat_template_kwargs"]["enable_thinking"])

    def test_request_body_is_not_mutated_by_later_continuations(self):
        original = {"model": "fixture", "messages": [{"role": "user", "content": "first"}]}
        copy = self.probe.request_body("/v1/chat/completions", original)
        original["messages"].append({"role": "user", "content": "second"})
        self.assertEqual(len(copy["messages"]), 1)

    def test_truncated_responses_stream_fails(self):
        self.probe.request = mock.Mock(return_value=[{"type": "response.created"}])
        with self.assertRaisesRegex(AssertionError, "response.completed"):
            self.probe.responses(stream=True)

    def test_missing_messages_stop_fails(self):
        self.probe.request = mock.Mock(return_value=[{"type": "message_start", "message": {}}])
        with self.assertRaisesRegex(AssertionError, "message_stop"):
            self.probe.messages(stream=True)

    def test_messages_rejects_unusable_text_and_truncated_answers(self):
        for content, stop in (([{"type": "thinking", "thinking": "Still thinking"}], "end_turn"),
                              ([{"type": "text", "text": "  "}], "end_turn"),
                              ([{"type": "text", "text": "Hello"}], "max_tokens"),
                              ([{"type": "text", "text": "Hello"}], None)):
            self.probe.request = mock.Mock(return_value={
                "type": "message", "role": "assistant", "id": "m", "usage": {},
                "content": content, "stop_reason": stop})
            with self.subTest(content=content, stop=stop), self.assertRaises(AssertionError):
                self.probe.messages()

    def test_messages_accepts_completed_text(self):
        self.probe.request = mock.Mock(return_value={
            "type": "message", "role": "assistant", "id": "m", "usage": {},
            "content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn"})
        self.probe.messages()

    def test_messages_stream_rejects_thinking_only(self):
        self.probe.request = mock.Mock(return_value=[
            {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "usage": {}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"}])
        with self.assertRaisesRegex(AssertionError, "text"):
            self.probe.messages(stream=True)

    def test_responses_roundtrip_preserves_call_id_and_multiple_instructions(self):
        self.probe.request = mock.Mock(side_effect=[{
            "id": "resp1", "object": "response", "status": "completed", "usage": {},
            "output": [{"type": "function_call", "name": "lookup_code", "call_id": "call1",
                        "arguments": '{"key":"fixture"}'}]},
            {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "violet-731"}]}]}])
        self.probe.responses(tools=True)
        body = self.probe.request.call_args_list[1].args[1]
        self.assertIn("instructions", body)
        self.assertEqual(body["input"][0]["role"], "developer")
        self.assertTrue(any(item.get("type") == "function_call_output" and item["call_id"] == "call1"
                            for item in body["input"]))

    def test_messages_stream_parses_incremental_json_arguments(self):
        events = [
            {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "usage": {}}},
            {"type": "content_block_start", "index": 0, "content_block": {
                "type": "tool_use", "id": "t1", "name": "lookup_code", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {
                "type": "input_json_delta", "partial_json": '{"key":'}},
            {"type": "content_block_delta", "index": 0, "delta": {
                "type": "input_json_delta", "partial_json": '"fixture"}'}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
        self.probe.request = mock.Mock(side_effect=[events, {
            "stop_reason": "end_turn", "content": [{"type": "text", "text": "violet-731"}]}])
        self.probe.messages(stream=True, tools=True)
        body = self.probe.request.call_args_list[1].args[1]
        self.assertEqual(body["messages"][1]["content"][0]["input"], {"key": "fixture"})

    def test_wrong_second_tool_arguments_fail(self):
        def response(name, arguments):
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}]}}]}
        self.probe.chat = mock.Mock(side_effect=[response("lookup_code", '{"key":"fixture"}'),
                                                 response("verify_code", '{"code":"invented"}')])
        with self.assertRaisesRegex(AssertionError, "arguments wrong"):
            self.probe.multi_tool("chat")


if __name__ == "__main__":
    unittest.main()
