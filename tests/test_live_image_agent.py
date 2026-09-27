"""Offline contracts for the live DuckDuckGo image-search agent.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.
No network: only the pure helpers (image magic-byte sniffing, URL shortening,
stale-page trimming, navigation allow-list) are exercised here.
"""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "live_image_agent", ROOT / "tests" / "live_image_agent.py")
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


class ImageKindTest(unittest.TestCase):
    def test_recognises_jpeg_png_gif_webp_by_magic_bytes(self):
        self.assertEqual(live.image_kind(b"\xff\xd8\xff\xe0rest-of-jpeg"), "jpg")
        self.assertEqual(live.image_kind(b"\x89PNG\r\n\x1a\nrest"), "png")
        self.assertEqual(live.image_kind(b"GIF89a..."), "gif")
        webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"rest"
        self.assertEqual(live.image_kind(webp), "webp")

    def test_rejects_non_image_bytes(self):
        self.assertIsNone(live.image_kind(b"<html><body>not an image</body></html>"))
        self.assertIsNone(live.image_kind(b""))

    def test_riff_without_webp_fourcc_is_not_an_image(self):
        avi = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"rest"
        self.assertIsNone(live.image_kind(avi))


class ShortenUrlsTest(unittest.TestCase):
    def test_leaves_short_urls_untouched(self):
        text = "See https://duckduckgo.com/?q=oranges for results."
        self.assertEqual(live.shorten_urls(text), text)

    def test_shortens_urls_over_300_chars(self):
        long_url = "https://duckduckgo.com/i.jpg?" + "a" * 300
        text = f"Image at {long_url} looks good."
        result = live.shorten_urls(text)
        self.assertIn("[long link shortened]", result)
        self.assertLess(len(result), len(text))
        self.assertTrue(result.startswith("Image at " + long_url[:120]))

    def test_boundary_length_just_under_threshold_is_untouched(self):
        url = "https://duckduckgo.com/" + "a" * (live.LONG_URL - len("https://duckduckgo.com/") - 1)
        text = f"See {url} now"
        self.assertEqual(live.shorten_urls(text), text)


class DropStalePagesTest(unittest.TestCase):
    def test_keeps_only_the_newest_large_tool_result(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "x" * 2000},
            {"role": "assistant", "content": "..."},
            {"role": "tool", "content": "y" * 2000},
        ]
        live.drop_stale_pages(messages)
        self.assertIn("[older page view removed to save context]", messages[1]["content"])
        self.assertLess(len(messages[1]["content"]), 500)
        # The newest tool message (last in the list) is never touched by
        # drop_stale_pages, even though it is also large.
        self.assertEqual(messages[3]["content"], "y" * 2000)

    def test_short_tool_messages_are_left_alone(self):
        messages = [{"role": "tool", "content": "short"}, {"role": "tool", "content": "z" * 2000}]
        live.drop_stale_pages(messages)
        self.assertEqual(messages[0]["content"], "short")

    def test_non_tool_messages_are_never_truncated(self):
        messages = [{"role": "assistant", "content": "a" * 2000}, {"role": "tool", "content": "b" * 2000}]
        live.drop_stale_pages(messages)
        self.assertEqual(messages[0]["content"], "a" * 2000)


class AllowedNavigationTest(unittest.TestCase):
    def test_allows_duckduckgo_and_its_subdomains(self):
        self.assertTrue(live.allowed_navigation("https://duckduckgo.com/?q=oranges"))
        self.assertTrue(live.allowed_navigation("https://duckduckgo.com/"))
        self.assertTrue(live.allowed_navigation("https://external-content.duckduckgo.com/iu/?u=1"))

    def test_blocks_other_hosts_including_lookalikes(self):
        self.assertFalse(live.allowed_navigation("https://example.com/"))
        self.assertFalse(live.allowed_navigation("https://notduckduckgo.com/"))
        self.assertFalse(live.allowed_navigation("https://duckduckgo.com.evil.example/"))
        self.assertFalse(live.allowed_navigation(""))


class ContractTest(unittest.TestCase):
    def test_tools_exclude_file_upload_and_include_save_image(self):
        self.assertNotIn("browser_file_upload", live.TOOLS)
        self.assertEqual(live.SAVE_TOOL["function"]["name"], "save_image")

    def test_task_and_system_prompt_scope_navigation_to_duckduckgo(self):
        self.assertIn("duckduckgo.com", live.TASK)
        self.assertIn("Only navigate to duckduckgo.com pages", live.SYSTEM)

    def test_image_byte_cap_is_15_mib(self):
        self.assertEqual(live.MAX_IMAGE_BYTES, 15 * 1024 * 1024)

    def test_cli_accepts_image_dir(self):
        import inspect
        source = inspect.getsource(live.main)
        self.assertIn("--image-dir", source)


if __name__ == "__main__":
    unittest.main()
