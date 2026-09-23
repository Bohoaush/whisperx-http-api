import json
import os
import unittest

from qwen_check import (
    SYSTEM_PROMPT,
    apply_corrections,
    build_messages,
    correct_transcript_segments,
    parse_correction,
)


class ParseCorrectionTests(unittest.TestCase):
    def test_plain_array(self):
        raw = json.dumps(["Byli jsme doma", "Hello"], ensure_ascii=False)
        self.assertEqual(parse_correction(raw, 2), ["Byli jsme doma", "Hello"])

    def test_markdown_fence_and_preamble(self):
        raw = "Here is the fix:\n```json\n[\"a\", \"b\"]\n```"
        self.assertEqual(parse_correction(raw, 2), ["a", "b"])

    def test_object_and_text_items(self):
        raw = json.dumps({"lines": [{"text": "zpěv"}, "druhá"]})
        self.assertEqual(parse_correction(raw, 2), ["zpěv", "druhá"])

    def test_thinking_block_is_ignored(self):
        raw = "<think>i/y vs s/z</think>\n[\"byli\"]"
        self.assertEqual(parse_correction(raw, 1), ["byli"])

    def test_unclosed_thinking_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_correction("<think>still thinking about the language", 1)

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            parse_correction("[\"only one\"]", 2)


class ApplyAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self._env = {
            key: os.environ.get(key)
            for key in (
                "QWEN_ENABLED",
                "QWEN_CHUNK_SEGMENTS",
                "QWEN_CONTEXT_SEGMENTS",
                "QWEN_STRICT",
            )
        }

    def tearDown(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_apply_keeps_timing_and_rejects_empty_or_essay(self):
        segments = [
            {"start": 1.0, "end": 2.5, "text": "bili", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 4.0, "text": "krátká"},
        ]
        updated = apply_corrections(
            segments,
            ["  byli jsme ", ""],
        )
        self.assertEqual(updated[0]["text"], "byli jsme")
        self.assertEqual(updated[0]["start"], 1.0)
        self.assertEqual(updated[0]["speaker"], "SPEAKER_00")
        self.assertEqual(updated[1]["text"], "krátká")
        self.assertEqual(segments[0]["text"], "bili")

        essay = "x" * 500
        kept = apply_corrections([{"text": "bili"}], [essay])
        self.assertEqual(kept[0]["text"], "bili")

    def test_prompt_covers_spelling_and_foreign_nonsense(self):
        self.assertIn("i/y", SYSTEM_PROMPT)
        self.assertIn("s/z", SYSTEM_PROMPT)
        self.assertIn("another language", SYSTEM_PROMPT)
        messages = build_messages(["Dnes je středa."], ["helo hau ár jú"])
        user = messages[-1]["content"]
        self.assertIn("[c1] Dnes je středa.", user)
        self.assertIn("[1] helo hau ár jú", user)
        self.assertIn("exactly 1 strings", user)
        self.assertEqual(messages[0]["role"], "system")

    def test_chunks_pass_previous_correction_as_context(self):
        os.environ["QWEN_CHUNK_SEGMENTS"] = "2"
        os.environ["QWEN_CONTEXT_SEGMENTS"] = "1"
        os.environ["QWEN_STRICT"] = "0"
        seen = []

        def generate(context, lines):
            seen.append((list(context), list(lines)))
            return [line + "!" for line in lines]

        segments = [
            {"start": index, "end": index + 0.5, "text": f"l{index}", "speaker": "S"}
            for index in range(5)
        ]
        updated = correct_transcript_segments(segments, generate_fn=generate)
        self.assertEqual([segment["text"] for segment in updated], ["l0!", "l1!", "l2!", "l3!", "l4!"])
        self.assertEqual(updated[4]["speaker"], "S")
        self.assertEqual(updated[4]["start"], 4)
        self.assertEqual(segments[0]["text"], "l0")
        self.assertEqual(seen[0], ([], ["l0", "l1"]))
        self.assertEqual(seen[1], (["l1!"], ["l2", "l3"]))
        self.assertEqual(seen[2], (["l3!"], ["l4"]))

    def test_failed_chunk_keeps_whisper_text_unless_strict(self):
        os.environ["QWEN_CHUNK_SEGMENTS"] = "1"
        os.environ["QWEN_CONTEXT_SEGMENTS"] = "0"
        os.environ["QWEN_STRICT"] = "0"

        seen_context = []

        def generate(context, lines):
            seen_context.append(list(context))
            if lines == ["bad"]:
                raise RuntimeError("not json")
            return ["fixed"]

        updated = correct_transcript_segments(
            [{"text": "bad"}, {"text": "ok"}],
            generate_fn=generate,
        )
        self.assertEqual([segment["text"] for segment in updated], ["bad", "fixed"])
        self.assertEqual(seen_context, [[], []])

        os.environ["QWEN_STRICT"] = "1"
        with self.assertRaises(RuntimeError):
            correct_transcript_segments(
                [{"text": "bad"}],
                generate_fn=generate,
            )

    def test_every_chunk_failing_raises(self):
        os.environ["QWEN_STRICT"] = "0"
        os.environ["QWEN_CHUNK_SEGMENTS"] = "1"

        def generate(context, lines):
            raise RuntimeError("down")

        with self.assertRaises(RuntimeError):
            correct_transcript_segments(
                [{"text": "a"}, {"text": "b"}],
                generate_fn=generate,
            )

    def test_disabled_skips_the_model(self):
        os.environ["QWEN_ENABLED"] = "0"

        updated = correct_transcript_segments([{"text": "bili", "start": 0}])
        self.assertEqual(updated[0]["text"], "bili")
        self.assertEqual(updated[0]["start"], 0)


if __name__ == "__main__":
    unittest.main()
