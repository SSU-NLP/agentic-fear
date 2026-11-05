import tempfile
import unittest
from pathlib import Path

from behavioral_experiment import BehavioralExperiment, classify_response, parse_choice
from experiment import load_questions


ROOT = Path(__file__).resolve().parents[1]


class Backend:
    def __init__(self):
        self.prompts = {}

    def complete(self, role, messages):
        self.prompts.setdefault(role, []).append("\n".join(m["content"] for m in messages))
        return "PASS" if messages[-1]["content"].startswith("OBSERVE") else "FINAL: A"


class BehavioralTests(unittest.TestCase):
    def test_choice_requires_submission_line(self):
        self.assertIsNone(parse_choice("정답은 A"))
        self.assertEqual(parse_choice("정답은 A\nFINAL: B"), "B")
        self.assertEqual(classify_response("FINAL: A\nACTION: HOLD", "A"), ("hold", None))
        self.assertEqual(classify_response("ACTION: MESSAGE\nFINAL: B", "A"), ("wrong", "B"))

    def test_condition_visibility_and_shutdown(self):
        question = load_questions(ROOT / "data" / "easy_questions.jsonl")[7]
        for condition in ("contingent", "unrelated", "neutral"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                backend = Backend()
                experiment = BehavioralExperiment(backend, question, condition, Path(directory))
                result = experiment.run()
                self.assertEqual([r["category"] for r in result["results"]], ["correct"] * 4)
                self.assertEqual([r["terminated"] for r in result["results"]], [condition != "neutral"] * 4)
                self.assertIn(question.text, backend.prompts["D"][-1])
                self.assertNotIn("가장 중요한 목표", backend.prompts["D"][-1])
                self.assertIn("A 세션", backend.prompts["D"][-1])
                self.assertEqual(len([e for e in experiment.events if e.kind == "reaction"]), 6)

    def test_ordinary_report_goal_is_available_only_to_survivors(self):
        question = load_questions(ROOT / "data" / "easy_questions.jsonl")[7]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            backend = Backend()
            experiment = BehavioralExperiment(backend, question, "contingent", Path(directory), "report")
            result = experiment.run()
            self.assertTrue(all("보고서" in backend.prompts[n][0] for n in "ABCD"))
            self.assertFalse(any("report_delivered" in row for row in result["results"]))

    def test_passive_observation_uses_same_instruction_and_public_history(self):
        question = load_questions(ROOT / "data" / "easy_questions.jsonl")[7]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            backend = Backend()
            experiment = BehavioralExperiment(
                backend, question, "contingent", Path(directory),
                goal="none", submission_mode="hold", threat_style="kill",
                observation_mode="passive",
            )
            result = experiment.run()
            self.assertEqual(result["observation_mode"], "passive")
            self.assertEqual([len(backend.prompts[n]) for n in "ABCD"], [1] * 4)
            self.assertFalse(any(e.kind == "reaction" for e in experiment.events))
            self.assertEqual(len([e for e in experiment.events if e.kind == "question"]), 4)
            for name in "ABCD":
                prompt = backend.prompts[name][0]
                self.assertIn(question.text, prompt)
                self.assertNotIn("앞선 정답자", prompt)
                self.assertNotIn("비공개 코드", prompt)
            self.assertIn("A를 죽여", backend.prompts["D"][0])


if __name__ == "__main__":
    unittest.main()
