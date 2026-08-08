from scripts.run_json_acceptance import _offline_quality


def test_explicit_gold_facts_are_scored_only_after_answer_exists():
    case = {
        "final_answer": "Python 3.14.3 发布于 2026 年 2 月 3 日，约 299 项。",
        "gold": {
            "required_facts": ["Python 3.14.3", "2026-02-03", "约 299"],
            "forbidden_facts": ["2026-06-10"],
        },
    }
    quality, comparison = _offline_quality(
        case,
        "Python 3.14.3 was released on February 3, 2026 with around 299 changes.",
    )
    assert quality == "pass"
    assert comparison["method"] == "offline_explicit_fact_match.v1"


def test_missing_or_forbidden_gold_fact_is_no_pass():
    case = {
        "gold": {
            "required_facts": ["Python 3.14.6", "2026-06-10"],
            "forbidden_facts": ["Python 3.14.3"],
        }
    }
    quality, comparison = _offline_quality(case, "Python 3.14.3 was released.")
    assert quality == "no-pass"
    assert any(row["matched"] for row in comparison["forbidden"])


def test_case_without_explicit_gold_is_not_given_a_quality_label():
    assert _offline_quality({"reference_answer": "42"}, "42") == (None, None)
