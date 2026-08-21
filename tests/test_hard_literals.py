from utils.hard_literals import (
    extract_hard_literals,
    hard_literal_keys,
    untrusted_hard_literals,
)


def test_specific_literals_suppress_nested_numeric_fragments():
    rows = extract_hard_literals(
        "CVE-2026-12345 shipped on 2026-08-14 as v3.14.1; record 987654."
    )

    assert [(row.kind, row.canonical_value) for row in rows] == [
        ("cve", "CVE-2026-12345"),
        ("date", "2026-08-14"),
        ("version", "3.14.1"),
        ("numeric_id", "987654"),
    ]


def test_trusted_full_date_also_authorizes_its_year_component():
    allowed = hard_literal_keys(["current_utc_date=2026-08-14"])

    assert ("date", "2026-08-14") in allowed
    assert ("year", "2026") in allowed
    assert not untrusted_hard_literals("latest release 2026", allowed_keys=allowed)


def test_changed_version_remains_untrusted():
    allowed = hard_literal_keys(["Use version 3.14.1"])

    rejected = untrusted_hard_literals("Use version 3.15.0", allowed_keys=allowed)

    assert [(row.kind, row.canonical_value) for row in rejected] == [
        ("version", "3.15.0")
    ]
