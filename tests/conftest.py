from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    """Write a clean, human-readable pytest result file after every test run."""
    root = Path(__file__).resolve().parents[1]
    output = root / "test_output.txt"

    lines = [
        "HOUSE PRICE PREDICTION - TEST OUTPUT",
        "=" * 62,
        f"Exit status: {exitstatus}",
        f"Tests collected: {session.testscollected}",
        "",
        "Result:",
    ]

    if exitstatus == 0:
        lines.append("  ALL TESTS PASSED")
    elif exitstatus == 1:
        lines.append("  SOME TESTS FAILED")
    elif exitstatus == 2:
        lines.append("  TEST RUN INTERRUPTED")
    elif exitstatus == 5:
        lines.append("  NO TESTS WERE COLLECTED")
    else:
        lines.append("  PYTEST ENDED WITH AN ERROR")

    lines.extend(
        [
            "",
            "The file is generated automatically by pytest via tests/conftest.py.",
            "Run command: pytest -q",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
