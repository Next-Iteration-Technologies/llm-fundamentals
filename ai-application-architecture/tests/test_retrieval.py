from step3_retrieval import search_known_issues


def test_file_too_large_finds_ki_031_first():
    matches = search_known_issues("My upload failed, it says the file is too large.")["matches"]
    assert matches[0]["id"] == "KI-031"


def test_an_unknown_problem_finds_nothing_and_says_hand_over():
    result = search_known_issues("The archive shows a blue screen when I log in.")
    assert result["matches"] == []
    assert "hand over" in result["note"]
