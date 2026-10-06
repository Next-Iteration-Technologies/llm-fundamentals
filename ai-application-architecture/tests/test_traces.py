import json

from step8_authentication import IDENTITY_PROVIDER
from step12_pii_redaction import Redactor
from step13_logging_traces import Tracer, check_case


def test_a_trace_line_masks_personal_data(tmp_path):
    tracer = Tracer(Redactor(), folder=tmp_path)
    tracer.start(IDENTITY_PROVIDER.sign_in("priya"), "Is jonas.becker@corp.example in Finance?")
    line = json.loads((tmp_path / tracer.file_name).read_text().splitlines()[0])
    assert line["kind"] == "question" and "[EMAIL_1]" in line["question"]


def test_a_golden_question_lists_what_went_wrong():
    case = {"expect_tools": ["search_document"], "expect_words": ["Finance"], "avoid_words": ["Meyer"]}
    assert check_case(case, "It is in Finance.", ["search_document"]) == []
    assert check_case(case, "Meyer Logistics", []) == [
        "did not use search_document", "answer lacks 'Finance'", "answer contains 'Meyer'"]
