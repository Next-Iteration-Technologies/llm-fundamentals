from scripted_llm import ScriptedLLM, asks_for
from step8_authentication import IDENTITY_PROVIDER
from step12_pii_redaction import Redactor, RedactingModel, personal_data_visible_to


def test_the_same_person_gets_the_same_placeholder():
    redactor = Redactor()
    text = redactor.redact("Priya Sharma wrote to priya.sharma@corp.example. Later Priya Sharma called.")
    assert text == "[PERSON_1] wrote to [EMAIL_1]. Later [PERSON_1] called."


def test_names_in_file_paths_are_found():
    assert Redactor().redact("/HR/Sharma_Priya/contract.pdf") == "/HR/[PERSON_1]_[PERSON_2]/contract.pdf"


def test_restore_puts_back_only_what_the_reader_may_see():
    redactor = Redactor()
    text = redactor.redact("Priya Sharma and Jonas Becker can see Finance.")
    priya = IDENTITY_PROVIDER.sign_in("priya").user
    assert redactor.restore(text, personal_data_visible_to(priya)) == "Priya Sharma and [PERSON_2] can see Finance."
    mia = IDENTITY_PROVIDER.sign_in("mia").user
    assert redactor.restore(text, personal_data_visible_to(mia)) == "Priya Sharma and Jonas Becker can see Finance."


def test_the_model_sees_placeholders_and_the_tool_gets_real_values():
    redactor = Redactor()
    llm = ScriptedLLM([asks_for("raise_access_request", doc_id="1234", reason="for [PERSON_1]")])
    reply = RedactingModel(llm, redactor).chat([{"role": "user", "content": "Access for Jonas Becker please"}])
    assert llm.requests[0]["messages"][0]["content"] == "Access for [PERSON_1] please"
    assert reply.tool_calls[0]["arguments"]["reason"] == "for Jonas Becker"
