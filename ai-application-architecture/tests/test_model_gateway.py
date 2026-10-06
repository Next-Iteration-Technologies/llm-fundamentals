import pytest

import llm_client
from scripted_llm import ScriptedLLM, answers
from step11_model_gateway import ModelGateway, contains_personal_data


class BrokenModel(ScriptedLLM):
    def chat(self, messages, system=None, tools=None):
        raise llm_client.SetupProblem("no network")


def test_personal_data_is_spotted():
    assert contains_personal_data("Ask mia.wagner@corp.example")
    assert contains_personal_data("file /HR/Sharma_Priya/contract.pdf")
    assert not contains_personal_data("Invoice 1234 is kept until 2035-12-31.")


def test_personal_data_goes_to_the_private_model():
    default, private = ScriptedLLM([]), ScriptedLLM([answers("ok")])
    ModelGateway(default, private).chat([{"role": "user", "content": "I'm Priya Sharma"}])
    assert len(private.requests) == 1 and not default.requests


def test_the_gateway_falls_back_to_the_private_model():
    private = ScriptedLLM([answers("ok")])
    reply = ModelGateway(BrokenModel([]), private).chat([{"role": "user", "content": "Where is invoice 1234?"}])
    assert reply.text == "ok"


def test_personal_data_never_falls_back_to_the_default_model():
    default = ScriptedLLM([answers("should not be called")])
    with pytest.raises(llm_client.SetupProblem):
        ModelGateway(default, BrokenModel([])).chat([{"role": "user", "content": "Ask Priya"}])
    assert not default.requests
