from scripted_llm import ScriptedLLM, answers
from step4_conversation_store import ConversationStore, answer_with_memory


def test_a_session_survives_a_new_store_on_the_same_file(tmp_path):
    database = tmp_path / "conversations.db"
    ConversationStore(database).save("priya", [{"role": "user", "content": "Where is invoice 1234?"}])
    assert ConversationStore(database).load("priya") == [{"role": "user", "content": "Where is invoice 1234?"}]


def test_save_only_adds_new_messages():
    store = ConversationStore(":memory:")
    first = [{"role": "user", "content": "one"}]
    store.save("s", first)
    store.save("s", first + [{"role": "assistant", "content": "two"}])
    assert [message["content"] for message in store.load("s")] == ["one", "two"]


def test_forget_removes_only_that_session():
    store = ConversationStore(":memory:")
    store.save("a", [{"role": "user", "content": "a"}])
    store.save("b", [{"role": "user", "content": "b"}])
    store.forget("a")
    assert store.load("a") == [] and len(store.load("b")) == 1


def test_the_second_question_is_sent_with_the_first():
    store = ConversationStore(":memory:")
    llm = ScriptedLLM([answers("It is in Finance."), answers("Until 2035.")])
    answer_with_memory(llm, store, "priya", "Where is invoice 1234?", tools=[], system="system")
    answer_with_memory(llm, store, "priya", "How long is it kept?", tools=[], system="system")
    sent = [message["content"] for message in llm.requests[1]["messages"]]
    assert sent == ["Where is invoice 1234?", "It is in Finance.", "How long is it kept?"]
