import threading

from app.api import deps


def test_inline_mode_ingests_on_a_background_thread_not_the_request_thread(monkeypatch):
    done = threading.Event()
    seen = {}

    def fake_run(version_id, user_id):
        seen["args"] = (version_id, user_id)
        seen["thread"] = threading.current_thread().name
        done.set()

    monkeypatch.setattr("app.workers.tasks.run_ingestion", fake_run)
    monkeypatch.setattr(deps, "_inline_pool", None)
    deps.inline_enqueue("v1", "u1")
    assert done.wait(5)
    assert seen["args"] == ("v1", "u1")
    assert seen["thread"].startswith("ingest") and seen["thread"] != threading.current_thread().name


def test_inline_ingestions_run_one_at_a_time(monkeypatch):
    running, peak, finished = 0, 0, threading.Event()
    lock = threading.Lock()
    count = {"n": 0}

    def slow(version_id, user_id):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        threading.Event().wait(0.05)
        with lock:
            running -= 1
            count["n"] += 1
            if count["n"] == 4:
                finished.set()

    monkeypatch.setattr("app.workers.tasks.run_ingestion", slow)
    monkeypatch.setattr(deps, "_inline_pool", None)
    for i in range(4):
        deps.inline_enqueue(f"v{i}", "u")
    assert finished.wait(10) and peak == 1  # a long ingestion cannot starve the API of CPU


def test_enqueue_dependency_follows_the_setting(monkeypatch):
    from app.config import Settings

    monkeypatch.setattr(deps, "get_settings", lambda: Settings(ingest_mode="inline"))
    assert deps.enqueue_dep() is deps.inline_enqueue
    monkeypatch.setattr(deps, "get_settings", lambda: Settings(ingest_mode="celery"))
    assert deps.enqueue_dep() is not deps.inline_enqueue
