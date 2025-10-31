from __future__ import annotations

from scripts import scheduler


def _run_scheduler_once(monkeypatch):
    instances: list[FakeScheduler] = []

    class FakeScheduler:
        def __init__(self):
            self.interval_minutes = None
            self.shutdown_called = False
            instances.append(self)

        def add_job(self, func, trigger, minutes, id):  # noqa: A002 - keep signature similar
            self.interval_minutes = minutes

        def start(self):
            pass

        def shutdown(self):
            self.shutdown_called = True

    def fake_sleep(_seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(scheduler, "BackgroundScheduler", FakeScheduler)
    monkeypatch.setattr(scheduler.time, "sleep", fake_sleep)

    scheduler.main()

    assert instances, "BackgroundScheduler was not instantiated"
    return instances[0]


def test_scheduler_uses_env_interval(monkeypatch):
    monkeypatch.setenv("SCHEDULER_INTERVAL_MINUTES", "7")

    fake_scheduler = _run_scheduler_once(monkeypatch)

    assert fake_scheduler.interval_minutes == 7
    assert fake_scheduler.shutdown_called


def test_scheduler_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SCHEDULER_INTERVAL_MINUTES", "0")

    fake_scheduler = _run_scheduler_once(monkeypatch)

    assert fake_scheduler.interval_minutes == scheduler.DEFAULT_INTERVAL_MINUTES
