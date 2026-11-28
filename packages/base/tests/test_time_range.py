from datetime import datetime

from concord.time_range import TimeList


def test_time_list_isolated_times():
    times = [
        "2024-01-01T00:00:00",
        "2024-01-01T00:05:00",
        "2024-01-01T00:12:00",
        "2024-01-01T00:25:00",
        "2024-01-01T00:54:00",
    ]
    time_list = TimeList([datetime.fromisoformat(t) for t in times])
    assert len(time_list.sequences) == 5
    assert len(time_list) == 5
    assert list(time_list) == [datetime.fromisoformat(t) for t in times]


def test_time_list_consecutive_times():
    times = [
        "2024-01-01T00:00:00",
        "2024-01-01T00:05:00",
        "2024-01-01T00:10:00",
        "2024-01-01T00:15:00",
        "2024-01-01T00:20:00",
    ]
    time_list = TimeList([datetime.fromisoformat(t) for t in times])
    assert len(time_list.sequences) == 1
    assert len(time_list) == 5
    assert list(time_list) == [datetime.fromisoformat(t) for t in times]


def test_time_list_mixed_times():
    times = [
        "2024-01-01T00:00:00",
        "2024-01-01T00:05:00",
        "2024-01-01T00:10:00",
        "2024-01-01T00:25:00",
        "2024-01-01T00:30:00",
        "2024-01-01T00:35:00",
    ]
    time_list = TimeList([datetime.fromisoformat(t) for t in times])
    assert len(time_list.sequences) == 2
    assert len(time_list) == 6
    assert list(time_list) == [datetime.fromisoformat(t) for t in times]
