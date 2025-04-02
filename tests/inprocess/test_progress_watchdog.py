import copy
import time
import unittest
from datetime import timedelta

import nvidia_resiliency_ext.inprocess as inprocess

from . import common  # noqa: F401


class Collect:
    def __init__(self):
        self.tstamps = []

    def send_timestamp(self, timestamp):
        self.tstamps.append(copy.copy(timestamp))


class TestProgressWatchdog(unittest.TestCase):
    def test_all(self):
        collect = Collect()
        watchdog = inprocess.progress_watchdog.ProgressWatchdog(
            0, collect, timedelta(milliseconds=5)
        )
        watchdog.start()
        watchdog.pause_and_synchronize()
        watchdog.resume()
        watchdog.shutdown()

    def test_auto(self):
        start = time.monotonic()
        collect = Collect()
        watchdog = inprocess.progress_watchdog.ProgressWatchdog(
            0, collect, timedelta(milliseconds=5)
        )
        watchdog.start()

        for _ in range(50):
            time.sleep(0.005)
        watchdog.shutdown()

        self.assertGreater(len(collect.tstamps), 10)
        for tstamp in collect.tstamps:
            self.assertIsInstance(tstamp, inprocess.progress_watchdog.Timestamp)
            self.assertIsInstance(tstamp.auto, float)
            self.assertGreater(tstamp.auto, start)
            self.assertIsNone(tstamp.manual)

        for t1, t2 in zip(collect.tstamps, collect.tstamps[1:]):
            self.assertLessEqual(t1.auto, t2.auto)

    def test_pause_resume(self):
        collect = Collect()
        watchdog = inprocess.progress_watchdog.ProgressWatchdog(
            0, collect, timedelta(milliseconds=1)
        )
        watchdog.start()

        for _ in range(100):
            time.sleep(1e-4)

        watchdog.pause_and_synchronize()
        pause_len = len(collect.tstamps)
        for _ in range(100):
            time.sleep(1e-4)
        self.assertEqual(pause_len, len(collect.tstamps))
        watchdog.resume()
        for _ in range(100):
            time.sleep(1e-4)
        resume_len = len(collect.tstamps)
        self.assertGreater(resume_len, pause_len)

        max_diff = 0
        for t1, t2 in zip(collect.tstamps[pause_len:], collect.tstamps[pause_len + 1 :]):
            max_diff = max(max_diff, t2.auto - t1.auto)

        self.assertGreater(max_diff, 1e-2)

        watchdog.shutdown()

    def test_double_resume(self):
        collect = Collect()
        watchdog = inprocess.progress_watchdog.ProgressWatchdog(
            0, collect, timedelta(milliseconds=1)
        )
        watchdog.start()

        for _ in range(100):
            time.sleep(1e-4)

        len_1 = len(collect.tstamps)
        watchdog.resume()

        for _ in range(100):
            time.sleep(1e-4)
        len_2 = len(collect.tstamps)

        watchdog.resume()
        for _ in range(100):
            time.sleep(1e-4)

        len_3 = len(collect.tstamps)

        self.assertGreater(len_3, len_2)
        self.assertGreater(len_2, len_1)
        watchdog.shutdown()

    def test_double_shutdown(self):
        collect = Collect()
        watchdog = inprocess.progress_watchdog.ProgressWatchdog(
            0, collect, timedelta(milliseconds=1)
        )
        watchdog.start()

        for _ in range(100):
            time.sleep(1e-4)

        len_1 = len(collect.tstamps)
        watchdog.shutdown()
        len_2 = len(collect.tstamps)

        for _ in range(100):
            time.sleep(1e-4)
        len_3 = len(collect.tstamps)
        watchdog.shutdown()
        self.assertEqual(len_1, len_2 - 1)
        self.assertEqual(len_2, len_3)

    def test_shutdown_resume(self):
        collect = Collect()
        watchdog = inprocess.progress_watchdog.ProgressWatchdog(
            0, collect, timedelta(milliseconds=1)
        )
        watchdog.start()
        watchdog.shutdown()
        with self.assertRaises(RuntimeError):
            watchdog.resume()
