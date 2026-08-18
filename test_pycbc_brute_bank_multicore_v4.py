"""Regression tests for the v4 multiprocessing lifecycle."""

import signal
import time
import unittest

import pycbc_brute_bank_multicore_v4 as v4


def worker_signal_state(_):
    return (
        signal.getsignal(signal.SIGTERM),
        signal.getsignal(signal.SIGINT),
    )


class WorkerPoolLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.old_sigterm = signal.getsignal(signal.SIGTERM)
        self.old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, v4._request_stop)
        signal.signal(signal.SIGINT, v4._request_stop)

    def tearDown(self):
        signal.signal(signal.SIGTERM, self.old_sigterm)
        signal.signal(signal.SIGINT, self.old_sigint)

    def test_workers_reset_parent_checkpoint_handlers(self):
        with v4.worker_pool(
                4, initializer=v4._waveform_worker_init) as pool:
            states = pool.map(worker_signal_state, range(8))
            workers = tuple(pool._pool)

        self.assertEqual(
            set(states),
            {(signal.SIG_DFL, signal.SIG_IGN)},
        )
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertIs(signal.getsignal(signal.SIGTERM), v4._request_stop)
        self.assertIs(signal.getsignal(signal.SIGINT), v4._request_stop)

    def test_exception_terminates_workers_without_signal_storm(self):
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with v4.worker_pool(
                    4, initializer=v4._waveform_worker_init) as pool:
                workers = tuple(pool._pool)
                for _ in workers:
                    pool.apply_async(time.sleep, (60,))
                raise RuntimeError("test failure")

        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(all(not worker.is_alive() for worker in workers))


if __name__ == "__main__":
    unittest.main()
