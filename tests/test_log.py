import logging
import unittest

from algo_tags.log import get_logger, setup_logging, timed


class TimedTest(unittest.TestCase):
    def setUp(self):
        setup_logging(log_file=None)
        self.log = get_logger("test")

    def tearDown(self):
        logging.getLogger("algo_tags").handlers.clear()

    def test_success_logs_the_duration(self):
        with self.assertLogs("algo_tags", level="INFO") as logs:
            with timed(self.log, "step"):
                pass
        self.assertTrue(any("step done in" in line for line in logs.output))

    def test_failure_logs_the_duration_and_passes_the_error_on(self):
        with self.assertLogs("algo_tags", level="ERROR") as logs:
            with self.assertRaises(RuntimeError):
                with timed(self.log, "step"):
                    raise RuntimeError("boom")
        self.assertTrue(any("step failed after" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
