"""
agents/base_agent.py
this is a motherclass (parent class of all ai_agent files) 

---------------------
Common parent class for every agent in the pipeline.
Every agent MUST inherit from BaseAgent and implement `run()`.

This gives us for free:
 - consistent logging
 - consistent error handling
 - a timing wrapper (execute()) that all agents share
 - a standard success/failure result format
"""

import time
import traceback # error ka ala ? trackbak history sangta 
from abc import ABC, abstractmethod # this is a decorator functions 

from utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class that all pipeline agents inherit from."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Core logic of the agent. Must be implemented by every subclass.
        Should return whatever output the next agent in the pipeline needs
        (e.g. a DataFrame, a dict, a model object, etc.)
        """
        raise NotImplementedError("Every agent must implement the run() method.")

    def execute(self, *args, **kwargs):
        """
        Wrapper around run() that adds timing, logging, and standard
        error handling. Always call agent.execute(...) from main.py,
        not agent.run(...) directly.

        Returns:
            dict: {
                "success": bool,
                "result": <output of run() or None>,
                "error": <error message or None>,
                "duration_sec": float
            }
        """
        self.logger.info(f"----- Starting {self.name} -----")
        start_time = time.time()

        try:
            result = self.run(*args, **kwargs)
            duration = round(time.time() - start_time, 3)
            self.logger.info(f"----- Finished {self.name} in {duration}s -----")
            return {
                "success": True,
                "result": result,
                "error": None,
                "duration_sec": duration
            }

        except Exception as e:
            duration = round(time.time() - start_time, 3)
            self.logger.error(f"{self.name} FAILED after {duration}s: {e}")
            self.logger.debug(traceback.format_exc())
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "duration_sec": duration
            }
