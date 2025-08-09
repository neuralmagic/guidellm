"""
Singleton pattern implementations for ensuring single instance classes.

Provides singleton mixins for creating classes that maintain a single instance
throughout the application lifecycle, with support for both basic and thread-safe
implementations.

Classes:
    SingletonMixin: Basic singleton implementation using class variables.
    ThreadSafeSingletonMixin: Thread-safe singleton using locking mechanisms.
"""

import threading
from typing import ClassVar

__all__ = ["SingletonMixin", "ThreadSafeSingletonMixin"]


class SingletonMixin:
    """
    Basic singleton mixin ensuring single instance per class.

    Implements the singleton pattern using class variables to control instance
    creation. Subclasses must call super().__init__() for proper initialization
    state management.
    """

    singleton_instance: ClassVar["SingletonMixin"] = None

    def __new__(cls, *args, **kwargs):
        """
        Create or return the singleton instance.

        :param args: Positional arguments passed to the constructor.
        :param kwargs: Keyword arguments passed to the constructor.
        :return: The singleton instance of the class.
        """
        if cls.singleton_instance is None:
            cls.singleton_instance = super().__new__(cls, *args, **kwargs)
            cls.singleton_instance.initialized = False
        return cls.singleton_instance

    def __init__(self):
        """Initialize the singleton instance exactly once."""
        if self.initialized:
            return
        self.initialized = True


class ThreadSafeSingletonMixin(SingletonMixin):
    """
    Thread-safe singleton mixin with locking mechanisms.

    Extends SingletonMixin with thread safety using locks to prevent race
    conditions during instance creation in multi-threaded environments.
    """

    singleton_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Create or return the singleton instance with thread safety.

        :param args: Positional arguments passed to the constructor.
        :param kwargs: Keyword arguments passed to the constructor.
        :return: The singleton instance of the class.
        """
        with cls.singleton_lock:
            if cls.singleton_instance is None:
                cls.singleton_instance = super().__new__(cls, *args, **kwargs)
                cls.singleton_instance.initialized = False
            return cls.singleton_instance

    def __init__(self):
        """Initialize the singleton instance with thread-local lock."""
        if not self.initialized:
            self.thread_lock = threading.Lock()
        super().__init__()
