import functools
import inspect
import io
import logging
import logging.handlers
import multiprocessing
import os
import pathlib
import re
import shutil
import sys
import threading
import weakref
from contextlib import (contextmanager, nullcontext, redirect_stderr,
                        redirect_stdout)
from datetime import datetime
from logging import LogRecord
from typing import Any, Callable, Generator, Optional, Type, Union

from rich.console import Console, ConsoleRenderable
from rich.live import Live
from rich.logging import RichHandler
from rich.text import Text
from tqdm import tqdm

LOG_LEVEL = logging.INFO
CONSOLE = Console()

LOG_STATE = {"handler": None, "loggers": weakref.WeakSet()}


# Handlers

class RichRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc_formatter = logging.Formatter()  # For tracebacks

    def emit(self, record):
        try:
            if self.shouldRollover(record):
                self.doRollover()

            msg = getattr(record, 'message', super().format(record))
            plain_text = Text.from_markup(msg).plain

            if record.exc_info:
                exc_text = self.exc_formatter.formatException(record.exc_info)
                plain_text += '\n' + '\n'.join(f"    {line}" for line in exc_text.split('\n') if line)

            header = f"[{datetime.fromtimestamp(record.created):%Y-%m-%d %H:%M:%S.%f}] {record.levelname:<8}"

            if getattr(record, 'summary', False):
                indented_msg = '\n'.join(f"        {line}" for line in plain_text.split('\n'))
                output = f"{header}\n{indented_msg}\n"
            else:
                output = f"{header} {plain_text}\n"

            self.stream.write(output)
            self.stream.flush()
        except Exception:
            self.handleError(record)


class CustomRichHandler(RichHandler):
    def render_message(self, record: LogRecord, message: str) -> ConsoleRenderable:
        if getattr(record, 'summary', False):
            return Text.from_markup(message) if self.markup else Text(message)
        message_renderable = super().render_message(record, message)
        return Text.assemble(Text(f"{record.name} ", style="bold cyan"), message_renderable)


# Filters

class FileDisplayFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.main_pid = os.getpid()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.process != self.main_pid:
            return getattr(record, 'summary', False)
        return True


class ConsoleDisplayFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.main_pid = os.getpid()

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, 'no_console', False):
            return False
        return True


class NoFileOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, 'file_only', False)


class DuplicateFilter:
    def __init__(self) -> None:
        self.msgs = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg in self.msgs:
            return False
        self.msgs.add(record.msg)
        return True


class SuppressFilter(logging.Filter):
    def filter(self, record):
        record.no_console = True
        return True


class RealtimeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, 'summary', False):
            record.realtime = True
        return True


# Streams

class UnclosableStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, *args, **kwargs):
        return self._stream.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        return self._stream.flush(*args, **kwargs)

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class LoggerStream:
    def __init__(self, logger):
        self.logger = logger

    def write(self, msg):
        if msg.strip():
            self.logger.info(msg.strip())

    def flush(self):
        pass


class TqdmRichLiveIO(io.StringIO):
    def __init__(self, live_instance: Live, handler: RichHandler, logger_name: str):
        super().__init__()
        self.live = live_instance
        self.handler = handler
        self.logger_name = logger_name
        self.last_text = ""
        self._find_caller()

    def _find_caller(self):
        self.caller_filename = ""
        self.caller_lineno = 0
        for frame_info in inspect.stack():
            if 'tqdm' not in frame_info.filename and __file__ not in frame_info.filename:
                self.caller_filename = frame_info.filename
                self.caller_lineno = frame_info.lineno
                break

    def write(self, s: str):
        text = s.strip()
        if text and text != self.last_text:
            self.last_text = text
            record = logging.LogRecord(
                name=self.logger_name,
                level=logging.INFO,
                pathname=self.caller_filename,
                lineno=self.caller_lineno,
                msg=text,
                args=(),
                exc_info=None,
            )
            message_renderable = self.handler.render_message(record, text)
            full_renderable = self.handler.render(
                record=record,
                traceback=None,
                message_renderable=message_renderable
            )
            self.live.update(full_renderable)

    def flush(self):
        pass


# ---


def is_in_ipython():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger = logging.getLogger("System.Crash")
    logger.critical(
        f"Uncaught exception: {exc_type.__name__}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    if LOG_STATE["handler"]:
        LOG_STATE["handler"].flush()

    # sys.__excepthook__(exc_type, exc_value, exc_traceback)


def handle_thread_exception(args):
    handle_uncaught_exception(args.exc_type, args.exc_value, args.exc_traceback)


def worker_init(log_queue: multiprocessing.Queue, level: int = logging.INFO):
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.propagate = True
    root.setLevel(level)
    sys.excepthook = handle_uncaught_exception
    threading.excepthook = handle_thread_exception
    handler = logging.handlers.QueueHandler(log_queue)
    root.addHandler(handler)
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.propagate = True
        logger.setLevel(level)


def setup_logging(
    output_file: Union[str, None] = None,
    logger: logging.Logger = logging.getLogger(""),
    level: Union[int, None] = None,
    console_markup: bool = False,
    queue: Union[multiprocessing.Queue, None] = None,
    max_file_size_mb: int = 50,
) -> logging.Logger:
    """
    Configures logger with Rich console output and rotating file logging. Adds global exception handlers.

    Parameters
    ----------
    output_file : str, optional
        Path to the log file. If None, generates a timestamped file in the 'logs/' directory.
    logger : logging.Logger, optional
        The specific logger instance to configure. Defaults to the root logger.
    level : int, optional
        The logging level (e.g., logging.INFO). Defaults to the global LOG_LEVEL.
    console_markup : bool, default False
        If True, Rich style markup (e.g., '[bold red]Text[/]') in log messages (console) will be applied.
    queue : multiprocessing.Queue, optional
        Queue for multiprocessing. If provided, configures a QueueHandler. (handled internally i.e. in parallel_session context)
    max_file_size_mb : int, default 50
        Maximum size in megabytes before the log file is rotated. Rotated files are named with incremental suffixes.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    level = level or LOG_LEVEL

    if queue is not None:
        root = logging.getLogger()
        if root.hasHandlers():
            root.handlers.clear()
        root.addHandler(logging.handlers.QueueHandler(queue))
        root.setLevel(level)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)
        return logger

    if multiprocessing.current_process().name != 'MainProcess':
        logger.propagate = True
        return logger

    root = logging.getLogger()
    root.setLevel(level)

    if any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
        return logger

    LOG_STATE["loggers"].add(logger)

    if logger.hasHandlers():
        logger.handlers.clear()

    if not any(isinstance(h, CustomRichHandler) for h in root.handlers):
        console_handler = CustomRichHandler(
            console=CONSOLE,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=True,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
            markup=console_markup,
        )
        console_handler.addFilter(NoFileOnlyFilter())
        console_handler.addFilter(ConsoleDisplayFilter())
        root.addHandler(console_handler)

    current_handler = LOG_STATE["handler"]

    target_path = (
        pathlib.Path(output_file).resolve() if output_file else
        pathlib.Path(current_handler.baseFilename).resolve() if current_handler else
        (pathlib.Path("logs") / f"job_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log").resolve()
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if current_handler is None or pathlib.Path(current_handler.baseFilename).resolve() != target_path:
        new_handler = RichRotatingFileHandler(
            filename=target_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=1000,
            encoding="utf-8"
        )
        new_handler.addFilter(FileDisplayFilter())

        if current_handler:  # migrate existing loggers
            for tracked in list(LOG_STATE["loggers"]):
                if current_handler in tracked.handlers:
                    tracked.removeHandler(current_handler)
            current_handler.close()

        LOG_STATE["handler"] = new_handler
        current_handler = new_handler

    if current_handler not in root.handlers:
        root.addHandler(current_handler)

    for tracked in LOG_STATE["loggers"]:
        tracked.propagate = True

    logger.archive_logs = archive_logs
    sys.excepthook = handle_uncaught_exception
    threading.excepthook = handle_thread_exception

    return logger


def archive_logs(destination: Union[str, pathlib.Path]):
    """
    Moves current and rotated log files to a specified archive directory. Currently active log file
    is copied with a new incremental suffix. Original log files remain in place.

    Parameters
    ----------
    destination : str or pathlib.Path
        The directory where log files should be moved. Folder will be created if it does not exist.
    """
    dest_dir = pathlib.Path(destination)
    archiver_log = logging.getLogger("System.Archiver")

    handler = LOG_STATE["handler"]
    if not handler:
        archiver_log.warning("Archive requested but no file handler is active.")
        return

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        archiver_log.info(f"Archiving log files to: {dest_dir.resolve()}")

        handler.flush()

        base_file = pathlib.Path(handler.baseFilename)

        max_n, existing_backups = 0, []
        for f in base_file.parent.glob(f"{base_file.name}.*"):
            if (match := re.search(r"\.(\d+)$", f.name)):
                n = int(match.group(1))
                max_n = max(max_n, n)
                existing_backups.append(f)

        for backup in existing_backups:
            shutil.copy2(backup, dest_dir / backup.name)

        if base_file.exists():
            new_name = f"{base_file.name}.{max_n + 1}"
            shutil.copy2(base_file, dest_dir / new_name)

    except Exception as e:
        archiver_log.error(f"Failed to archive logs: {e}")


class LogContext:
    """
    Context managers for controlling logging behavior within a specific scopes.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to manipulate.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    @contextmanager
    def grouped_logs(self, worker_name: str) -> Generator[None, None, None]:
        """
        Buffers log output for the duration of the context.

        Realtime output is streamed to the console. File output is buffered and written as a
        single indented block with a header/footer upon exit into the file log only.

        Parameters
        ----------
        worker_name : str
            The name to display in the header/footer of the log block.
        """

        class SimpleBuffer(logging.Handler):
            def __init__(self):
                super().__init__()
                self.buffer = []

            def emit(self, record):
                msg_text = self.format(record)
                filename = os.path.basename(record.pathname)
                path_info = f"{filename}:{record.lineno}"
                formatted_lines, lines, target_width, n_indent = [], msg_text.split("\n"), 172, 4

                for i, line in enumerate(lines):
                    indented_line = " " * n_indent + line
                    if i == 0:
                        formatted_lines.append(indented_line.ljust(target_width - len(path_info)) + path_info)
                    else:
                        formatted_lines.append(indented_line)
                self.buffer.append("\n".join(formatted_lines))

        buffer = SimpleBuffer()
        buffer.setFormatter(logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        injector = RealtimeFilter()
        root = logging.getLogger()  # Attach to root: capture all loggers in process
        root.addHandler(buffer)
        root.addFilter(injector)

        for name in logging.root.manager.loggerDict:  # force propagation and clear handlers
            lg = logging.getLogger(name)
            if lg.hasHandlers():
                lg.handlers.clear()
            lg.propagate = True
            lg.setLevel(self.logger.level)

        try:
            yield
        finally:
            root.removeHandler(buffer)
            root.removeFilter(injector)
            if buffer.buffer:
                block = "\n".join(buffer.buffer)
                header = f"[bold cyan]{'=' * 30} START WORKER: {worker_name} {'=' * 30}[/]"
                footer = f"[bold cyan]{'=' * 31} END WORKER: {worker_name} {'=' * 31}[/]"
                self.logger.info(f"{header}\n{block}\n{footer}", extra={'summary': True, 'no_console': True})

    @contextmanager
    def parallel_session(self) -> Generator[dict, None, None]:
        """
        Initializes a QueueListener to handle logs from multiple processes safely.

        Starts a listener in the main process that directs worker logs to the configured
        handlers (console and file).

        Yields
        ------
        dict
            Configuration dictionary containing 'initializer' and 'initargs' that are to be
            passed to the ProcessPoolExecutor.
        """
        manager = multiprocessing.Manager()
        log_queue = manager.Queue()

        listener = logging.handlers.QueueListener(
            log_queue,
            *logging.getLogger().handlers,
            respect_handler_level=True
        )
        listener.start()
        pool_config = {
            "initializer": worker_init,
            "initargs": (log_queue, self.logger.getEffectiveLevel()),
        }

        try:
            yield pool_config
        finally:
            listener.stop()
            manager.shutdown()

    @contextmanager
    def redirect_tqdm(self) -> Generator[None, None, None]:
        """
        Redirects tqdm output to the logger's console handler.

        Ensures progress bars render correctly via Rich's Live display without duplications
        preventing spam in the console and log file. Works only in the main process.
        """
        handlers = self.logger.handlers + logging.getLogger().handlers
        handler = next((h for h in handlers if isinstance(h, RichHandler) and hasattr(h, 'console')), None)

        if handler:
            live = Live(console=handler.console, transient=True, refresh_per_second=20)
            tqdm_io = TqdmRichLiveIO(live, handler, self.logger.name)
            target_stream = UnclosableStream(tqdm_io)
            ctx_mgr = live

            def cleanup():
                self.logger.info(tqdm_io.last_text, stacklevel=3) if tqdm_io.last_text else None
        else:
            target_stream = LoggerStream(self.logger)
            ctx_mgr = nullcontext()

            def cleanup():
                return None

        original_init = tqdm.__init__

        def patched_init(p_self, *args, **kwargs):
            if 'file' not in kwargs:
                kwargs['file'] = target_stream
            if 'disable' not in kwargs:
                kwargs['disable'] = False
            original_init(p_self, *args, **kwargs)

        tqdm.__init__ = patched_init
        try:
            with ctx_mgr:
                yield
        finally:
            tqdm.__init__ = original_init
            cleanup()

    @contextmanager
    def suppress_console_logging(self) -> Generator[None, None, None]:
        """
        Temporarily prevents logs from appearing in the console.

        Logs are still written to the file handler.
        """
        suppress_filter = SuppressFilter()
        self.logger.addFilter(suppress_filter)
        try:
            yield
        finally:
            self.logger.removeFilter(suppress_filter)

    @contextmanager
    def duplicate_filter(self) -> Generator[None, None, None]:
        """
        Temporarily filters out consecutive duplicate log messages. They are logged only once.
        """
        if any(isinstance(f, DuplicateFilter) for f in self.logger.filters):
            yield
        else:
            dup_filter = DuplicateFilter()
            self.logger.addFilter(dup_filter)
            try:
                yield
            finally:
                self.logger.removeFilter(dup_filter)

    @contextmanager
    def logging_raised_Error(self) -> Generator[None, None, None]:
        """
        Catches any exception raised within the context, logs it with a traceback, and then
        re-raises the exception. This is automatically covered within setup_logging's excepthook
        and threading excepthook. Usage is only for cases where the global excepthook is
        replaced or unavailable.
        """
        try:
            yield
        except Exception as e:
            self.logger.exception(e)
            raise

    @contextmanager
    def set_logging_level(self, level: int) -> Generator[None, None, None]:
        """
        Temporarily changes the logger's severity level.

        Parameters
        ----------
        level : int
            The new logging level (e.g., logging.DEBUG).
        """
        _old_level = self.logger.level
        self.logger.setLevel(level)
        try:
            yield
        finally:
            self.logger.setLevel(_old_level)

    @contextmanager
    def suppress_logging(self) -> Generator[None, None, None]:
        """
        Temporarily disables all logging output by raising the level to CRITICAL + 1.
        """
        original_level = self.logger.level
        self.logger.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            self.logger.setLevel(original_level)

    @contextmanager
    def log_and_suppress(self, *exceptions: Type[Exception], msg: str = "An exception was suppressed"):
        """
        Catches specified exceptions, logs them, and suppresses propagation, i.e. for cases like:

        try:
            x = 1 / 0
        except ZeroDivisionError as e:
            msg = f"An exception was suppressed: {type(e).__name__} - {e}"
            logger.exception(msg, exc_info=True)

        Parameters
        ----------
        *exceptions : Type[Exception]
            Variable length argument list of exception types to catch.
        msg : str, default "An exception was suppressed"
            The message to log accompanying the traceback.
        """
        try:
            yield
        except exceptions or (Exception,) as e:
            self.logger.exception(f"{msg}: {type(e).__name__} - {e}", exc_info=True)

    @contextmanager
    def suppress_terminal_print(self) -> Generator[None, None, None]:
        """
        Redirects standard output (stdout) and standard error (stderr) to os.devnull.
        Useful for not flooding the terminal with output from libraries that print directly
        to the console.
        """
        if is_in_ipython():
            from IPython.display import display
            from ipywidgets import Output

            out = Output()
            with out:
                yield
        else:
            with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
                yield


class LogDecorator:
    """
    Decorators applying log context behaviors to functions.

    Parameters
    ----------
    logger : logging.Logger, optional
        The logger to use. If None, a logger corresponding to the function name is used.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.default_logger = logger

    def _create_resolver(
        self,
        extractor: Optional[Callable],
        mapper: Callable[[logging.Logger, Any, str], tuple[logging.Logger, tuple, dict]],
    ) -> Callable[[logging.Logger, Callable, Any, Any], tuple[logging.Logger, tuple, dict]]:
        def resolver(
            default_logger: logging.Logger,
            func: Callable,
            *args,
            **kwargs,
        ) -> tuple[logging.Logger, tuple, dict]:
            return mapper(
                default_logger,
                extractor(*args, **kwargs) if extractor else None,
                func.__name__,
            )
        return resolver

    def _create_decorator(
        self,
        context_getter: Callable[[logging.Logger], Callable],
        resolver: Optional[Callable[[logging.Logger, Callable, Any, Any], tuple[logging.Logger, tuple, dict]]] = None,
    ) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                logger, c_args, c_kwargs = self.default_logger or logging.getLogger(func.__name__), (), {}
                if resolver:
                    logger, c_args, c_kwargs = resolver(logger, func, *args, **kwargs)

                with context_getter(logger)(*c_args, **c_kwargs):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def grouped_logs(self, extractor: Optional[Callable] = None) -> Callable:
        """
        Decorator to buffer logs generated by the function.

        Parameters
        ----------
        extractor : Callable, optional
            A function that accepts the arguments of the decorated function
            and returns a string to use as the worker/group name.
            If None, the function name is used.

            example:
            ```python
            @log.LogDecorator().grouped_logs(lambda file: f"Worker-{file}")
            def myfunc(file):
                ...
            ```

        Returns
        -------
        Callable
            The decorated function.
        """

        def mapper(default_logger: logging.Logger, target: Any, func_name: str) -> tuple[logging.Logger, tuple, dict]:
            if target:
                if isinstance(target, logging.Logger):
                    return target, (target.name,), {}
                name = str(target)
            else:
                name = func_name
            return logging.getLogger(name), (name,), {}

        return self._create_decorator(
            lambda logger: LogContext(logger).grouped_logs,
            self._create_resolver(extractor, mapper),
        )

    def suppress_console_logging(self) -> Callable:
        """
        Decorator that suppresses console logging for the duration of the function.
        """
        return self._create_decorator(lambda logger: LogContext(logger).suppress_console_logging)

    def logging_raised_Error(self) -> Callable:
        """
        Decorator that logs any exception raised by the function before re-raising it.
        """
        return self._create_decorator(lambda logger: LogContext(logger).logging_raised_Error)
