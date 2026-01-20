# CustomLogging Submodule

A wrapper for standard Python logging providing Rich formatting, multiprocessing safety via QueueListeners, and tqdm integration.

## Dependencies

- rich
- tqdm

## Setup

After adding to the project:

```python
import logging
import CustomLogging as log

# Initializes handlers, formatting, and file rotation (default 50MB)
# Automatically sets sys.excepthook to catch crashes.
logger = log.setup_logging(
    logger=logging.getLogger(__name__),
    output_file="logs/app.log",  # Optional: defaults to timestamped file in logs/
    max_file_size_mb=10,  # Will create a new file after current one exceeds 10 MB
    console_markup=False         # Set True to allow [bold red]Rich[/] markup in logs
)
```

## Multiprocessing

Standard logging is not process-safe. Use `parallel_session` to initialize a `QueueListener` in the main process and pass configuration to workers.

```python
from concurrent.futures import ProcessPoolExecutor

# 1. Start session (starts QueueListener)
with log.LogContext(logger).parallel_session() as pool_config:
    # 2. Pass pool_config to executor
    with ProcessPoolExecutor(max_workers=4, **pool_config) as executor:
        executor.submit(my_func)
```

## tqdm Integration

Prevents progress bars from conflicting with log output (working for non multiprocessing use cases)

```python
from tqdm import tqdm

with log.LogContext(logger).redirect_tqdm():
    for item in tqdm(items):
        logger.info("Processing...") # Prints above the progress bar, progress bar appears only once
```

## Context Managers & Decorators

Utilities are available via `log.LogContext(logger)` (Context Manager) or `@log.LogDecorator()` (Decorator). `logger` is optional in case of `@log.LogDecorator`

### Grouped Logs
Buffers logs from a block.
- **Console**: Realtime output.
- **File**: Writes as a single indented summary block with header/footer upon completion.

**Context Manager:**
```python
with log.LogContext(logger).grouped_logs("Worker-1"):
    logger.info("Task started")
    logger.info("Task Running")
```

**Decorator:**
Defaults to function name. Optional `extractor` allows dynamic naming based on function arguments.
```python
import logging
from CustomLogging import Logging as log

logger = logging.getLogger(__name__)
@log.LogDecorator(logger).grouped_logs()
def process_data():
    ...

# or

@log.LogDecorator().grouped_logs(extractor=lambda x: f"{x}")  # be provided file_path
def process_file(file_path):
    ...
```

### Suppress Console Logging
Logs strictly to the file handler; suppresses console output.

```python
# Context
with log.LogContext(logger).suppress_console_logging():
    logger.info("File only")

# Decorator
@log.LogDecorator(logger).suppress_console_logging()
def silent_task():
    ...
```

### Logging Raised Error
If the block/function raises an exception, it is logged immediately before bubbling up. All globally raised Errors are logged automatically. This functionality is for cases where this functionality is removed.

```python
# Context
with log.LogContext(logger).logging_raised_Error():
    raise ValueError("Logged automatically")

# Decorator
@log.LogDecorator().logging_raised_Error()
def risky_func(): ...
```

### Log and Suppress
Catches specified exceptions, logs them with a traceback, and prevents the program from crashing. Context Manager only.

```python

# will wrapp this functionality 
try:
    x = 1 / 0
except ZeroDivisionError as e:
    msg = f"An exception was suppressed: {type(e).__name__} - {e}"
    logger.exception(msg, exc_info=True)

# to

with log.LogContext(logger).log_and_suppress(ZeroDivisionError):
    x = 1 / 0  # Logged as error, execution continues
```

## Utility Context Managers

These are available only as Context Managers via `log.LogContext(logger)`.

| Function | Description |
| :--- | :--- |
| `duplicate_filter()` | Temporarily suppresses identical consecutive log messages. |
| `suppress_terminal_print()` | Redirects `stdout` and `stderr` to `os.devnull`. |
| `set_logging_level(level)` | Temporarily changes the logger level (e.g., to `logging.DEBUG`). |
| `suppress_logging()` | Temporarily disables all logging (sets level to CRITICAL+1). |

## Log Archival

Moves current log files (including all rotated backups) to a specified directory.

```python
logger.archive_logs("backup/logs/")
```
