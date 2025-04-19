import logging
import time
from colors import Colors # Assuming 'colors' library is installed (pip install ansicolors) or your custom Colors class

def setup_logging(level=logging.INFO) -> None:
    # Check if the root logger already has handlers to avoid adding them multiple times
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        # It's generally recommended to set the level on the logger itself
        root_logger.setLevel(level)
        # Prevent propagation if you are adding handlers here,
        # otherwise messages might get duplicated if a parent logger also has handlers.
        # However, since this IS the root logger, propagation doesn't go further up.
        # root_logger.propagate = False # Optional, but good practice if adding handlers here

        # --- Define Custom Record Factory FIRST ---
        old_factory = logging.getLogRecordFactory()

        def custom_record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            now = time.localtime(record.created)
            cs = int((record.created % 1) * 100)  # centiseconds
            # Create a custom attribute for the short time format
            #record.short_time = time.strftime("%H:%M:%S", now) + f".{cs:02d}"
            record.short_time = time.strftime("%M:%S", now) + f".{cs:02d}"
            return record

        logging.setLogRecordFactory(custom_record_factory)
        # --- Custom Record Factory Set ---


        # --- Define Format String using the Custom Attribute ---
        #prefix = Colors.apply("🖥️").gray
        prefix = Colors.apply("🖥️").gray
        # Use the custom attribute '%(short_time)s' instead of '%(asctime)s'
        timestamp = Colors.apply("%(short_time)s").blue
        #levelname = Colors.apply("%(levelname)s").green.bold
        levelname = Colors.apply("%(levelname)-4.4s").green.bold
        #message = Colors.apply("%(message)s").yellow
        message = Colors.apply("%(message)s")
        # logger_name = Colors.apply("%(name)s").gray
        logger_name = Colors.apply("%(name)-10.10s").gray


        # Note: Using f-string here means Colors.apply runs *once* during setup.
        # The %(...) placeholders are evaluated by the Formatter for each log record.
        log_format = f"{timestamp} {logger_name} {levelname} {message}"
        # --- Format String Defined ---


        # --- Configure logging using basicConfig OR explicit Handler/Formatter ---

        # Option 1: Using basicConfig (simpler for basic cases)
        logging.basicConfig(
            level=level, # Level set here applies to the handler created by basicConfig
            format=log_format     # Pass the format string using %(short_time)s
        )
        # Set the converter *after* basicConfig or on the specific formatter if created manually
        # Note: This global change affects ALL formatters unless they override it.
        logging.Formatter.converter = time.localtime
