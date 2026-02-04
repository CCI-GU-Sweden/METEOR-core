import logging  # info, warning, error and critical
import sys


class CustomFormatter(logging.Formatter):
    green = "\x1b[36m"
    grey = "\x1b[38m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31m"
    reset = "\x1b[0m"
    log_format = "%(process)d: %(asctime)s -%(levelname)s-: %(message)s"

    FORMATS = {
        logging.DEBUG: green + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y%m%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(level=logging.DEBUG):
    log_instance = logging.getLogger("METEOR_core")
    # Avoid adding multiple handlers by checking if they already exist
    if not log_instance.hasHandlers():
        fmtStr = "%(process)d: %(asctime)s -%(levelname)s-: %(message)s"

        if level == logging.DEBUG:
            logging.basicConfig(filename="METEOR_core.log", level=level, format=fmtStr)
            localLogger = logging.StreamHandler(sys.stdout)
            llFmt = CustomFormatter()
            localLogger.setFormatter(llFmt)
            localLogger.setLevel(level)
            log_instance.addHandler(localLogger)
        else:  # Only log to stdout
            logging.basicConfig(stream=sys.stdout, level=level, format=fmtStr)

    # 🔇 Silence OMERO internals
    logging.getLogger("omero").setLevel(logging.WARNING)
    logging.getLogger("omero.gateway").setLevel(logging.WARNING)
    logging.getLogger("omero.api.RawPixelsStore").setLevel(logging.WARNING)
    # Flask internal
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # Matplotlib
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    str_level = logging.getLevelName(level)
    info(f"Logger set up with level: {str_level}")


def log(level: str, msg: str):
    if level == "info":
        info(msg)
    elif level == "debug" or level == "dbg":
        debug(msg)
    elif level == "warning" or level == "warn":
        warning(msg)
    elif level == "error" or level == "err":
        error(msg)
    else:
        info(msg)


def info(msg: str):
    logger().info(msg)


def warning(msg: str):
    logger().warning(msg)


def error(msg: str):
    logger().error(msg)


def debug(msg: str):
    logger().debug(msg)


def logger():
    return logging.getLogger("METEOR_core")
