from enum import Enum
from typing import Type

DEBUG = True #choose debug mode #TODO: create config file

TYPE_SUCCESS = "SUCCESS"
TYPE_DEBUG = "DEBUG"
TYPE_ERROR = "ERROR"
TYPE_WARNING = "WARNING"
TYPE_INFO = "INFO"

LOG_LVL = [TYPE_ERROR, TYPE_WARNING, TYPE_INFO, TYPE_SUCCESS] #choose log level here TODO: create config file

class Debug(Enum):
  SUCCESS = TYPE_SUCCESS
  DEBUG = TYPE_DEBUG
  ERROR = TYPE_ERROR
  WARNING = TYPE_WARNING
  INFO = TYPE_INFO

class Color(Enum):
  PURPLE = '\033[95m'
  BLUE = '\033[94m'
  CYAN = '\033[96m'
  GREEN = '\033[92m'
  ORANGE = '\033[93m'
  RED = '\033[91m'
  CLEAR = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

ColorMap = {
  Debug.SUCCESS: Color.GREEN.value,
  Debug.ERROR: Color.RED.value + Color.BOLD.value,
  Debug.DEBUG: Color.PURPLE.value,
  Debug.INFO: Color.CYAN.value,
  Debug.WARNING: Color.ORANGE.value
}

def print_debug(*args, where: str = None, type: Type[Debug] = Debug.DEBUG, color: bool = True):
  if not DEBUG and type.value not in LOG_LVL:
    return
  color_start = ColorMap[type] if color else ""
  if where:
    print(f"{color_start}[{type.value}] ({where}):", *args, Color.CLEAR.value)
  else:
    print(f"{color_start}[{type.value}]:", *args, Color.CLEAR.value)