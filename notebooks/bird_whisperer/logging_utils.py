import os
import sys
import time

# Define a Tee class to duplicate output to both stdout and a log file
class Tee:
  def __init__(self, *files):
    self.files = files

  def write(self, text):
    for file in self.files:
      file.write(text)
      file.flush()
 
  def flush(self):
    for file in self.files:
        file.flush()

# Define a function to redirect stdout and stderr to a log file
def redirect_output_to_log(log_file_path):
    # Open the log file in append mode
    log = open(log_file_path, "a")
 
    # Duplicate stdout and stderr
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)

    return log

# Define a function to setup logging
def setup_logging():
  logs_dir = os.path.join(os.getcwd(), "logs")
  os.makedirs(logs_dir, exist_ok=True)

  timestamp = time.time()

  log_file_path = os.path.join(logs_dir, f"{timestamp}.log")
  json_log_file_path = os.path.join(logs_dir, f"{timestamp}.json")

  log_file = redirect_output_to_log(log_file_path) # redirect terminal output to log file

  print(f"Logging to {log_file_path}")

  return log_file, json_log_file_path