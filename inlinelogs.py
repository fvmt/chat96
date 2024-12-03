import logging
import re

class StreamlitLogHandler(logging.Handler):
    # Initializes a custom log handler with a Streamlit container for displaying logs
    def __init__(self, container):
        super().__init__()
        # Store the Streamlit container for log output
        self.container = container
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')  # Regex to remove ANSI codes
        self.log_area = self.container.empty()  # Prepare an empty container for log output

    def emit(self, record):
        msg = self.format(record)
        clean_msg = msg # self.ansi_escape.sub('', msg)  # Strip ANSI codes
        import time
        self.log_area.markdown(clean_msg)

    def clear_logs(self):
        pass
        self.log_area.empty()  # Clear previous logs


# Set up logging to capture all info level logs from the root logger
def setup_logging(st):
    root_logger = logging.getLogger()  # Get the root logger
    log_container = st.container()  # Create a container within which will display logs
    handler = StreamlitLogHandler(log_container)
    # handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return handler
