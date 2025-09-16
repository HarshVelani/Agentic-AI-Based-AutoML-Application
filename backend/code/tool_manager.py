from code.pipeline_generator import PipelineGenerator
import datetime

class ToolManager:
    def __init__(self):
        # Initialize tools
        current_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S_")
        self.session_id = f"{current_time}_session"