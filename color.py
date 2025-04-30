# color.py

class Color:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Additional colors
    LIGHT_GRAY = "\033[90m"
    GRAY = "\033[91m"
    LIME = "\033[92m"
    ORANGE = "\033[93m"
    LIGHT_BLUE = "\033[94m"
    LIGHT_MAGENTA = "\033[95m"
    LIGHT_CYAN = "\033[96m"
    BOLD_WHITE = "\033[97m"

    @staticmethod
    def reset():
        return "\033[0m"  # Reset color to default
