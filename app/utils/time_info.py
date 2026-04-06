from datetime import datetime

def get_time_information() -> str:
    now = datetime.now()
    return now.strftime("%A, %b %d, %Y, %I:%M %p")