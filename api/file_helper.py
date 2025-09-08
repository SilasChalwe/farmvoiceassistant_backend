import os

UPLOAD_FOLDER = 'uploads'

def delete_file(filename):
    """
    Delete a file from the uploads folder if it exists.
    """
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def get_unique_filename(original_filename):
    """
    Generate a unique filename by appending a timestamp.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    return f"{timestamp}_{original_filename}"
