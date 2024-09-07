import os
import time

# Path to the .joblib file
file_path = 'pickup_cluster_model.joblib'

# Define the new modification time (timestamp)
# You can get the current time, or use a specific time
new_modification_time = time.mktime(time.strptime('2024-09-01 14:30:00', '%Y-%m-%d %H:%M:%S'))

# Set the file's modification and access times to the new time
os.utime(file_path, (new_modification_time, new_modification_time))