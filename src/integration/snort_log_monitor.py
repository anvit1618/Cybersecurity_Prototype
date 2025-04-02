import time
import os

def follow(thefile):
    """Generator function that yields new lines in a file."""
    thefile.seek(0, os.SEEK_END) # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1) # Sleep briefly
            continue
        yield line

if __name__ == '__main__':
    # Example usage: Monitor 'snort.log'
    log_file_path = 'snort.log'
    try:
        with open(log_file_path, 'r') as logfile:
            print(f"Monitoring {log_file_path} for new lines...")
            loglines = follow(logfile)
            for line in loglines:
                print(line, end='') # Print new lines as they appear
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")