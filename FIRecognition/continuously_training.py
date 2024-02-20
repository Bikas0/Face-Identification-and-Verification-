import schedule
import time
from updatepickle import new_pickle_file


def job():
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    new_pickle_file("/home/hajj_images", "/home/moudud/HajjProject/new_extract_face")
    print(f"{current_time}: Do the Task")


# Schedule the job to run every 5 minutes
# schedule.every(5).minutes.do(job)
# Schedule the job to run every day at 12 AM
schedule.every().day.at("00:00").do(job)

# Run the scheduled jobs
while True:
    schedule.run_pending()
    time.sleep(0)
