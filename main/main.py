import multiprocessing
from config import VIDEO_URLS
from camera import camera_process
import pycuda.autoinit


def main():
    # Create a process for each camera
    processes = []
    for idx, camera_url in enumerate(VIDEO_URLS):
        process = multiprocessing.Process(target=camera_process, args=(idx, camera_url))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Ensures proper initialization on Windows/macOS
    main()
