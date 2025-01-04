import cv2
import numpy as np

#print(cv2.getBuildInformation())
#print(cv2.cuda.getCudaEnabledDeviceCount())
#print(cv2.cuda.printCudaDeviceInfo(0))

try:
    # Load images
    frame1 = cv2.imread(r'./images/img_1.jpg', cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(r'./images/img_2.jpg', cv2.IMREAD_GRAYSCALE)
    if frame1 is None or frame2 is None:
        raise ValueError("Image files could not be loaded. Check the file paths.")

    # Image dimensions
    rows, cols = frame1.shape
    print(f"Image dimensions: {rows} x {cols}")
    print(f"Frame1 shape: {frame1.shape}, dtype: {frame1.dtype}")
    print(f"Frame2 shape: {frame2.shape}, dtype: {frame2.dtype}")

    # Detect corners in frame1 (good feature points)
    points_prev1 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if points_prev1 is None or len(points_prev1) == 0:
        raise ValueError("No points detected. Check the feature detection parameters.")
    
    gpu_test = cv2.cuda_GpuMat()
    gpu_test.upload(frame1)
    downloaded_frame = gpu_test.download()
    if downloaded_frame is None or downloaded_frame.shape != frame1.shape:
        print("Error: GpuMat upload/download failed.")
    else:
        print("GpuMat upload/download successful.")
    # Reshape for CUDA compatibility
    points_prev = points_prev1.astype(np.float32).reshape(1, -1, 2)  # Reshape for CUDA compatibility
    print(f"Points shape (host): {points_prev.shape}, dtype: {points_prev.dtype}")
    #print(f"Points data (host): {points_prev}")
    if points_prev is None or len(points_prev) == 0:
        raise ValueError("No feature points detected. Adjust `cv2.goodFeaturesToTrack` parameters.")

    # Upload images to GPU
    gpu_frame1 = cv2.cuda_GpuMat()
    gpu_frame2 = cv2.cuda_GpuMat()
    gpu_frame1.upload(frame1)
    gpu_frame2.upload(frame2)
    print(f"Frame1 on GPU shape: {gpu_frame1.download().shape}")
    print(f"Frame2 on GPU shape: {gpu_frame2.download().shape}")

    # Upload initial points to GPU
    gpu_points_prev = cv2.cuda_GpuMat()
    gpu_points_prev.upload(points_prev)
    print(f"Uploaded points shape to GPU: {gpu_points_prev.download().shape}")

    # Initialize SparsePyrLKOpticalFlow
    optical_flow = cv2.cuda.SparsePyrLKOpticalFlow.create(winSize=(15, 15), maxLevel=3)
    #print(optical_flow)
    print(f"Optical flow initialized with winSize={optical_flow.getWinSize()}, maxLevel={optical_flow.getMaxLevel()}")

    # Prepare output variables
    #empty_array = np.array([]).astype(np.float32).reshape(1, -1, 2)
    gpu_points_next = cv2.cuda_GpuMat()
    #gpu_points_next.upload(empty_array)   # Initialize as empty
    #print(gpu_points_next.empty())
    gpu_status = cv2.cuda_GpuMat()
    gpu_err = cv2.cuda_GpuMat()
    #gpu_frame1 = gpu_frame1.download()
    #gpu_frame2 = gpu_frame2.download()
    #print(gpu_points_next)
    try:
        if gpu_frame1.empty() or gpu_frame2.empty():
            raise ValueError("Failed to upload frames to GPU. Check the images or GPU memory.")
        #print("Uploaded points (GPU):", gpu_points_prev.download())
        try:
            gpu_points_next, gpu_status, gpu_err = optical_flow.calc(gpu_frame1, gpu_frame2, gpu_points_prev, None)
            #print("Output:", list(output))
        except cv2.error as e:
            print("CUDA Optical Flow Error:", e)
        #print(len(optical_flow.calc(gpu_frame1, gpu_frame2, gpu_points_prev, gpu_points_next, gpu_status)))
        points_next_host = gpu_points_next.download()
        status_host = gpu_status.download()
        error_host = gpu_err.download()
        if gpu_points_next.empty() or gpu_status.empty():
            print("Error: Output matrices are empty.")
        print("SparsePyrLK Optical Flow calculation successful.")
        print(f"Error: {error_host}")
        print(f"Next points: {points_next_host}")
        print(f"Status: {status_host}")
        print("Next points (GPU):", gpu_points_next.download())
        print("Status (GPU):", gpu_status.download())

    except Exception as e:
        print(f"Error during SparsePyrLK Optical Flow calculation: {e}")

    # Optional: Compare CPU version
    try:
        points_prev_cpu = points_prev1.astype(np.float32).reshape(-1, 1, 2)  # Reshape for CPU compatibility
        print(f"points_prev_cpu shape: {points_prev_cpu.shape}, dtype: {points_prev_cpu.dtype}")
        #print(f"points_prev_cpu data: {points_prev_cpu}")
        points_next_cpu, status_cpu, err_cpu = cv2.calcOpticalFlowPyrLK(
    frame1, frame2, points_prev_cpu, None, winSize=(15, 15), maxLevel=3
)

        if points_next_cpu is None or status_cpu is None:
            print("Error: CPU optical flow calculation failed.")
        else:
            print("CPU optical flow calculation successful.")
            #print(f"Next points (CPU): {points_next_cpu}")
            #print(f"Status (CPU): {status_cpu}")

    except Exception as e:
        print("Error during CPU optical flow:", str(e))

    # Check if any points were tracked successfully
    if not gpu_points_next.empty() and np.all(status_host == 0):
        raise ValueError("No valid optical flow points calculated. Check frame quality or parameters.")

    # Visualize the optical flow
    print("HERE")
    for prev, next_, status in zip(points_prev[:, 0], points_next_host[:, 0], status_host[:, 0]):
        if status == 1:  # Valid point
            prev_x, prev_y = prev
            next_x, next_y = next_
            cv2.arrowedLine(frame1, (int(prev_x), int(prev_y)), (int(next_x), int(next_y)), (255, 0, 0), 2)
    
    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Optical Flow", 400, 296)
    cv2.imshow("Optical Flow", frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("Error:", str(e))
