import numpy as np
import imutils
import argparse
import cv2
import pytesseract
import tkinter as tk
from tkinter import messagebox

def align_image_to_self(image, maxFeatures=500, keepPercent=0.2, debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)

    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsA, None)  # Match image to itself

    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsA[m.trainIdx].pt  # Matched points are from the same image

    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    (h, w) = image.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    
    # Calculate rotation angle based on the homography matrix
    angle = np.arctan2(H[1, 0], H[0, 0]) * (180.0 / np.pi)
    
    # Rotate the aligned image to make it straight
    rotated_aligned = imutils.rotate_bound(aligned, -angle)  # Use imutils to handle rotation
    
    # Get screen resolution
    screen_width = 1920
    screen_height = 1080
    
    # Resize the rotated image to fit the screen without cropping or distortion
    aspect_ratio = rotated_aligned.shape[1] / rotated_aligned.shape[0]
    if aspect_ratio > screen_width / screen_height:
        resized_rotated_aligned = cv2.resize(rotated_aligned, (screen_width, int(screen_width / aspect_ratio)))
    else:
        resized_rotated_aligned = cv2.resize(rotated_aligned, (int(screen_height * aspect_ratio), screen_height))
    
    # Convert the resized and rotated image to grayscale
    grayscale_aligned_image = cv2.cvtColor(resized_rotated_aligned, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to make the image black and white
    _, binary_aligned_image = cv2.threshold(grayscale_aligned_image, 128, 255, cv2.THRESH_BINARY)

    return binary_aligned_image

def remove_horizontal_lines(image):
    # Create a horizontal kernel for morphological operations
    kernel = np.ones((2, 1), np.uint8)

    # Perform morphological closing to remove horizontal lines
    image_no_lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image_no_lines

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image that we'll align to template")
    args = vars(ap.parse_args())

    # Load the input image
    print("[INFO] loading image...")
    image = cv2.imread(args["image"])

    # Call the align_image_to_self function
    rotated_aligned_resized = align_image_to_self(image, maxFeatures=500, keepPercent=0.2, debug=True)

    # Convert the original image to grayscale
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the original grayscale image to match the height of the rotated_aligned_resized image
    original_resized_gray = cv2.resize(original_gray, (rotated_aligned_resized.shape[1], rotated_aligned_resized.shape[0]))

    # Perform OCR on the aligned binary image to extract text
    aligned_text = pytesseract.image_to_string(original_resized_gray)

    # Print the extracted text from the aligned image
    print("Aligned Image Text:")
    print(aligned_text)

    # Export the extracted text to a text file
    output_text_filename = args["image"].replace(".jpg", "_output.txt")
    with open(output_text_filename, "w") as text_file:
        text_file.write(aligned_text)


    # Rotate the images to display them in landscape view
    original_resized_gray_landscape = cv2.rotate(original_resized_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_aligned_resized_landscape = cv2.rotate(rotated_aligned_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Remove horizontal lines from the aligned image
    rotated_aligned_resized_no_lines = remove_horizontal_lines(rotated_aligned_resized_landscape)

    # Combine the images horizontally using hstack
    combined_image = np.hstack([original_resized_gray_landscape, rotated_aligned_resized_no_lines])

    # Create a window and add scrollbars
    window_name = "Comparison (Landscape View)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Horizontal Scroll", window_name, 0, combined_image.shape[1] - rotated_aligned_resized_no_lines.shape[1], lambda x: None)
    cv2.createTrackbar("Vertical Scroll", window_name, 0, combined_image.shape[0] - rotated_aligned_resized_no_lines.shape[0], lambda x: None)
    cv2.createTrackbar("Zoom", window_name, 100, 200, lambda x: None)

    # Save the rotated and aligned image to the same folder as the input image
    output_filename = args["image"].replace(".jpg", "_output.jpg")
    cv2.imwrite(output_filename, rotated_aligned_resized_landscape)

    # Save the rotated and aligned image with horizontal lines removed
    output_filename_no_lines = args["image"].replace(".jpg", "_output_no_lines.jpg")
    cv2.imwrite(output_filename_no_lines, rotated_aligned_resized_no_lines)

    # Perform OCR on the aligned binary image to extract text
    aligned_text = pytesseract.image_to_string(rotated_aligned_resized_no_lines)

    # Print the extracted text from the aligned image
    print("Lines removed Image Text:")
    print(aligned_text)

    # Export the extracted text to a text file
    output_text_filename = args["image"].replace(".jpg", "_LinesRemovedoutput.txt")
    with open(output_text_filename, "w") as text_file:
        text_file.write(aligned_text)

    # Display a message box to notify that the output image has been exported
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showinfo("Image Exported", f"The images are shown side by side here. Use scrollbars to view them. \n\nAlso, the output image has been exported to:\n{output_filename}")

    while True:
        # Get scrollbar positions
        h_scroll = cv2.getTrackbarPos("Horizontal Scroll", window_name)
        v_scroll = cv2.getTrackbarPos("Vertical Scroll", window_name)
        zoom = cv2.getTrackbarPos("Zoom", window_name) / 100

        # Calculate the size of the view with zoom
        view_height = int(rotated_aligned_resized_no_lines.shape[0] * zoom)
        view_width = int(rotated_aligned_resized_no_lines.shape[1] * zoom)

        # Create a view of the combined image based on scrollbar positions and zoom
        view = combined_image[v_scroll:v_scroll+view_height, h_scroll:h_scroll+view_width]

        # Resize the view for display
        display_view = cv2.resize(view, (rotated_aligned_resized_no_lines.shape[1], rotated_aligned_resized_no_lines.shape[0]))

        # Display the view in the window
        cv2.imshow(window_name, display_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press Esc to exit
            break

        # Check if the window was closed using the X button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()