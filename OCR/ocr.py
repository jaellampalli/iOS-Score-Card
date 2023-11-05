import requests
import os
import re
import cv2
import numpy as np

# Replace the API_KEY with your OCR.space API key
API_KEY = 'K85675661788957'

def extract_text(image_path):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': API_KEY,
        'language': 'eng',
        'isTable': 'false',
        'OCREngine': '2',  # OCR.space OCR engine
        'filetype': 'jpg'  # Specify the file type
    }
    with open(image_path, 'rb') as file:
        image_data = file.read()
    files = {'image': image_data}
    response = requests.post(url, data=payload, files=files)
    response_data = response.json() if response else {}
    parsed_results = response_data.get('ParsedResults', [])
    if parsed_results:
        text_overlay = parsed_results[0].get('TextOverlay', {})
        parsed_text = parsed_results[0].get('ParsedText', '')
        return parsed_text, text_overlay
    return "", None

def extract_specific_data(parsed_text, target_format):
    try:
        print(f"Trying to match {parsed_text} with target_format {target_format}")
        regex_pattern = rf"{target_format}"
        matches = re.findall(regex_pattern, parsed_text, re.IGNORECASE)
        extracted_data = []
        for match in matches:
            for item in match:
                if isinstance(item, str):
                    item = item.strip()
                    if extracted_data and extracted_data[-1].isdigit() and item.isdigit():
                        # Concatenate consecutive digits into a single string
                        extracted_data[-1] += item
                    else:
                        extracted_data.append(item)
        return extracted_data
    except re.error:
        print("The given regular expression is invalid.")
        exit()

def export_text_to_file(label, text, output_file_path, target_format=""):
    if text:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(f"Label: {label}\nTarget Format: {target_format}\nValue: {text}")
        print(f"Extracted text saved as {output_file_path}")
    else:
        print("No text extracted from the image.")

def redact_image(image_path, target_label, target_format, offset, output_image_path):
    parsed_text, text_overlay = extract_text(image_path)
    if text_overlay:
        image = cv2.imread(image_path)
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        label_found = False
        label_line_index = 0

        if text_overlay and 'Lines' in text_overlay:
            for line_index, line in enumerate(text_overlay['Lines']):
                if 'Words' in line:
                    for word in line['Words']:
                        word_text = word.get('WordText', '').strip()
                        if target_label.lower() in word_text.lower():
                            left = int(word.get('Left', 0))
                            top = int(word.get('Top', 0))
                            width = int(word.get('Width', 0))
                            height = int(word.get('Height', 0))
                            #cv2.rectangle(image, (left, top), (left + width, top + height), (255, 0, 0), 2)  # Blue color for label
                            print(f"Label matched: {word_text}")  # Print label match
                            label_found = True
                            label_line_index = line_index
                            break

            if label_found:
                for line_index in range(label_line_index, label_line_index+offset+1):
                    line = text_overlay['Lines'][line_index]
                    if 'Words' in line:
                        #print(f"Current line forward for retaining {line_index}: {line}")  # Print label match
                        for word in line['Words']:
                            word_text = word.get('WordText', '').strip()
                            left = int(word.get('Left', 0))
                            top = int(word.get('Top', 0))
                            width = int(word.get('Width', 0))
                            height = int(word.get('Height', 0))
                            cv2.rectangle(mask, (left, top), (left + width, top + height), 0, -1)  # Black rectangle to exclude the label area

                for line_index in range(label_line_index, label_line_index-offset-1, -1):
                    line = text_overlay['Lines'][line_index]
                    if 'Words' in line:
                        #print(f"Current line backward for retaining {line_index}: {line}")  # Print label match
                        for word in line['Words']:
                            word_text = word.get('WordText', '').strip()
                            left = int(word.get('Left', 0))
                            top = int(word.get('Top', 0))
                            width = int(word.get('Width', 0))
                            height = int(word.get('Height', 0))
                            cv2.rectangle(mask, (left, top), (left + width, top + height), 0, -1)

        
        image[np.where(mask == 255)] = [255, 255, 255]  # Redact everything except the retained area with white color

        cv2.imwrite(output_image_path, image)
        print(f"Redacted image saved as {output_image_path}")
    else:
        print("No 'Lines' field found in text overlay or empty text overlay.")

def highlight_text_on_image(image_path, text_overlay, label, output_image_path, target_format, offset=1):
    image = cv2.imread(image_path)
    label_coordinates = None
    label_found = False

    if text_overlay and 'Lines' in text_overlay:
        for line_index, line in enumerate(text_overlay['Lines']):
            if 'Words' in line:
                for word in line['Words']:
                    word_text = word.get('WordText', '').strip()
                    if label.lower() in word_text.lower():
                        left = int(word.get('Left', 0))
                        top = int(word.get('Top', 0))
                        width = int(word.get('Width', 0))
                        height = int(word.get('Height', 0))
                        cv2.rectangle(image, (left, top), (left + width, top + height), (255, 0, 0), 2)  # Blue color for label
                        print(f"Label matched: {word_text}")  # Print label match
                        label_coordinates = (left, top, width, height)
                        label_found = True
                        break

            if label_found:
                break
    
    target_data_found = False

    if label_coordinates:
        line_counter = line_index

        for line_index in range(line_counter, line_counter+offset+1):
            if line_index < len(text_overlay['Lines']):
                line = text_overlay['Lines'][line_index]
                print(f"Current line {line_index} (forward): {line.get('LineText', '').strip()}")  # Print line
                if 'Words' in line:
                    extracted_data = extract_specific_data(line.get('LineText', '').strip(), target_format)
                    # Export the specific item to a text file if data is available
                    if extracted_data:
                        specific_data_text = ''.join(extracted_data)
                        specific_output_file_path = os.path.join(output_dir, f"{specific_output_file_prefix}_{base_filename}.txt")
                        export_text_to_file(label, specific_data_text, specific_output_file_path, target_format)

                    for word in line['Words']:
                        word_text = word.get('WordText', '').strip()
                        if any(extracted_word in word_text for extracted_word in extracted_data):
                            target_data_index = [extracted_word in word_text for extracted_word in extracted_data].index(True)
                            target_data = extracted_data[target_data_index]
                            target_data_left = int(word.get('Left', 0))
                            target_data_top = int(word.get('Top', 0))
                            target_data_width = int(word.get('Width', 0))
                            target_data_height = int(word.get('Height', 0))
                            cv2.rectangle(image, (target_data_left, target_data_top), (target_data_left + target_data_width, target_data_top + target_data_height), (0, 255, 0), 2)  # Green color for target data
                            extracted_data.pop(target_data_index)
                            extracted_data.insert(target_data_index, target_data)
                            print(f"Target data matched (forward): {target_data}")  # Print target data match
                            target_data_found = True
                            break
                    if target_data_found == True:
                            export_text_to_file(label, target_data, specific_output_file_path, target_format)
                            break
    if not target_data_found and offset > 0:
        line_counter = line_index

        for line_index in range(line_counter, line_counter-offset-1, -1):
            if line_index < len(text_overlay['Lines']):
                line = text_overlay['Lines'][line_index]
                print(f"Current line {line_index} (backward): {line.get('LineText', '').strip()}")  # Print line
                if 'Words' in line:
                    extracted_data = extract_specific_data(line.get('LineText', '').strip(), target_format)
                    # Export the specific item to a text file if data is available
                    if extracted_data:
                        specific_data_text = ''.join(extracted_data)
                        specific_output_file_path = os.path.join(output_dir, f"{specific_output_file_prefix}_{base_filename}.txt")
                        export_text_to_file(label, specific_data_text, specific_output_file_path, target_format)

                    for word in line['Words']:
                        word_text = word.get('WordText', '').strip()
                        if any(extracted_word in word_text for extracted_word in extracted_data):
                            target_data_index = [extracted_word in word_text for extracted_word in extracted_data].index(True)
                            target_data = extracted_data[target_data_index]
                            target_data_left = int(word.get('Left', 0))
                            target_data_top = int(word.get('Top', 0))
                            target_data_width = int(word.get('Width', 0))
                            target_data_height = int(word.get('Height', 0))
                            cv2.rectangle(image, (target_data_left, target_data_top), (target_data_left + target_data_width, target_data_top + target_data_height), (0, 255, 0), 2)  # Green color for target data
                            extracted_data.pop(target_data_index)
                            extracted_data.insert(target_data_index, target_data)
                            print(f"Target data matched (backward): {target_data}")  # Print target data match
                            target_data_found = True
                            break

    if not target_data_found:
        print("No 'Lines' field found in text overlay or empty text overlay.")

    cv2.imwrite(output_image_path, image)
    print(f"Highlighted image saved as {output_image_path}")

def process_image(image_path):
    # Extract text from the image and get text overlay information
    original_parsed_text, original_text_overlay = extract_text(image_path)

    # Export the full page extraction to a text file
    output_file_path = os.path.join(output_dir, f"{output_file_prefix}_extracted_text_{base_filename}.txt")
    export_text_to_file('Full Page', original_parsed_text, output_file_path)

    # Prompt the user for specific data extraction inputs
    desired_label = input("Enter the desired label: ")
    target_format = input("Enter the target data format (as a regular expression pattern): ")
    offset = int(input("Enter the number of lines to retain while redacting the rest: "))

    # Redact the image based on the extracted data
    redacted_image_path = os.path.join(output_dir, f"redacted_{base_filename}.jpg")
    redact_image(image_path, desired_label, target_format, offset, redacted_image_path)

    # Extract full page OCR from the redacted image
    redacted_parsed_text, redacted_text_overlay = extract_text(redacted_image_path)

    # Export the full page OCR of the redacted image to a text file
    redacted_output_file_path = os.path.join(output_dir, f"redacted_{output_file_prefix}_extracted_text_{base_filename}.txt")
    export_text_to_file(f"Full Page (Redacted with {offset} lines retained)", redacted_parsed_text, redacted_output_file_path, target_format)

    # Extract specific data from the redacted OCR
    #extracted_data = extract_specific_data(redacted_parsed_text, target_format)

    # Highlight the text and extract specific data from the redacted image
    highlighted_image_path = os.path.join(output_dir, f"{specific_output_file_prefix}_highlighted_{base_filename}.jpg")
    highlight_text_on_image(redacted_image_path, redacted_text_overlay, desired_label, highlighted_image_path, target_format,offset)

    

    # Export the specific item to a text file
    specific_output_file_path = os.path.join(output_dir, f"{specific_output_file_prefix}_{base_filename}.txt")
    #export_text_to_file('Specific Item', extracted_data, specific_output_file_path)

# Set the paths
image_folder = r'C:\Users\ajabi\Desktop\OCR\scorecard12'
image_file = 'scorecard12.jpg'
output_dir = image_folder

# Generate output file names
base_filename = os.path.splitext(os.path.basename(image_file))[0]
output_file_prefix = "OCR.space_Original"  # Specify the OCR engine name
specific_output_file_prefix = "Specific"  # Specify the prefix for specific item output file

# Construct the full image path
image_path = os.path.join(image_folder, image_file)

process_image(image_path)


