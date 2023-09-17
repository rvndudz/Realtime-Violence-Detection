from PIL import Image
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
import queue

FRAMES_TO_ANALYZE = 5  # number of frames to analyze in each iteration
FRAME_FREQUENCY = 0.4  # seconds

FRAME_WIDTH = 224
FRAME_HEIGHT = 224

output_queue = queue.Queue()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to(device)


if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

def get_probabilities_for_frame(image, labels=['violent scene', 'non-violent scene']):
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    probs_dict = {labels[i]: probs[0][i].item()
                  for i in range(len(labels))}
    return probs_dict

def annotate_frame(frame, probability_threshold):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (0, 0, 255)  # Red
    font_thickness = 2

    if probability_threshold >= 0.8:
        text = f"Violent Probability: {probability_threshold:.2f}"
        cv2.putText(frame, text, bottom_left_corner, font, font_scale, font_color, font_thickness)


def process_video(input_path):
    vs = cv2.VideoCapture(input_path)

    last_scores = []
    frame_count = 0

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        try:
            image = Image.fromarray(frame)
            image = image.convert("RGB")
            probs = get_probabilities_for_frame(image)
            violent_probability = probs["violent scene"]
        except Exception as e:
            violent_probability = 0

        last_scores.append(violent_probability)
        if len(last_scores) > FRAMES_TO_ANALYZE:
            last_scores = last_scores[1:]

        final_score = max(last_scores)
        output_queue.put((frame_count, final_score))

        annotate_frame(frame, final_score)

        cv2.imshow("Processed Video", frame)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

    # Print or process the results here
    while not output_queue.empty():
        frame_num, score = output_queue.get()
        print(f"Frame {frame_num}: Violent Probability - {score}")

    print("[INFO] Video processing complete.")

def process_webcam():
    vs = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        try:
            image = Image.fromarray(frame)
            image = image.convert("RGB")
            probs = get_probabilities_for_frame(image)
            violent_probability = probs["violent scene"]
        except Exception as e:
            violent_probability = 0

        annotate_frame(frame, violent_probability)

        cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Investigate a video file")
    print("2. Examine live webcam input")
    option = input("Enter the option (1/2): ")

    if option == "1":
        input_video_path = '1.mp4'  
        process_video(input_video_path)
    elif option == "2":
        process_webcam()
    else:
        print("Invalid option. Please choose 1 or 2.")
