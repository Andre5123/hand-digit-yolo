# This script captures training images on the fly by taking pictures every few seconds automatically while providing cues for the user to adhere to in each photo. Press q to quit.

import cv2
import time

cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

frozen = False
countdown_start = None

image_counter = 0
with open("image_counter.txt", "r") as f:
    image_counter = int(f.read().strip())

capture_sequence_captions = [] # Customize this. It will display these captions for however many images you choose. Use them as visual cues to remind you what image you want to take.



break_interval = 10 # After how many photo captures should you take a break? 
break_time = 5 # How long the break should last
countdown_time = 0.5 # How long the break should last


single_hand = False
# Single hand images. Remember to set break interval to 17.
if single_hand:
    capture_sequence_captions.extend(["0 Left"]*50)
    capture_sequence_captions.extend(["0 Right"]*50)
    capture_sequence_captions.extend(["1 Left"]*50)
    capture_sequence_captions.extend(["1 Right"]*50)
    capture_sequence_captions.extend(["2 Left"]*50)
    capture_sequence_captions.extend(["2 Right"]*50)
    capture_sequence_captions.extend(["3 Left"]*50)
    capture_sequence_captions.extend(["3 Right"]*50)
    capture_sequence_captions.extend(["4 Left"]*50)
    capture_sequence_captions.extend(["4 Right"]*50)
    capture_sequence_captions.extend(["5 Left"]*50)
    capture_sequence_captions.extend(["5 Right"]*50)
else: # Toggle for adding two hand pictures. Remember to set break interval to 4. 
    capture_sequence_captions.extend(["5 Left - 5 Right"]*10)
    capture_sequence_captions.extend(["0 Left - 0 Right"]*20)

    capture_sequence_captions.extend(["0 Left - 1 Right"]*10)
    capture_sequence_captions.extend(["0 Left - 2 Right"]*10)
    capture_sequence_captions.extend(["0 Left - 3 Right"]*10)
    capture_sequence_captions.extend(["0 Left - 4 Right"]*10)
    capture_sequence_captions.extend(["0 Left - 5 Right"]*10)
    capture_sequence_captions.extend(["1 Left - 0 Right"]*10)
    capture_sequence_captions.extend(["2 Left - 0 Right"]*10)
    capture_sequence_captions.extend(["3 Left - 0 Right"]*10)
    capture_sequence_captions.extend(["4 Left - 0 Right"]*10)
    capture_sequence_captions.extend(["5 Left - 0 Right"]*10)

    capture_sequence_captions.extend(["1 Left - 1 Right"]*20)
    
    capture_sequence_captions.extend(["1 Left - 2 Right"]*10)
    capture_sequence_captions.extend(["1 Left - 3 Right"]*10)
    capture_sequence_captions.extend(["1 Left - 4 Right"]*10)
    capture_sequence_captions.extend(["1 Left - 5 Right"]*10)
    capture_sequence_captions.extend(["2 Left - 1 Right"]*10)
    capture_sequence_captions.extend(["3 Left - 1 Right"]*10)
    capture_sequence_captions.extend(["4 Left - 1 Right"]*10)
    capture_sequence_captions.extend(["5 Left - 1 Right"]*10)

    capture_sequence_captions.extend(["2 Left - 2 Right"]*20)

    capture_sequence_captions.extend(["2 Left - 3 Right"]*10)
    capture_sequence_captions.extend(["2 Left - 4 Right"]*10)
    capture_sequence_captions.extend(["2 Left - 5 Right"]*10)
    capture_sequence_captions.extend(["3 Left - 2 Right"]*10)
    capture_sequence_captions.extend(["4 Left - 2 Right"]*10)
    capture_sequence_captions.extend(["5 Left - 2 Right"]*10)

    capture_sequence_captions.extend(["3 Left - 3 Right"]*20)

    capture_sequence_captions.extend(["3 Left - 4 Right"]*10)
    capture_sequence_captions.extend(["3 Left - 5 Right"]*10)
    capture_sequence_captions.extend(["4 Left - 3 Right"]*10)
    capture_sequence_captions.extend(["5 Left - 3 Right"]*10)

    capture_sequence_captions.extend(["4 Left - 4 Right"]*20)

    capture_sequence_captions.extend(["4 Left - 5 Right"]*10)
    capture_sequence_captions.extend(["5 Left - 4 Right"]*10)

    capture_sequence_captions.extend(["5 Left - 5 Right"]*20)



for i in range(len(capture_sequence_captions)):

    if i % break_interval == 0: #A break after every 4 captures giving the option to quit
        ret, frame = cam.read()
        print("breaktime")
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Break time. Next up: {capture_sequence_captions[i]}, image {image_counter+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255.0,0), 4)
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(break_time*1000)  # wait for a few seconds
        if key == ord('q'):
            break
        else:
            pass
            # No key pressed, or key != 'q' (quit).
        

    # Take a picture every 5 seconds
    image_counter += 1
    countdown_start = time.time()
    elapsed = 0
    remaining = countdown_time - elapsed

    while elapsed < countdown_time:
        print("countdown started")
        elapsed = time.time() - countdown_start
        remaining = countdown_time - int(elapsed)

        ret, display = cam.read()
        display = cv2.flip(display, 1)
        cv2.putText(display, capture_sequence_captions[i], (frame_width//4, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255.0,0), 4)
        cv2.putText(display, str(remaining), (frame_width//2, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0,255.0,0), 4)
        cv2.imshow('Camera', display)
        cv2.waitKey(1)

    ret, capturedImage = cam.read()
    cv2.imwrite(f"../../data/custom/images/frame_{image_counter:05d}.jpg", capturedImage) # Save the current frame to images
    print("image taken")

with open("image_counter.txt", "w") as f:
    f.write(str(image_counter))
cam.release()
cv2.destroyAllWindows()