# A script that annotates any unlabelled images captured by capture.py with bounding boxes around the hands. 
# Press b, then click and drag to apply a bounding box. 
# Press d when you are done annotating an image to move on to the next one.

from pathlib import Path
import cv2

imgs_dir = Path("../../data/custom/images")
labels_dir = Path("../../data/custom/labels")
crops_dir = Path("../../data/custom/crops")
unlabelled_dir = crops_dir / "unlabelled" # After cropping all of the crops will end up in this folder to be classified into categories later.

for img_path in imgs_dir.glob("*.jpg"):
    label_path = labels_dir / (img_path.stem +".txt") # This checks if the image already has an annotated txt file (sharing the same name). If it does not, then it has not been annotated yet
    if label_path.exists():
        continue
    
    img = cv2.imread(str(img_path))
    display = img.copy()
    cv2.putText(display, img_path.stem, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
    cv2.imshow(img_path.stem, display)

    img_height, img_width, img_channels = img.shape

    done = False
    quit = False

    crop_number = 0
    while not done:
        key = cv2.waitKey(1)
        if key == ord('d'):

            if not label_path.exists(): # Create a blank txt file for images with no objects 
                label_path.touch()

            done = True
            continue
        if key == ord('q'):
            done = True
            quit = True
            continue
        if key == ord("b"): #Put a box.
            crop_number += 1
            roi = cv2.selectROI(img_path.stem, display) #User clicks and drags to put a bounding box around a hand.
            x_pixels = roi[0]
            y_pixels = roi[1]
            w_pixels = roi[2]
            h_pixels = roi[3]
            # Visualize the box & digit number
            cv2.rectangle(display, (x_pixels, y_pixels), (x_pixels+w_pixels, y_pixels+h_pixels), (0,255,0), 4)
            cv2.imshow(img_path.stem, display)

            print("Box dimensions", x_pixels, y_pixels, w_pixels, h_pixels)
            
            x = (x_pixels+w_pixels/2)/img_width
            y = (y_pixels+h_pixels/2)/img_height
            w = (w_pixels)/img_width
            h = (h_pixels)/img_height

            with open(f"../../data/custom/labels/{img_path.stem}.txt", "a") as f:
                f.write(f"0 {x} {y} {w} {h}\n")
            crop = img[y_pixels:y_pixels+h_pixels, x_pixels:x_pixels+w_pixels]
            crop = cv2.resize(crop, (128,128))
            cv2.imwrite(str(unlabelled_dir / f"crop_{img_path.stem[-5:]}-{crop_number}.jpg"), crop) # Add the crop of the hand
            
    cv2.destroyAllWindows()
    if quit == True:
        break
             