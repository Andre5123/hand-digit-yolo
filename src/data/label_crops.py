from pathlib import Path
import cv2
import shutil
import random

crops_dir = Path("../../data/custom/crops")
unlabelled_dir = crops_dir / "unlabelled" # After cropping all of the crops end up in this folder to be classified into categories later.
unclear_dir = crops_dir / "unclear" # This is where crops that do not fit into one digit label go (crops that are unclear)

img_paths = list(unlabelled_dir.glob("*.jpg"))
for img_path in img_paths:
    img = cv2.imread(str(img_path))
    display = img.copy()
    cv2.imshow(img_path.stem, display)
    done = False
    quit = False
    while not done:
        key = cv2.waitKey(1)
        if key in range(ord('0'), ord('5')+1):
            done = True
            digit = int(chr(key))
            print("Box class: ", digit)
            shutil.move(str(img_path), str(crops_dir / str(digit) / (img_path.stem+".jpg")))

        elif key == ord("u"): # Put in the unclear category
            done = True
            shutil.move(str(img_path), str(unclear_dir / (img_path.stem+".jpg")))
        elif key == ord("q"):
            quit = True
            break
    cv2.destroyAllWindows()
    if quit == True:
        break

    