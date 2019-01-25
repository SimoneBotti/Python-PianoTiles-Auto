import time
import pyautogui
from mss import mss
from PIL import Image
import numpy
import cv2
import imutils



def displayMultiplePic(im):
    cv2.imshow("OpenCV/Numpy normal", im)



def displayPicture(im):
    cv2.imshow("OpenCV/Numpy normal", im)
    cv2.waitKey(0)



def putMouseinPosition(x,y,height):
    startX=655
    startY=32
    offH = (height / 2) - 15
    pyautogui.moveTo(startX+x, startY+y+offH)
    print("Putted Mouse in X: "+str(startX+x)+" Y: "+str(startY+y)+" Position New: "+str(offH))

def putMouseinPositionandClick(x,y,height):
    startX=655
    startY=32
    offH=(height/2)-15
    pyautogui.click(startX+x, startY+y+offH)
    print("Putted Mouse in X: "+str(startX+x)+" Y: "+str(startY+y)+" Position New: "+str(offH))


def putMouseinPositionandClick(x,y):
    startX=655
    startY=32
    if(y<30):
        y+=150
    if(y<100):
        y+=100
    pyautogui.click(startX+x, startY+y)
    print("Putted Mouse in X: "+str(startX+x)+" Y: "+str(startY+y))


#SORT CONTOURS BOTTOM TO TOP
def sort_contours(cnts, method="bottom-to-top"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def openCVmultipleScreen():
    with mss() as sct:
        monitor = {"top": 32, "left": 655, "width": 605, "height": 753}

      #  last_time = time.time()
      #  ti = time.time()
      #  print("Riconoscimento black square: " + str(ti - last_time))
        while "Screen capturing":
            im = sct.grab(monitor)
            image = cv2.cvtColor(numpy.array(im), cv2.COLOR_BGR2RGB)
            lower = numpy.array([0, 0, 0])
            upper = numpy.array([70, 170, 250])
            shapeMask = cv2.inRange(image, lower, upper)
            #displayMultiplePic(shapeMask)
            cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            orig = image.copy()
            #print("I found {} black shapes".format(len(cnts)))
            # Sorting the countours
            (cnts, boundingBoxes) = sort_contours(cnts)
            threshold_area = 10000
            # loop over the contours
            for c in cnts:
                area = cv2.contourArea(c)
                if area > threshold_area:
                    x, y, w, h = cv2.boundingRect(c)
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # print("Contorno x: "+str(x)+" Y: "+str(y)+" Width: "+str(w)+" Height: "+str(h));
                    putMouseinPositionandClick(cX, cY)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break





def openCV():
    with mss() as sct:
        monitor = {"top": 32, "left": 655, "width": 605, "height": 753}
        # Get raw pixels from the screen, save it to a Numpy array

        last_time = time.time()

        im = sct.grab(monitor)
        image = cv2.cvtColor(numpy.array(im), cv2.COLOR_BGR2RGB)
        lower = numpy.array([0, 0, 0])
        upper = numpy.array([70, 170, 250])
        shapeMask = cv2.inRange(image, lower, upper)
        #displayPicture(shapeMask)
        ti=time.time()
        print("Riconoscimento black square: "+str(ti-last_time))
        cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        orig = image.copy()
        print("I found {} black shapes".format(len(cnts)))

        #Sorting the countours
        (cnts, boundingBoxes) = sort_contours(cnts)
        threshold_area = 1000
        # loop over the contours
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            print("Area :"+str(area))
            if area>threshold_area:

                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print("Contorno x: "+str(x)+" Y: "+str(y)+" Width: "+str(w)+" Height: "+str(h));
                putMouseinPosition(cX,cY,h)
                cv2.drawContours(shapeMask, [c], 0, (50, 50, 50), 3)
                time.sleep(2)
            #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
        displayPicture(shapeMask)
        # find the contours in the mask




def captureOneScreen():
    columns=4
    width=604
    height=752

    with mss() as sct:
        monitor = {"top": 32, "left": 655, "width": 605, "height": 753}
        # Get raw pixels from the screen, save it to a Numpy array
       # im=sct.grab(monitor)
        img = numpy.array(sct.grab(monitor))

        for w in range(width):
            for h in range(height):
                if(img[w,h,0]<20 and img[w,h,1]<20 and img[w,h,2]<20 ):
                    print("Found a black pixel["+str(w)+","+str(h)+"]: "+str(img[w,h]))

        print("Pixel in 75,75: "+str(img[604,10]))
        # displayPicture(img)




def captureScreenshot():
    with mss() as sct:
        monitor={"top": 32, "left": 655, "width": 605, "height": 753}

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            print()
            #displayPicture(img)
           # print("fps: {}".format(1 / (time.time() - last_time)))

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
               cv2.destroyAllWindows()
               break


def moveMouseandClick(x,y):
    pyautogui.click(x, y)

def moveMouse(x,y):
    pyautogui.moveTo(x, y)



def main():
    openCVmultipleScreen()


if __name__== "__main__":
  main()
