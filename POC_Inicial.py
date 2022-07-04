from cvzone.HandTrackingModule import HandDetector
import cvzone
import cv2
import pickle
import numpy as np

# cap = cv2.VideoCapture(0)
# cap.set(3, 680)
# cap.set(4, 680)
# detector = HandDetector(detectionCon=0.8, maxHands=2)
# colorR = (255, 0, 255)
#
# width, height = 107, 48
# CarParkPos = 'CarParkPos'
# with open(CarParkPos, 'rb') as f:
#     posList = pickle.load(f)
#
# def detect_object(img):
#
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
#     imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                          cv2.THRESH_BINARY_INV, 25, 16)
#     imgMedian = cv2.medianBlur(imgThreshold, 5)
#     kernel = np.ones((3, 3), np.uint8)
#     imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
#
#     # Find Contours
#     # contours, hierarchy = cv2.findContours(imgThreshold, 1, 2)
#     contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     cnt = contours[0]
#     M = cv2.moments(cnt)
#     # print(M['m10'], M['m01'], M['m00'])
#     # cx = int(M['m10'] / M['m00'])
#     # cy = int(M['m01'] / M['m00'])
#     # area = cv2.contourArea(cnt)
#     # draw contours on the original image
#     image_copy = img.copy()
#     cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
#                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#
#     # print(cx, cy, area)
#     # imgDilate = imgDilate[:cx, :cy]
#
#     return image_copy #imgDilate
#
# def checkParkingSpace(imgPro):
#     spaceCounter = 0
#
#     for pos in posList:
#         x, y = pos
#
#         imgCrop = imgPro[y:y + height, x:x + width]
#         # cv2.imshow(str(x * y), imgCrop)
#         count = cv2.countNonZero(imgCrop)
#
#         if count < 900:
#             color = (0, 255, 0)
#             thickness = 5
#             spaceCounter += 1
#         else:
#             color = (0, 0, 255)
#             thickness = 2
#
#         cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
#         cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
#                            thickness=2, offset=0, colorR=color)
#
#     cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
#                        thickness=5, offset=20, colorR=(0, 200, 0))
#
#
# while True:
#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#
#     if hands:
#         # Hand 1
#         hand1 = hands[0]
#         lmList1 = hand1["lmList"]  # List of 21 Landmark points
#         bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
#         centerPoint1 = hand1['center']  # center of the hand cx,cy
#         handType1 = hand1["type"]  # Handtype Left or Right
#
#         fingers1 = detector.fingersUp(hand1)
#
#         if len(hands) == 2:
#             # Hand 2
#             hand2 = hands[1]
#             lmList2 = hand2["lmList"]  # List of 21 Landmark points
#             bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
#             centerPoint2 = hand2['center']  # center of the hand cx,cy
#             handType2 = hand2["type"]  # Hand Type "Left" or "Right"
#
#             fingers2 = detector.fingersUp(hand2)
#
#
#             # Find Distance between two Landmarks. Could be same hand or different hands
#             # length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
#             length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[4][0:2], img)  # with draw
#
#     teste = detect_object(img)
#
#     cv2.imshow("Image", teste)
#     if cv2.waitKey(1) == ord('q'):
#         break
#         cv2.destroyAllWindows()




class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.color = colorR

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

    def resize(self, newH, newW):
        self.size = [newH, newW]

    def check_right_or_wrong(self, imgPro):
        self.spaceCounter = 0

        for pos in imgPro:
            w, h = self.size
            x, y = self.posCenter

            imgCrop = imgPro[y:y + h, x:x + w]
            # cv2.imshow(str(x * y), imgCrop)
            count = cv2.countNonZero(imgCrop)
            number_of_white_pix = np.sum(imgCrop == 255)
            number_of_black_pix = np.sum(imgCrop == 0)
            # print(count, number_of_black_pix, number_of_white_pix)
            # print(self.size, imgCrop.size)

            if count <= 4100:
                # self.color = (255, 0, 0)
                # self.thickness = 0
                # self.spaceCounter += 1
                print("certo")
            else:
                # self.color = (0, 0, 255)
                # self.thickness = 0
                print("errado")
            # cv2.rectangle(pos, self.posCenter, (self.posCenter[0] + w, self.posCenter[1] + h), self.color, self.thickness)
            # cvzone.putTextRect(self, str(count), (x, y + h - 3), scale=1,
            #                    thickness=2, offset=0, colorR=self.color)

        # cvzone.putTextRect(imgPro, f'Free: {spaceCounter}/{len(imgPro)}', (100, 50), scale=3,
        #                    thickness=5, offset=20, colorR=(0, 200, 0))

def detect_object(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16) #cv2.THRESH_BINARY_INV
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    #Sobel test
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(imgGray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(imgGray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    t, teste = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return imgDilate, imgThreshold, teste, grad

# Inicio
cap = cv2.VideoCapture(0)
cap.set(3, 680)
cap.set(4, 680)

detector = HandDetector(detectionCon=0.5)
startDist = None
scale = 0
cx, cy = 150, 150
# colorR = (255, 0, 255)
colorR = (0, 255, 0)


rectList = []
for x in range(1):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, flipType=True)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[4][0:2], img)

        if length < 60:
            cursor = lmList1[8][0:2]  # index finger tip landmark
            # call the update here
            for rect in rectList:
                rect.update(cursor)

    if len(hands) == 2:
        # print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            # print("Zoom Gesture")
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            # point 8 is the tip of the index finger
            if startDist is None:
                # length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[4][0:2], img)
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

                startDist = length

            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

            scale = int((length - startDist) // 2)
            cx, cy = info[4:]

            newH, newW = ((h + scale) // 2) * 2, ((w + scale) // 2) * 2

            if scale != 0:
                for rect in rectList:
                    rect.resize(newH, newW)

    else:
        startDist = None

    # Draw solid
    imgNew = np.zeros_like(img, np.uint8)
    thickness = 5
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, thickness)
        # cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # # Draw Transperency
    # imgNew = np.zeros_like(img, np.uint8)
    # thickness = 5
    # for rect in rectList:
    #     cx, cy = rect.posCenter
    #     w, h = rect.size
    #     cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
    #                   (cx + w // 2, cy + h // 2), colorR, thickness) #cv2.FILLED
    #     cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
    #
    # out = img.copy()
    # alpha = 0.5
    # mask = imgNew.astype(bool)
    # out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    #chek empty or not empty

    img_dilate, img_threshold, otsu, sobel = detect_object(img)
    for rect in rectList:
        rect.check_right_or_wrong(img_dilate)

    teste = cvzone.stackImages([img, img_dilate, otsu], 3, 0.5)

    cv2.imshow("Image", teste)

    if cv2.waitKey(1) == ord('q'):
        break



