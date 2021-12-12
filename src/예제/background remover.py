import cv2

img = cv2.imread("data/GRU.png", cv2.IMREAD_COLOR)

for i, x in enumerate(img):
    for j, y in enumerate(x):
        if list(y) == [255, 255, 255]:
            img[i][j] = [255, 237, 243]

cv2.imshow("Adfa", img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("GRU.PNG", img)