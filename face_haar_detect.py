import cv2

# 加载Haar级联分类器

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 读取图像
image = cv2.imread('2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 在图像上标记人脸,只保留最大的人脸
if len(faces) > 0:
    x, y, w, h = faces[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示结果
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('face_detection2.jpg', image)
