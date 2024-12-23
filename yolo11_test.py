from ultralytics import YOLO
import cv2
if __name__ == "__main__":
    # Load a pretrained YOLOv8 model
    model = YOLO("best.pt")

    # Perform inference on an image
    results = model("1.jpg")

    #截取框中的图片
    img=cv2.imread("1.jpg")
    for result in results:
        print(result.boxes.xyxy)
        a=int(result.boxes.xyxy[0][0])
        b=int(result.boxes.xyxy[0][1])
        c=int(result.boxes.xyxy[0][2])
        d=int(result.boxes.xyxy[0][3])
        img=img[b:d,a:c]
        cv2.imwrite("1_2.jpg",img)
        result.show()