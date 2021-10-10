import numpy as np
import statistics
from statistics import mode
import pickle
import os
import cv2
import imutils
def get_name():
    curr_path = os.getcwd()

    print('Loading Face detector')

    proto_path = os.path.join(curr_path , 'model','deploy.prototxt')
    model_path = os.path.join(curr_path, 'model','res10_300x300_ssd_iter_140000.caffemodel')
    face_detector = cv2.dnn.readNetFromCaffe(prototxt = proto_path, caffeModel = model_path)

    print('Loading face recognition Model')
    recognition_model = os.path.join(curr_path,'model','openface.nn4.small2.v1.t7')
    face_recognizer = cv2.dnn.readNetFromTorch(model= recognition_model)

    recognizer = pickle.loads(open('recognizer.pickle','rb').read())
    le = pickle.loads(open('le.pickle','rb').read())

    print('starting Video')

    vs = cv2.VideoCapture(0)
    #time.sleep(1)

    name_list = []


    while True:
        
        ret , frame = vs.read()
        frame = imutils.resize(frame, width = 600)

        (h,w) = frame.shape[:2]

        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),1.0,(300,300),(104.0,177.0,123.0),False,False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0,0,i,2]
            
            if confidence >= 0.5:
                box = face_detections[0,0,i,3:7] *np.array([w,h,w,h])
                (startX, startY,endX,endY) = box.astype('int')

                face = frame[startY:endY, startX:endX]

                (fh,fw) = face.shape[:2]

                face_blob = cv2.dnn.blobFromImage(face,1.0/255, (96,96),(0,0),True,False)

                face_recognizer.setInput(face_blob)
                vec = face_recognizer.forward()

                preds = recognizer.predict_proba(vec)[0]
                j= np.argmax(preds)
                proba = preds[j]
                if proba >= 0.5:
                    name = le.classes_[j]
                else:
                    name = 'Unknown'

                name_list.append(name)

                text = '{}: {:.2f}'.format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 100
                cv2.rectangle(frame,(startX,startY),(endX, endY),(0,0,255),2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_PLAIN, 0.65, (0,0,255),2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    
    cv2.destroyAllWindows()
    print(mode(name_list))
    return mode(name_list)

if __name__ == '__main__':
    get_name()

    