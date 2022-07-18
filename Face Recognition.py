import face_recognition as fr
import cv2
import serial

img = fr.load_image_file('Hossam.jpg')
img_encodings = fr.face_encodings(img)[0]

face_names = ['Hossam']

video = cv2.VideoCapture(0)

face_locations = []
video_face_encodings = []
while True:
    grapped,frame = video.read()
    org = cv2.flip(frame,1)
    frame = cv2.cvtColor(org,cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(frame)
    video_face_encodings = fr.face_encodings(frame,face_locations)

    for x in video_face_encodings:
        matches = fr.compare_faces(img_encodings,video_face_encodings)
        print(matches)
        if matches:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(org, (left, top), (right, bottom), (0, 0, 255), 2)
                center = (left+right)//2,(top+bottom)//2
                print(f"{center[0]}  {center[1]}")
                cv2.circle(org,center,5,(0,255,0),-1)
                cv2.putText(org, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                

    cv2.imshow('video',org)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
