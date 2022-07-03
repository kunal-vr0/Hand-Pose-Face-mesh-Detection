#========================================================================================================================
import cv2
import mediapipe as mp
width = 1080
height = 720
#========================================================================================================================
class HandDet():
    #========================================================================================================================
    def __init__(self, maxHand = 2, mode = False, modelComplexity = 1 , tol1 = 0.5, tol2 = 0.5):
        self.mode = mode
        self.maxHand = maxHand 
        self.modeComplexity = modelComplexity 
        self.tol1 = tol1 
        self.tol2 = tol2 
        self.mpHand = mp.solutions.hands
        self.myHand = self.mpHand.Hands(self.mode, self.maxHand, self.modeComplexity, self.tol1, self.tol2)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
    #========================================================================================================================
    #============================================function for hand detection===================================================================
    def findHands(self, frame, draw = True):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.myHand.process(RGBframe)
        bothHands = []
        labels = []
        if results.multi_hand_landmarks:
            #=======================================retriving labels=============================================================
            for abd in results.multi_handedness:
                for bcd in abd.classification:
                    labels.append(bcd.label)
            #====================================================================================================================
            #========================================retriving land and drawing connections================================================================
            for handLMs in results.multi_hand_landmarks:
                oneHand = []
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLMs, self.mpHand.HAND_CONNECTIONS, self.mpStyle.get_default_hand_landmarks_style(), self.mpStyle.get_default_hand_connections_style() )
                for LM in handLMs.landmark:
                    oneHand.append((int(LM.x * width), int(LM.y * height)))
                bothHands.append(oneHand)
            #========================================================================================================================
        return bothHands, labels
    #========================================================================================================================
#========================================================================================================================
class PoseDet():
    #========================================================================================================================
    def __init__(self, segmentaion = False, still = False, modelComplexity = 1, smoothData = True, smoothSeg = True, tol1 = 0.5, tol2 = 0.5):
        self.still = still 
        self.modelComplexity = modelComplexity
        self.smoothData = smoothData
        self.segmentation = segmentaion
        self.smoothSeg = smoothSeg
        self.tol1 = tol1
        self.tol2 = tol2 
        self.mpPose = mp.solutions.pose
        self.myPose = self.mpPose.Pose(self.still, self.modelComplexity, self.smoothData, self.segmentation, self.smoothSeg, self.tol1, self.tol2)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
    #========================================================================================================================
    #========================================================================================================================
    def findPose(self, frame, draw = True):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.myPose.process(RGBframe)
        poseLandmarks = []
        #=================================================retriving landmarks and drawing connections========================================================
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for LM in results.pose_landmarks.landmark:
                poseLandmarks.append((int(width*LM.x),int(height*LM.y)))
        #========================================================================================================================
        if self.segmentation:
            return poseLandmarks, results.segmentation_mask

        return poseLandmarks
    #========================================================================================================================
#========================================================================================================================
#========================================================================================================================
class FaceMeshDet():
    #========================================================================================================================
    def __init__(self, still = False, maxFace = 1, refineLM = True, tol1 = 0.5, tol2 = 0.5):
        self.still = still 
        self.maxFace = maxFace 
        self.refineLM = refineLM
        self.tol1 = tol1 
        self.tol2 = tol2
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.mpStyle = mp.solutions.drawing_styles
        self.myFaceMesh = self.mpFaceMesh.FaceMesh(self.still, self.maxFace, self.refineLM, self.tol1, self.tol2)
    #========================================================================================================================
    #========================================================================================================================
    def findFace(self, frame, draw = True, ifcolor = False, color = (0,0,125)):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.myFaceMesh.process(RGBframe)
        circleSpec = self.mpDraw.DrawingSpec(thickness = 2, circle_radius = 2, color = color)
        faceLMs = []
        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:
                if draw:
                    if ifcolor:
                        self.mpDraw.draw_landmarks(frame, face, self.mpFaceMesh.FACEMESH_TESSELATION, circleSpec, self.mpStyle.get_default_face_mesh_tesselation_style())
                        #self.mpDraw.draw_landmarks(frame, face, self.mpFaceMesh.FACEMESH_CONTOURS, circleSpec, self.mpStyle.get_default_face_mesh_contours_style())
                        #self.mpDraw.draw_landmarks(frame, face, self.mpFaceMesh.FACEMESH_IRISES, circleSpec, self.mpStyle.get_default_face_mesh_iris_connections_style())

                    else:
                        self.mpDraw.draw_landmarks(frame, face, self.mpFaceMesh.FACEMESH_TESSELATION, None, self.mpStyle.get_default_face_mesh_tesselation_style())
                        #self.mpDraw.draw_landmarks(frame, face, self.mpFaceMesh.FACEMESH_CONTOURS, None, self.mpStyle.get_default_face_mesh_contours_style())
                        #self.mpDraw.draw_landmarks(frame, face, self.mpFaceMesh.FACEMESH_IRISES, None, self.mpStyle.get_default_face_mesh_iris_connections_style())

                for lm in face.landmark:
                    faceLMs.append((int(width*lm.x), int(height*lm.y)))
        
        return faceLMs
    #========================================================================================================================
#========================================================================================================================


BodyPose = PoseDet()
HandPose = HandDet()
FacePose = FaceMeshDet()
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

radius = 10
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ig, frame = cam.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)
    #Hands, labels = HandPose.findHands(frame)
    #bodyLM = BodyPose.findPose(frame)
    faceLM = FacePose.findFace(frame)
    #for hand, label in zip(Hands, labels):
    #    for lm in hand:
    #        cv2.circle(frame, lm, radius, (122, 80, 20), -1)
    #    cv2.putText(frame, label, hand[8], font, 2, (8,156,255), 2)
    #for LM in bodyLM:
    #    cv2.circle(frame, LM, radius, (23,123,24), -1)
    i = 0
    for lm in faceLM:
        cv2.putText(frame, str(i), lm, font, 0.2, (0,0,255), 1)
        i += 1
    
    cv2.imshow('Body Hand', frame)
    cv2.moveWindow('Body Hand', 0,0)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()



        
