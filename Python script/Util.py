import bleach
import cv2
from cv2 import Mat
import numpy as np
import mediapipe as mp
from math import atan, pi
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

class Util:
    
    neutral_angle = None
    
    def __init__(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=True,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
    def draw_coordinates(self, image, points, color = [150, 0, 200]):
        neighbor_vector = [(0,0), (1,0), (0,1), (-1,0), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for pt in points:
            for i in neighbor_vector:
                image[pt[1] + i[1] ][pt[0] + i[0]] = color
        return image
    
    def get_rectangle_coordinates(self, coordinates: list, angle: int) -> tuple:
        angle -= self.neutral_angle
        if angle == 0:
            return coordinates[0], coordinates[3]
        if angle < 0:
            return (coordinates[0][0], coordinates[2][1]), (coordinates[3][0], coordinates[1][1])
        else:
            return (coordinates[1][0], coordinates[0][1]), (coordinates[2][0], coordinates[3][1]) 
        
    def remove_whitespace(self, image: Mat, blend: np.ndarray, x: int, y: int, threshold: int=225) -> None:
        for i in range(blend.shape[0]):
            for j in range(blend.shape[1]):
                for k in range(3):
                    if blend[i][j][k] > threshold:
                        blend[i][j][k] = image[i + y][j + x][k]

    def get_angle(self, coordinates: list) -> float:
        height = (coordinates[1][1] - coordinates[2][1])
        base = (coordinates[1][0] - coordinates[2][0])
        
        angle = atan(height/base) * 180/pi
        if self.neutral_angle == None:
            self.neutral_angle = angle
            print("Neutral angle => {}".format(self.neutral_angle))
        return angle
    
    def get_rotated_image(self, im: Mat, angle: float) -> Mat:
        mask = (im == [0,0,0,0]).all(axis=2)
        im[mask] = [255, 255, 255, 0]
        
        imHeight, imWidth = im.shape[0], im.shape[1]
        centreX, centreY = imWidth//2, imHeight//2
            
        rotationMat = cv2.getRotationMatrix2D(
            center=(centreX, centreY),
            angle=angle - self.neutral_angle,
            scale=1
        )
        
        cos = np.abs(rotationMat[0][0])
        sin = np.abs(rotationMat[1][0])
        
        newWidth = int((imHeight * sin) + (imWidth * cos))
        newHeight = int((imHeight * cos) + (imWidth * sin))
        
        rotationMat[0][2] += newWidth/2 - centreX
        rotationMat[1][2] += newHeight/2 - centreY
        
        dst_mat = np.full((newHeight, newWidth, 4), [255, 255, 255, 0])
        rotatedMat = cv2.warpAffine(
            im,
            rotationMat,
            (newWidth, newHeight),
            np.ascontiguousarray(dst_mat, np.uint8),
            borderMode=cv2.BORDER_TRANSPARENT
        )
        return rotatedMat[:, ::-1]
    
    def get_landmarks(self, image: Mat) -> tuple:
        h,w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_mesh_result = self.face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if face_mesh_result.multi_face_landmarks:
            landmarksList: NormalizedLandmarkList = face_mesh_result.multi_face_landmarks[0]
            landmarks = []
            for i, landmark in enumerate(landmarksList.landmark):
                if i in [71, 123, 301, 352]:
                    landmarks.append((int(landmark.x * w), int(landmark.y * h)))
        return landmarks, self.get_angle(landmarks)
    
    def add_effect(self, image: Mat, effect_path) -> None:
        cords, angle = self.get_landmarks(image)
        
        rotated_image: Mat = self.get_rotated_image(
        cv2.imread(
            effect_path,
            cv2.IMREAD_UNCHANGED
           ),
            angle
        )
        
        top_left, bottom_right = self.get_rectangle_coordinates(cords, angle)

        roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        w, h = roi.shape[:2]
        effect = cv2.resize(rotated_image, (h, w))
        blend = cv2.addWeighted(roi, 0, effect, 1.0, 0)
        
        self.remove_whitespace(image, blend, top_left[0], top_left[1])
        image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] = blend
        
        
if __name__ == '__main__':
    util = Util()
    image = cv2.imread(r'images/img.jpg', -1)
    print(image.shape)
    face_landmark_cords, angle = util.get_landmarks(image)
    effect = cv2.imread(r'effects/spec2.png', -1)
    effect = util.get_rotated_image(effect, angle)