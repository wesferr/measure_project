import torch as tc
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix
from src.mesh_manipulation import save_obj

class BBox():
    def __init__(self, center=None, length=1, width=1, height=1, orientation=None):
        if center is None:
            self.center = tc.FloatTensor([0.0,0.0,0.0]).to('cuda')
        else:
            self.center = center

        if orientation is None:
            self.orientation = tc.FloatTensor([
                [1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0],
            ]).to('cuda')
        else:
            self.orientation = orientation

        self.length = length
        self.width = width
        self.height = height

    def rotate(self, angles=(0,0,0)):
        rotation_matrix = euler_angles_to_matrix(tc.tensor(angles).to('cuda')/180*tc.pi, "XYZ")
        self.orientation = self.orientation @ rotation_matrix

    def check_inside(self, points):
        direction  = points - self.center
        tests = [
            tc.sum(direction * self.orientation[0], axis=1) <= (self.width/2),
            tc.sum(direction * self.orientation[0], axis=1) >= (-self.width/2),

            tc.sum(direction * self.orientation[1], axis=1) <= (self.height/2),
            tc.sum(direction * self.orientation[1], axis=1) >= (-self.height/2),

            tc.sum(direction * self.orientation[2], axis=1) <= (self.length/2),
            tc.sum(direction * self.orientation[2], axis=1) >= (-self.length/2),
        ]
        return tc.all(tc.column_stack(tests), dim=1)
    
    def save_limits(self, path):
        rigth = self.center[0] + self.width/2
        left = self.center[0] - self.width/2
        up = self.center[1] + self.height/2
        down = self.center[1] - self.height/2
        front = self.center[2] + self.length/2
        back = self.center[2] - self.length/2
        pontos = [
            (rigth, up, front),
            (left, up, front),
            (rigth, down, front),
            (left, down, front),
            (rigth, up, back),
            (left, up, back),
            (rigth, down, back),
            (left, down, back),
        ]
        faces = np.array([
            [0,1,2], [1,2,3],
            [4,5,6], [5,6,7],
            [1,3,5], [5,3,7],
            [0,4,2], [2,4,6],
            [0,1,5], [0,5,4],
            [2,3,7], [2,7,6]
        
        ])+1
        save_obj(path=path, pontos=pontos, faces=faces)

    def __str__(self) -> str:
        return f"{self.center} {self.height} {self.width}"
