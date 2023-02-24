import torch as tc
from pytorch3d.transforms import euler_angles_to_matrix

class BBox():
    def __init__(self, center=None, length=1000, width=1000, height=1000, orientation=None):
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

    def __str__(self) -> str:
        return f"{self.center} {self.height} {self.width}"
