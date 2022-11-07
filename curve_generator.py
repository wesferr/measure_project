import numpy as np
import torch as tc
from curve_utils import CurveUtils
import pandas as pd

from pytorch3d.transforms import euler_angles_to_matrix

class BBox():
    def __init__(self, center=None, length=1000, width=1000, height=1000, orientation=None, angles=None):
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
        dir  = points - self.center
        tests = [
            tc.sum(dir * self.orientation[0], axis=1) <= (self.width/2),
            tc.sum(dir * self.orientation[0], axis=1) >= (-self.width/2),

            tc.sum(dir * self.orientation[1], axis=1) <= (self.height/2),
            tc.sum(dir * self.orientation[1], axis=1) >= (-self.height/2),

            tc.sum(dir * self.orientation[2], axis=1) <= (self.length/2),
            tc.sum(dir * self.orientation[2], axis=1) >= (-self.length/2),
        ]
        return tc.all(tc.column_stack(tests), dim=1)
        

class CurveGenerator():

    def __init__(self, mesh_path=None, template_path=None, vertices=None, faces=None, body_measures=None) -> None:

        if body_measures is None:
            raise Exception("Measures cannot be None")
        else:
            self.body_measures = body_measures

        if mesh_path:
            self.vertices = CurveUtils.load_mesh(mesh_path)
        if len(vertices)>0:
            self.vertices = vertices
        self.vertices = tc.cuda.FloatTensor(self.vertices)

        if template_path:
            self.faces, _ = CurveUtils.load_template(template_path)
        if len(faces)>0:
            self.faces = faces
        self.faces = tc.cuda.LongTensor(self.faces)

        self.curves = {}
        self.measures = {}
        self.positions = {}

    def generate_positions(vertices, coordinates):
        x = vertices[coordinates[:,0].to(tc.long)].T * coordinates[:,3]
        y = vertices[coordinates[:,1].to(tc.long)].T * coordinates[:,4]
        z = vertices[coordinates[:,2].to(tc.long)].T * coordinates[:,5]
        p = x+y+z
        return p.T

    def calculate_distances(positions):
        if positions.numel() == 0:
            return 0
        distances = tc.cdist(positions, positions)
        distance = distances.diagonal(1).sum()
        return distance*100

    def baricentric(self, a, b, c, p):
        
        vab = b - a
        vbc = c - b
        vca = a - c
        
        vap = p - a
        vbp = p - b
        vcp = p - c


        cross_norm = tc.linalg.norm(tc.cross(vab,vbc), axis=1)
        n = tc.cross(vab,vbc) / tc.column_stack([cross_norm, cross_norm, cross_norm])
        
        ABC = (n * tc.cross(vab, vbc)).sum(axis=1) / 2
        ABP = (n * tc.cross(vab, vbp)).sum(axis=1) / 2
        BCP = (n * tc.cross(vbc, vcp)).sum(axis=1) / 2
        CAP = (n * tc.cross(vca, vap)).sum(axis=1) / 2

        w = ABP/ABC
        u = BCP/ABC
        v = CAP/ABC

        return tc.column_stack([u,v,w])

    def intersection(self, p0, p1, p2, f1,f2,f3):

        vd = p0 - p1
        integer_part = tc.inner(p0, self.nplane)
        incognita_part = tc.inner(self.nplane, vd)
        t = -(self.dplane + integer_part)/incognita_part
        p = p0 + (tc.column_stack([t,t,t])*vd)
        return tc.column_stack([tc.column_stack([f1,f2,f3]), self.baricentric(p0,p1,p2,p)])


    def calculate_colision(self):


        v = self.vertices[self.faces]
        v = v.reshape(-1, 3)
        colision = (v@self.nplane) + self.dplane
        colision = colision.reshape(-1, 3)
        v = v.reshape(-1,3,3)

        t1 = tc.all(colision < 0, axis=1)
        t2 = tc.all(colision > 0, axis=1)
        t3 = ~(tc.any(tc.column_stack((t1,t2)), axis=1))

        f = self.faces[t3]
        colision = colision[t3]
        v = v[t3]
        
        t1 = colision[:,1] * colision[:,2] > 0
        t2 = colision[:,0] * colision[:,2] > 0
        t3 = colision[:,0] * colision[:,1] > 0

        p = tc.row_stack([
            self.intersection(v[t1][:,0], v[t1][:,1], v[t1][:,2], f[t1][:,0], f[t1][:,1], f[t1][:,2]),
            self.intersection(v[t1][:,0], v[t1][:,2], v[t1][:,1], f[t1][:,0], f[t1][:,2], f[t1][:,1]),

            self.intersection(v[t2][:,1], v[t2][:,2], v[t2][:,0], f[t2][:,1], f[t2][:,2], f[t2][:,0]),
            self.intersection(v[t2][:,1], v[t2][:,0], v[t2][:,2], f[t2][:,1], f[t2][:,0], f[t2][:,2]),

            self.intersection(v[t3][:,2], v[t3][:,0], v[t3][:,1], f[t3][:,2], f[t3][:,0], f[t3][:,1]),
            self.intersection(v[t3][:,2], v[t3][:,1], v[t3][:,0], f[t3][:,2], f[t3][:,1], f[t3][:,0])
        ])

        return p

    def remove_mirror(self, positions, coordinates):
        absolute_positions = abs(positions)[:,0]
        min_point = absolute_positions.min(axis=0)[0]
        if min_point > 0.003:
            invalid_positions = positions[:,0] < 0
            coordinates = coordinates[invalid_positions]
        return coordinates


    def sort_curve(self, positions, measure):
        centroid = positions.mean(axis=0)
        vectors = positions - centroid
        sort_around = vectors.var(axis=0).argmin()
        if sort_around == 0:
            angles = tc.arctan2(vectors[:,1], vectors[:,2])
        if sort_around == 1:
            angles = tc.arctan2(vectors[:,0], vectors[:,2])
        if sort_around == 2:
            angles = tc.arctan2(vectors[:,0], vectors[:,1])
        return measure[tc.argsort(angles)]

    def calculate_curves(self, axis_number, plane_normal, axis_name, bbox, exbox):
        min_y = self.vertices[:,axis_number].min()
        max_y = self.vertices[:,axis_number].max()
        coordinates_array = []
        for i in tc.arange(min_y, max_y, 1e-3):
            self.nplane, self.dplane = tc.cuda.FloatTensor(plane_normal), -i
            coordinates = self.calculate_colision()
            if coordinates.size(0):
                positions = CurveGenerator.generate_positions(self.vertices, coordinates)
                if coordinates.size(0) > 0:
                    filter = bbox.check_inside(positions)
                    for box in exbox:
                        f = ~box.check_inside(positions)
                        filter = tc.all(tc.column_stack([f, filter]), dim=1)
                    positions = positions[filter]
                    coordinates = coordinates[filter]
                    if coordinates.size(0) > 0:
                        coordinates = self.sort_curve(positions, coordinates)
                        coordinates_array.append(coordinates)
        self.curves[axis_name] = coordinates_array
        return coordinates_array

    def partial_computate(self, axes, bounding_boxes, index, bbignore):
        curves = self.calculate_curves(*axes[index], bounding_boxes[index], bbignore)
        all_positions = []
        for curve in curves:
                position = CurveGenerator.generate_positions(self.vertices, curve)
                all_positions.extend(position)
        CurveUtils.save_obj('teste.obj', all_positions)

    def computate(self, axes, bounding_boxes):
        curves_bust = self.calculate_curves(*axes[0], bounding_boxes[0],[bounding_boxes[3]])
        curves_torso = self.calculate_curves(*axes[1], bounding_boxes[1],[])
        curves_leg = self.calculate_curves(*axes[2], bounding_boxes[2],[])
        curves_arm = self.calculate_curves(*axes[3], bounding_boxes[3], bounding_boxes[:3])
        curves_neck = self.calculate_curves(*axes[4], bounding_boxes[4], [])

        curves_blocks = []
        curves_blocks.append(curves_bust)
        curves_blocks.append(curves_torso)
        curves_blocks.append(curves_leg)
        curves_blocks.append(curves_arm)
        curves_blocks.append(curves_neck)

        all_positions = []
        all_measures = []
        for curves in curves_blocks:
            curves_positions = []
            curves_measures = []
            for curve in curves:
                position = CurveGenerator.generate_positions(self.vertices, curve)[::2]
                measures = CurveGenerator.calculate_distances(position)
                curves_positions.append(position)
                curves_measures.append(measures)
            all_positions.append(curves_positions)
            all_measures.append(curves_measures)

        return curves_blocks, all_measures, all_positions


def get_all_curves(selected_subjects, selected_measures, template, device):
    tcenter = lambda a: tc.FloatTensor(a).to('cuda')
    results = dict()
    for gender in ['female']:
        
        print(f"PROCESSING {gender.upper()}")
        bodies = tc.load(f'data/{gender}_bodies_t.pt')
        poses = np.load(f'data/{gender}_poses.npy')
        body = bodies[selected_subjects[gender]]

        CurveUtils.save_obj('body.obj', body, template+1)

        body_measures = selected_measures[gender]/1000
        body_portion = body_measures['height']/8

        body_min = body[:,1].min()
        body_width = body[:,0].max() - body[:,0].min()
        
        # bust planes box
        height = body_portion
        center = (0, body_min + (body_portion*5.1) + (height/2), 0)
        width = body_measures['bust_chest_girth']/tc.pi*1.3
        bust_box = BBox(width=width, height=height, center=tcenter(center))
        
        # torso planes box
        width = body_measures['hip_girth']/tc.pi*1.3
        height = body_portion*1.5
        center = (0, body_min + (body_portion*3.75) + (height/2), 0)
        torso_box = BBox(width=width, height=height, center=tcenter(center))
        
        # leg planes box
        width = body_width/2 #body_measures['thigh_girth']/tc.pi*1.2
        height = body_portion*3
        to_right = body_width/4# body_measures['hip_girth']/tc.pi/3
        center = (-to_right, body_min + body_portion + (height/2) ,0)
        leg_box = BBox(width=width, height=height, center=tcenter(center))
        
        # arm planes box
        width = body_portion*4
        height = body_width
        c = body_measures['bust_chest_girth']/tc.pi/2*1.1
        center = (-c-(width/2), body_portion*5.5,0)
        arm_box = BBox(width=width, height=height, center=tcenter(center))
        
        # neck planes box
        height = body_portion
        width = body_portion
        center = (0, body_min + (body_portion*6.5) + (height/2), 0)
        neck_box = BBox(width=width, height=height, center=tcenter(center))
        neck_rotation_matrix = euler_angles_to_matrix(tc.tensor([-17.5*(tc.pi/180),0,0]).to('cuda'), "XYZ")
        neck_plane_normal = tc.FloatTensor([0.0,1.0,0.0]).to(device) @ neck_rotation_matrix
        
        axes = [
            (1,[0.0,1.0,0.0], 'b'),
            (1,[0.0,1.0,0.0], 't'),
            (1,[0.0,1.0,0.0], 'l'),
            (0,[1.0,0.0,0.0], 'a'),
            (1,neck_plane_normal, 'n')
        ]


        bounding_boxes = [
            bust_box,
            torso_box,
            leg_box,
            arm_box,
            neck_box,
        ]
        
        generator = CurveGenerator(vertices=body, faces=template, body_measures=selected_measures[gender])
        result = generator.computate(axes, bounding_boxes)
        results[gender] = result
    return results

def select_better(selected_subjects, selected_measures, device):
    curve_index = {
        'neck_girth':4, # 5.3.2
        'bust_chest_girth': 0, # 5.3.4
        'waist_girth': 1, # 5.3.10
        'hip_girth': 1, # 5.3.13
        'upper_arm_girth': 3, # 5.3.16
        'thigh_girth': 2, # 5.3.20
    }

    for gender in ['female']:
        bodies = tc.load(f'./data/{gender}_bodies_t.pt')
        body = bodies[selected_subjects[gender]]
        body_min = body[:,1].min()
        body_max = body[:,1].max()

        result = tc.load(f"./data/{gender}_result.zip")
        curves = result[0]
        measures = result[1]
        positions = result[2]

        all_positions = []
        all_measures = []
        best_curves = []
        for measure in curve_index.keys():
            measures_index = tc.FloatTensor(measures[curve_index[measure]]).to(device)
            diff = abs(measures_index - selected_measures[gender][measure]/10)
            best = diff.argmin()
            all_positions.append(positions[curve_index[measure]][best])
            all_measures.append(measures_index[best].cpu().numpy())
            best_curves.append(curves[curve_index[measure]][best])

        height = body_max - body_min
        all_measures.append(height.cpu().numpy()*100)

        # waist_height = all_positions[2].mean(0)[1] - body_min
        # all_measures.append(waist_height.cpu().numpy()*100)

        # bust_height = all_positions[1].mean(0)[1] - body_min
        # all_measures.append(bust_height.cpu().numpy()*100)
            

        data = {
            'original': selected_measures[gender][list(curve_index.keys())+['height']]/10,
            'measured': all_measures,
            'error(mm)': abs(selected_measures[gender][list(curve_index.keys())+['height']]/10 - all_measures)*10
        }
        data = pd.DataFrame(data)
        print(f"\n{gender.upper()}")
        print(data)
        
        tc.save(best_curves, f'data/{gender}_best_curves.zip')