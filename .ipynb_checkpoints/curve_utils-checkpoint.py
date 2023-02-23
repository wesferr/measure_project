import numpy as np
import torch as tc


class CurveUtils():

    def load_template(file_path):
        faces = []
        normals = []
        i = 0
        with open(file_path) as file:
            for line in file:
                if line[0] == "f":
                    line_split = line[1:].split()


                    line_split = np.array([i.split("//") for i in line_split])
                    face_temp = list(map(int, line_split[:, 0]))
                    faces.append(face_temp)

                    if line_split[0].size > 1 and line_split[0][1] != '':
                        normal_temp = list(map(int, line_split[:, 1]))
                        normals.append(normal_temp)

                    i += 1
        return tc.LongTensor(faces)-1, tc.FloatTensor(normals)
    def load_mesh(file_path):
        vertex = []
        with open(file_path, "r") as f:
            for line in f:
                if line[0] == 'v':
                    line.replace('\n', ' ')
                    tmp = list(map(float, line[1:].split()))
                    vertex.append(tmp)
                else:
                    continue
        return tc.FloatTensor(vertex)
        
    def save_obj(path, pontos, faces=[]):
        with open(path, "w") as file:
            for ponto in pontos:
                file.write("v {} {} {}\n".format(ponto[0], ponto[1], ponto[2]))
            for face in faces:
                file.write("f {} {} {}\n".format(face[0], face[1], face[2]))

    def calculate_distances(positions, closed=True):
        if positions.numel() == 0:
            return 0
        distances = tc.cdist(positions, positions)
        if closed==True:
            distance = distances.diagonal(1).sum()
        else:
            bigger = distances.diagonal(1).max()
            distance = distances.diagonal(1).sum() - bigger
        return distance*100

    def calculate_height(floor, coordiantes, body):
        positions = CurveUtils.generate_positions(coordiantes, body)
        height = abs(floor - positions.mean(axis=0)[1])
        return height * 100

    def calculate_angle(vectors, axis):
        inner_product = (vectors * axis.unsqueeze(0)).sum(dim=1)
        a_norm = tc.linalg.norm(vectors,axis=1)
        b_norm = tc.linalg.norm(axis)
        cos = inner_product / (2 * a_norm * b_norm)
        angle = tc.acos(cos)
        return angle

    def plane_triangle_colision(planes, triangles, device, faces=None):
        centroid = tc.row_stack([triangles[:,0], triangles[:,1], triangles[:,2], triangles[:,0], triangles[:,1], triangles[:,2]])
        vectors = tc.row_stack([
            triangles[:,1] - triangles[:,0],
            triangles[:,0] - triangles[:,1],
            triangles[:,0] - triangles[:,2],
            triangles[:,2] - triangles[:,0],
            triangles[:,2] - triangles[:,1],
            triangles[:,1] - triangles[:,2],
        ])
        n,d,p0,p1,p2 = CurveUtils.define_planes(planes)
        P, _, _ = CurveUtils.calculate_colision_1(centroid, vectors, n, d)
        baricentric, triangles_filter = CurveUtils.baricentric_coordinates(triangles[:,0].repeat(6,1), triangles[:,1].repeat(6,1), triangles[:,2].repeat(6,1), P+centroid)
        _, planes_filter = CurveUtils.baricentric_coordinates(p0, p1, p2, P+centroid)
        filter = triangles_filter&planes_filter
        coordinates = tc.cat([faces.repeat(6,1).unsqueeze(0), baricentric], dim=-1)
        positions, coordinates = (P+centroid)[filter], coordinates[filter]

        positions = positions.round(decimals=6)
        positions, index, counts = positions.unique(dim=0, return_inverse=True, return_counts=True)

        # get index of first recurrence of values in a array
        _, ind_sort = index.sort(stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = tc.cat((tc.LongTensor([0]).to(device), cum_sum[:-1]))
        index = ind_sort[cum_sum]
        coordinates = coordinates[index]
        return positions, coordinates
        
    def ray_polygon_cosilion(polygons, positions):

        position = []
        centroid, nvectors = CurveUtils.define_vectors(positions)
        n, d, p0, p1, p2 = CurveUtils.define_planes(polygons)
        P, pfilter, t = CurveUtils.calculate_colision(centroid, nvectors, n, d)
        baricentric, bfilter = CurveUtils.baricentric_coordinates(p0, p1, p2, P+centroid)
        result = CurveUtils.get_closest_point(P+centroid, t, pfilter, bfilter)
        return result, CurveUtils.calculate_distances(result), position

    def define_vectors(positions):
        # project points
        ones = tc.ones(positions.shape[0]).to("cuda")
        lower_dimension = abs(positions).std(axis=0).argmin()
        A = positions.clone()
        A[:,lower_dimension] = ones
        # A = tc.column_stack([positions[:,0], ones, positions[:,2]])
        new_y = A @ tc.linalg.solve(A.T@A, A.T@positions[:,lower_dimension])
        A[:,lower_dimension] = new_y


        # calculate centroid and vectors
        centroid = A.mean(axis=0)
        vectors = A - centroid
        vectors = vectors/tc.linalg.norm(vectors, axis=1).unsqueeze(1)
        return centroid, vectors

    def define_planes(polygons):
        p0 = polygons[:,0]
        p1 = polygons[:,1]
        p2 = polygons[:,2]
        cross = tc.cross(p0 - p1, p0 - p2, dim=1)
        n = cross / tc.linalg.norm(cross, axis=1).unsqueeze(1)
        d = -tc.sum(p0*n, axis=1)
        return n, d, p0, p1, p2

    def calculate_colision_1(origin, vectors, normals, distances):
        nd = (normals@vectors.T)
        pn = (normals@origin.T)
        t = -((pn+distances.unsqueeze(1))/nd)
        P = vectors.unsqueeze(0) * t.unsqueeze(2)
        return P, (t >= 0), t

    def calculate_colision(origin, vectors, normals, distances):
        nd = (vectors@normals.T)
        pn = (origin@normals.T)
        t = -((pn+distances.unsqueeze(0))/nd)
        P = t.unsqueeze(2) @ vectors.unsqueeze(1)
        return P, (t >= 0), t

    def baricentric_coordinates(a, b, c, p):
        
        vab = (b - a).unsqueeze(0)
        vac = (c - a).unsqueeze(0)
        vap = p - a

        dbb = (vab * vab).sum(axis=2)
        dbc = (vab * vac).sum(axis=2)
        dcc = (vac * vac).sum(axis=2)
        dpb = (vap * vab).sum(axis=2)
        dpc = (vap * vac).sum(axis=2)

        denom = dbb * dcc - dbc * dbc
        v = (dcc * dpb - dbc * dpc) /denom
        w = (dbb * dpc - dbc * dpb) /denom
        u = 1.0 - v - w

        test = (u>=0) & (v>=0) & (w>=0)
        return tc.stack([u,v,w], axis=2), test


    def get_closest_point(P, t, pfilter, bfilter):
        indices = tc.where(pfilter & bfilter)
        matrix = tc.sparse_coo_tensor(tc.row_stack(indices), t[indices]-100).to_dense()
        indices = indices[0].unique()
        best = matrix.min(axis=1)[1]
        selected_indices = [indices, best[indices]]
        return P[selected_indices]

    def generate_positions(coordinates, vertices):
        indices = coordinates[:,:3].type(tc.LongTensor)
        parameters = coordinates[:,3:6]
        positions = vertices[indices] * parameters.unsqueeze(2)
        return positions.sum(axis=1)

    def sort_curve(positions, coordinates):
        centroid = positions.mean(axis=0)
        vectors = positions - centroid
        sort_around = vectors.var(axis=0).argmin()
        if sort_around == 0:
            angles = tc.arctan2(vectors[:,1], vectors[:,2])
        if sort_around == 1:
            angles = tc.arctan2(vectors[:,0], vectors[:,2])
        if sort_around == 2:
            angles = tc.arctan2(vectors[:,0], vectors[:,1])
        return positions[tc.argsort(angles)], coordinates[tc.argsort(angles)]

    def calculate_curve(body, bounding_box, triangle, axis, template, device, closed=True):

        if axis==1:
            bmin = bounding_box.center[1]-(bounding_box.height/2)
            bmax = bounding_box.center[1]+(bounding_box.height/2)
        if axis==0:
            bmin = bounding_box.center[0]-(bounding_box.width/2)
            bmax = bounding_box.center[0]+(bounding_box.width/2)

        height_values = tc.arange(bmin, bmax, 0.001)
        all_positions = []
        all_coordinates = []
        all_measures = []

        for value in height_values:

            triangles = triangle(value).unsqueeze(0)
            positions, coordinates = CurveUtils.plane_triangle_colision(triangles, body[template], device, template)

            filter = bounding_box.check_inside(positions)
            positions, coordinates = positions[filter], coordinates[filter]

            positions, coordinates = CurveUtils.sort_curve(positions, coordinates)
            measures = CurveUtils.calculate_distances(positions, closed)
        

            all_positions.append(positions)
            all_coordinates.append(coordinates)
            all_measures.append(measures)
        return all_coordinates, all_measures, all_positions

    def get_curves(selected_body, selected_measure, template, device):

        tcenter = lambda a: tc.FloatTensor(a).to(device)
        from curve_generator import BBox

        body_measures = selected_measure/1000
        body_portion = body_measures['stature']/8
        body_min = selected_body[:,1].min()
        body_width = selected_body[:,0].max() - selected_body[:,0].min()

        # neck planes box
        from pytorch3d.transforms import euler_angles_to_matrix
        height = body_portion
        width = (body_portion+1)
        center = (0, body_min + (body_portion*6.5) + (height/2), 0)
        neck_box = BBox(width=width, height=height, center=tcenter(center))

        neck_all_cordinates = []
        neck_all_positions = []
        neck_all_measures = []
        for i in np.arange(0,23, 0.5):
            neck_rotation_matrix = euler_angles_to_matrix(tc.tensor([float(-24)*(tc.pi/180),0,0]).to(device), "XYZ")
            neck_triangle = lambda   x: tc.FloatTensor([
                [20, x, 11.6],
                [-20, x, 11.6],
                [0, x, -23.2]
            ]).to(device) @ neck_rotation_matrix
            neck_curves = CurveUtils.calculate_curve(selected_body, neck_box, neck_triangle, 0, template, device)
            neck_all_cordinates.extend(neck_curves[0])
            neck_all_positions.extend(neck_curves[1])
            neck_all_measures.extend(neck_curves[2])
            neck_all_curves = [neck_all_cordinates, neck_all_positions, neck_all_measures]

        # return list(zip(*[neck_all_curves]))
        
        # bust planes box
        height = body_portion
        center = (0, body_min + (body_portion*5.1) + (height/2), 0)
        width = body_measures['bust_chest_girth']/tc.pi*1.3
        bust_box = BBox(width=width, height=height, center=tcenter(center))
        bust_triangle = lambda x: tc.FloatTensor([[10, x, 10], [10, x, -10], [-10, x, 0]]).to(device)

        # torso planes box
        width = body_measures['hip_girth']/tc.pi*1.3
        height = body_portion*1.5
        center = (0, body_min + (body_portion*3.75) + (height/2), 0)
        torso_box = BBox(width=width, height=height, center=tcenter(center))
        torso_triangle = lambda x: tc.FloatTensor([[10, x, 10], [10, x, -10], [-10, x, 0]]).to(device)

        # leg planes box
        width = body_width/2
        height = body_portion*3
        to_right = body_width/4
        center = (-to_right, body_min + body_portion + (height/2) ,0)
        leg_box = BBox(width=width, height=height, center=tcenter(center))
        leg_triangle = lambda x: tc.FloatTensor([[10, x, 10], [10, x, -10], [-10, x, 0]]).to(device)

        # arm planes box
        width = body_portion*2
        height = body_width
        c = body_measures['bust_chest_girth']/tc.pi/2*1.1
        center = (-c-(width/2), body_portion*5.5,0)
        arm_box = BBox(width=width, height=height, center=tcenter(center))
        arm_triangle = lambda x: tc.FloatTensor([[x, 10, 10], [x, 10, -10], [x, -10, 0]]).to(device)

        all_segments = [
            CurveUtils.calculate_curve(selected_body, bust_box, bust_triangle, 1, template, device),
            CurveUtils.calculate_curve(selected_body, torso_box, torso_triangle, 1, template, device),
            CurveUtils.calculate_curve(selected_body, leg_box, leg_triangle, 1, template, device),
            CurveUtils.calculate_curve(selected_body, arm_box, arm_triangle, 0, template, device),
            # CurveUtils.calculate_curve(selected_body, neck_box, neck_triangle, 0, template, device),
            neck_all_curves,
        ]

        return list(zip(*all_segments))# transpose trick