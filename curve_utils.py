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

    def calculate_distances(positions):
        if positions.numel() == 0:
            return 0
        distances = tc.cdist(positions, positions)
        distance = distances.diagonal(1).sum()
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

    def plane_triangle_colision(planes, triangles):
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
        P, _, _ = CurveUtils.calculate_colision(centroid, vectors, n, d)
        baricentric, triangles_filter = CurveUtils.baricentric_coordinates(triangles[:,0].repeat(6,1), triangles[:,1].repeat(6,1), triangles[:,2].repeat(6,1), P+centroid)
        baricentric, planes_filter = CurveUtils.baricentric_coordinates(p0, p1, p2, P+centroid)
        uniqueness_filter = (P+centroid).unique(dim=0)
        return (P+centroid)[triangles_filter&planes_filter]
        
    def ray_polygon_cosilion(polygons, positions):
        centroid, nvectors = CurveUtils.define_vectors(positions)
        n, d, p0, p1, p2 = CurveUtils.define_planes(polygons)
        P, pfilter, t = CurveUtils.calculate_colision(centroid, nvectors, n, d)
        baricentric, bfilter = CurveUtils.baricentric_coordinates(p0, p1, p2, P+centroid)
        result = CurveUtils.get_closest_point(P+centroid, t, pfilter, bfilter)
        return result, CurveUtils.calculate_distances(result)

    def define_vectors(positions):
        # project points
        ones = tc.ones(positions.shape[0]).to("cuda")
        A = tc.column_stack([positions[:,0], ones, positions[:,2]])
        new_y = A @ tc.linalg.solve(A.T@A, A.T@positions[:,1])
        A[:,1] = new_y

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

    def calculate_colision(origin, vectors, normals, distances):
        if origin.shape == vectors.shape:
            nd = (normals@vectors.T)
            pn = (normals@origin.T)
            t = -((pn+distances.unsqueeze(1))/nd)
            P = vectors.unsqueeze(0) * t.unsqueeze(2)
        else:
            nd = (vectors@normals.T)
            pn = (origin@normals.T)
            t = -((pn+distances.unsqueeze(0))/nd)
            P = t.unsqueeze(2) @ vectors.unsqueeze(1)
        return P, (t >= 0), t

    # def baricentric_coordinates(a, b, c, p):
        
    #     vab = b - a
    #     vbc = c - b
    #     vca = a - c

    #     vap = p - a
    #     vbp = p - b
    #     vcp = p - c

    #     cross = tc.cross(vab,vbc)
    #     n = cross / tc.linalg.norm(cross)
        
    #     ABC = (n * tc.cross(vab, vbc)).sum(axis=1) / 2
    #     ABP = (n * tc.cross(vab.unsqueeze(0), vbp)).sum(axis=2) / 2
    #     BCP = (n * tc.cross(vbc.unsqueeze(0), vcp)).sum(axis=2) / 2
    #     CAP = (n * tc.cross(vca.unsqueeze(0), vap)).sum(axis=2) / 2

    #     w = ABP/ABC
    #     u = BCP/ABC
    #     v = CAP/ABC

    #     test = (u>=0) & (v>=0) & (w>=0)

    #     return tc.stack([u,v,w], axis=2), test

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
        return tc.stack([u,v,w], axis=1), test


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