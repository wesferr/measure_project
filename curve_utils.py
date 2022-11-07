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


    def ray_trace(vertices, faces, positions):
        nrays, centroid, nvectors = CurveUtils.define_vectors(faces, positions)
        n, d, p0, p1, p2 = CurveUtils.define_planes(vertices, faces, nrays)
        P, pfilter, t = CurveUtils.calculate_colision(centroid, nvectors, n, d)
        baricentric, bfilter = CurveUtils.baricentric_coordinates(p0, p1, p2, P)
        result = CurveUtils.get_closest_point(P, t, pfilter, bfilter, positions)
        return result, CurveUtils.calculate_distances(result)

    def define_vectors(faces, positions):
        # project points
        ones = tc.ones(positions.shape[0]).to("cuda")
        A = tc.column_stack([positions[:,0], ones, positions[:,2]])
        new_y = A @ tc.linalg.solve(A.T@A, A.T@positions[:,1])
        A[:,1] = new_y

        # calculate centroid and vectors
        centroid = A.mean(axis=0)
        vectors = A - centroid
        vectors = vectors/tc.linalg.norm(vectors, axis=1).unsqueeze(1)
        nrays = vectors.shape[0]
        nvectors = vectors.repeat_interleave(faces.shape[0], 0)
        return nrays, centroid, nvectors

    def define_planes(vertices, faces, nrays):
        polygons = vertices[faces]
        p0 = polygons[:,0].repeat(nrays, 1)
        p1 = polygons[:,1].repeat(nrays, 1)
        p2 = polygons[:,2].repeat(nrays, 1)
        cross = tc.cross(p0 - p1, p0 - p2, dim=1)
        n = cross / tc.linalg.norm(cross, axis=1).unsqueeze(1)
        d = -tc.sum(p0*n, axis=1)
        return n, d, p0, p1, p2

    def calculate_colision(origin, vectors, normals, distances):
        nd = (normals*vectors).sum(1)
        pn = (origin*normals).sum(1)
        t = -((pn+distances)/nd)
        P = origin + (vectors * t.unsqueeze(1))
        return P, (t >= 0), t

    def baricentric_coordinates(a, b, c, p):
        
        vab = b - a
        vbc = c - b
        vca = a - c
        
        vap = p - a
        vbp = p - b
        vcp = p - c


        cross = tc.cross(vab,vbc)
        n = cross / tc.linalg.norm(cross)
        
        ABC = (n * tc.cross(vab, vbc)).sum(axis=1) / 2
        ABP = (n * tc.cross(vab, vbp)).sum(axis=1) / 2
        BCP = (n * tc.cross(vbc, vcp)).sum(axis=1) / 2
        CAP = (n * tc.cross(vca, vap)).sum(axis=1) / 2

        w = ABP/ABC
        u = BCP/ABC
        v = CAP/ABC

        test = (u>=0) & (v>=0) & (w>=0)

        return tc.column_stack([u,v,w]), test

    def get_closest_point(P, t, pfilter, bfilter, position):

        psize = position.shape[0]
        tsize = t.shape[0]//psize

        tempP = P.reshape([psize, tsize, 3])
        tempt = t.reshape([psize, tsize])
        pfilter = pfilter.reshape([psize, tsize])
        bfilter = bfilter.reshape([psize, tsize])

        indices = tc.where(pfilter & bfilter)
        matrix = tc.sparse_coo_tensor(tc.row_stack(indices), tempt[indices]-100).to_dense()
        selected_indices = [indices[0].unique(), matrix.min(axis=1)[1]]
        return tempP[selected_indices]

    def generate_positions(coordinates, vertices):
        indices = coordinates[:,:3].type(tc.LongTensor)
        parameters = coordinates[:,3:6]
        positions = vertices[indices] * parameters.unsqueeze(2)
        return positions.sum(axis=1)