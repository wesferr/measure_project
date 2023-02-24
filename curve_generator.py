import torch as tc
from pytorch3d.transforms import euler_angles_to_matrix

from bbox import BBox
from curve_utils import CurveUtils


class CurveGenerator():

    @classmethod
    def calculate_curve(
        cls, body, bounding_box, axis, template,
        device, closed=True, rotation=tc.FloatTensor([])):

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

            triangles = CurveGenerator.define_triangle(axis, value, rotation, device)
            positions, coordinates = CurveGenerator.plane_triangle_colision(
                triangles, body[template], device, template
            )
            inside_filter = bounding_box.check_inside(positions)
            positions, coordinates = positions[inside_filter], coordinates[inside_filter]
            positions, coordinates = CurveUtils.sort_curve(positions, coordinates)
            measures = CurveUtils.calculate_distances(positions, closed)
            all_positions.append(positions)
            all_coordinates.append(coordinates)
            all_measures.append(measures)
        return all_coordinates, all_measures, all_positions

    @classmethod
    def define_triangle(cls, axis, displ, rotation, device):
        if axis == 0:
            triangle = [[ displ,  20,  11.6],
                        [ displ, -20,  11.6],
                        [ displ,   0, -23.2]]
        elif axis == 1:
            triangle = [[ 20, displ,  11.6],
                        [-20, displ,  11.6],
                        [  0, displ, -23.2]]
        if rotation.numel() == 0:
            triangle = tc.FloatTensor(triangle).to(device)
        else:
            triangle = tc.FloatTensor(triangle).to(device) @ rotation
        return triangle.unsqueeze(0)

    @classmethod
    def get_curves(cls, selected_body, selected_measure, template, device):

        body_measures = selected_measure/1000
        body_portion = body_measures['stature']/8
        body_min = selected_body[:,1].min()
        body_width = selected_body[:,0].max() - selected_body[:,0].min()

        # neck planes box
        height = body_portion
        width = body_portion
        center = (0, body_min + (body_portion*6.25) + (height/2), 0)
        neck_box = BBox(width=width, height=height, center=tc.FloatTensor(center).to(device))

        neck_all_cordinates = []
        neck_all_positions = []
        neck_all_measures = []
        for i in tc.arange(0,25, 0.5):
            neck_rotation_matrix = euler_angles_to_matrix(
                tc.tensor([float(-i)*(tc.pi/180),0,0]).to(device),
                "XYZ"
            )
            neck_curves = CurveGenerator.calculate_curve(
                selected_body, neck_box, 1, template,
                device, closed=False, rotation=neck_rotation_matrix)
            neck_all_cordinates.extend(neck_curves[0])
            neck_all_positions.extend(neck_curves[1])
            neck_all_measures.extend(neck_curves[2])
        neck_all_curves = [neck_all_cordinates, neck_all_positions, neck_all_measures]

        # bust planes box
        height = body_portion
        center = (0, body_min + (body_portion*5.1) + (height/2), 0)
        width = body_measures['bust_chest_girth']/tc.pi*1.3
        bust_box = BBox(width=width, height=height, center=tc.FloatTensor(center).to(device))

        # torso planes box
        width = body_measures['hip_girth']/tc.pi*1.3
        height = body_portion*1.5
        center = (0, body_min + (body_portion*3.75) + (height/2), 0)
        torso_box = BBox(width=width, height=height, center=tc.FloatTensor(center).to(device))

        # leg planes box
        width = body_width/2
        height = body_portion*3
        to_right = body_width/4
        center = (-to_right, body_min + body_portion + (height/2) ,0)
        leg_box = BBox(width=width, height=height, center=tc.FloatTensor(center).to(device))

        # arm planes box
        width = body_portion*2
        height = body_width
        chest_girth = body_measures['bust_chest_girth']/tc.pi/2*1.1
        center = (-chest_girth-(width/2), body_portion*5.5,0)
        arm_box = BBox(width=width, height=height, center=tc.FloatTensor(center).to(device))

        all_segments = [
            CurveGenerator.calculate_curve(selected_body, bust_box, 1, template, device),
            CurveGenerator.calculate_curve(selected_body, torso_box, 1, template, device),
            CurveGenerator.calculate_curve(selected_body, leg_box, 1, template, device),
            CurveGenerator.calculate_curve(selected_body, arm_box, 0, template, device),
            neck_all_curves,
        ]

        return list(zip(*all_segments))# transpose trick

    @classmethod
    def plane_triangle_colision(cls, planes, triangles, device, faces=tc.LongTensor()):
        centroid = tc.row_stack([
            triangles[:,0], triangles[:,1], triangles[:,2],
            triangles[:,0], triangles[:,1], triangles[:,2]
        ])
        vectors = tc.row_stack([
            triangles[:,1] - triangles[:,0], triangles[:,0] - triangles[:,1],
            triangles[:,0] - triangles[:,2], triangles[:,2] - triangles[:,0],
            triangles[:,2] - triangles[:,1], triangles[:,1] - triangles[:,2],
        ])

        # plane def = plane_normal, plane_offset, point 0, point 1, point 2
        plane_def = CurveGenerator.define_planes(planes)

        colision_points, _, _ = CurveGenerator.calculate_colision_1(
            centroid, vectors, plane_def[0], plane_def[1]
        )
        baricentric, triangles_filter = CurveGenerator.baricentric_coordinates(
            triangles[:,0].repeat(6,1),
            triangles[:,1].repeat(6,1),
            triangles[:,2].repeat(6,1),
            colision_points+centroid
        )
        _, planes_filter = CurveGenerator.baricentric_coordinates(
            plane_def[2], plane_def[3], plane_def[4], colision_points+centroid
        )
        all_filters = triangles_filter&planes_filter
        coordinates = tc.cat([faces.repeat(6,1).unsqueeze(0), baricentric], dim=-1)
        positions, coordinates = (colision_points+centroid)[all_filters], coordinates[all_filters]

        positions = positions.round(decimals=6)
        positions, index, counts = positions.unique(dim=0, return_inverse=True, return_counts=True)

        # get index of first recurrence of values in a array
        _, ind_sort = index.sort(stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = tc.cat((tc.LongTensor([0]).to(device), cum_sum[:-1]))
        index = ind_sort[cum_sum]
        coordinates = coordinates[index]
        return positions, coordinates

    @classmethod
    def ray_polygon_colision(cls, polygons, positions):
        centroid, nvectors = CurveGenerator.define_vectors(positions)
        #  plane def = plane_normal, plane_offset, point 0, point 1, point 2
        plane_def = CurveGenerator.define_planes(polygons)
        colision_points, points_filter, ray_offset = CurveGenerator.calculate_colision(
            centroid, nvectors, plane_def[0], plane_def[1]
        )
        _, bfilter = CurveGenerator.baricentric_coordinates(
            plane_def[2], plane_def[3], plane_def[4], colision_points+centroid
        )
        result = CurveUtils.get_closest_point(
            colision_points+centroid, ray_offset, points_filter, bfilter
        )
        return result, CurveUtils.calculate_distances(result)

    @classmethod
    def baricentric_coordinates(cls, point_a, point_b, point_c, goal_point):
        vab = (point_b - point_a).unsqueeze(0)
        vac = (point_c - point_a).unsqueeze(0)
        vap = goal_point - point_a

        dbb = (vab * vab).sum(axis=2)
        dbc = (vab * vac).sum(axis=2)
        dcc = (vac * vac).sum(axis=2)
        dpb = (vap * vab).sum(axis=2)
        dpc = (vap * vac).sum(axis=2)

        denom = dbb * dcc - dbc * dbc
        area_v = (dcc * dpb - dbc * dpc) /denom
        area_w = (dbb * dpc - dbc * dpb) /denom
        area_u = 1.0 - area_v - area_w

        test = (area_u>=0) & (area_v>=0) & (area_w>=0)
        return tc.stack([area_u,area_v,area_w], axis=2), test

    @classmethod
    def define_vectors(cls, positions):

        ones = tc.ones(positions.shape[0]).to("cuda")
        positions_range = positions.max(axis=0).values - positions.min(axis=0).values
        lower_dimension = positions_range.argmin()
        points_matrix = positions.clone()
        points_matrix[:,lower_dimension] = ones
        points_optimization = tc.linalg.solve(
            points_matrix.T@points_matrix,
            points_matrix.T@positions[:,lower_dimension]
        )
        new_y = points_matrix @ points_optimization
        points_matrix[:,lower_dimension] = new_y

        centroid = points_matrix.mean(axis=0)
        vectors = points_matrix - centroid
        vectors = vectors/tc.linalg.norm(vectors, axis=1).unsqueeze(1)
        return centroid, vectors

    @classmethod
    def define_planes(cls, polygons):
        polygon_0 = polygons[:,0]
        polygon_1 = polygons[:,1]
        polygon_2 = polygons[:,2]
        cross = tc.cross(polygon_0 - polygon_1, polygon_0 - polygon_2, dim=1)
        plane_normal = cross / tc.linalg.norm(cross, axis=1).unsqueeze(1)
        plane_offset = -tc.sum(polygon_0*plane_normal, axis=1)
        return plane_normal, plane_offset, polygon_0, polygon_1, polygon_2

    @classmethod
    def calculate_colision_1(cls, origin, vectors, normals, distances):
        prod_norm_vec = normals@vectors.T
        prod_norm_orig = normals@origin.T
        ray_offset = -((prod_norm_orig+distances.unsqueeze(1))/prod_norm_vec)
        colision_point = vectors.unsqueeze(0) * ray_offset.unsqueeze(2)
        return colision_point, (ray_offset >= 0), ray_offset

    @classmethod
    def calculate_colision(cls, origin, vectors, normals, distances):
        prod_norm_vec = vectors@normals.T
        prod_norm_orig = origin@normals.T
        ray_offset = -((prod_norm_orig+distances.unsqueeze(0))/prod_norm_vec)
        colision_point = ray_offset.unsqueeze(2) @ vectors.unsqueeze(1)
        return colision_point, (ray_offset >= 0), ray_offset
