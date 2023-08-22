import torch as tc
import pandas as pd

class CurveUtils():

    @classmethod
    def calculate_distances(cls, positions, closed=True):
        if positions.numel() <= 3:
            return tc.FloatTensor(0)
        distances = tc.cdist(positions, positions)
        if closed is True:
            distance = distances.diagonal(1).sum()
        else:
            bigger = distances.diagonal(1).max()
            distance = distances.diagonal(1).sum() - bigger
        return distance*100
    
    @classmethod
    def linear_distance(cls, positions):
        higher_dimension = positions.std(axis=0).argmax()
        positions = positions[positions[:, higher_dimension].argsort()]
        distances = tc.cdist(positions, positions)
        distance = distances.diagonal(1).sum()
        return distance*100

    @classmethod
    def calculate_height(cls, floor, coordiantes, body):
        positions = CurveUtils.generate_positions(coordiantes, body)
        height = abs(floor - positions.mean(axis=0)[1])
        return height * 100

    @classmethod
    def calculate_angle(cls, vectors, axis):
        inner_product = (vectors * axis.unsqueeze(0)).sum(dim=1)
        a_norm = tc.linalg.norm(vectors,axis=1)
        b_norm = tc.linalg.norm(axis)
        cos = inner_product / (2 * a_norm * b_norm)
        angle = tc.acos(cos)
        return angle

    @classmethod
    def get_closest_point(cls, colision_point, ray_offset, pfilter, bfilter):
        indices = tc.where(pfilter & bfilter)
        matrix = tc.sparse_coo_tensor(tc.row_stack(indices), ray_offset[indices]-100).to_dense()
        indices = indices[0].unique()
        best = matrix.min(axis=1)[1]
        selected_indices = [indices, best[indices]]
        return colision_point[selected_indices]

    @classmethod
    def generate_positions(cls, coordinates, vertices):
        indices = coordinates[:,:3].type(tc.LongTensor)
        parameters = coordinates[:,3:6]
        positions = vertices[indices] * parameters.unsqueeze(2)
        return positions.sum(axis=1)

    @classmethod
    def sort_curve(cls, positions, coordinates):
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

    @classmethod
    def calculate_derived(cls, selected_body_t):
        aditional_measures = []
        aditional_positions = []

        body_min = selected_body_t[:,1].min()
        body_max = selected_body_t[:,1].max()
        height = body_max - body_min
        aditional_measures.append(height*100)

        return aditional_measures, aditional_positions

    @classmethod
    def select_better(cls, result, body, selected_measure, device):
        curve_index = {
            'neck_girth':4, # 5.3.2
            'bust_chest_girth': 0, # 5.3.4
            'waist_girth': 1, # 5.3.10
            'hip_girth': 1, # 5.3.13
            'upper_arm_girth': 3, # 5.3.16
            'thigh_girth': 2, # 5.3.20
        }
        curve_names = list(curve_index.keys())

        curves = result[0]
        measures = result[1]
        positions = result[2]

        all_positions = []
        all_measures = []
        best_curves = []

        for measure in curve_names:
            measures_index = tc.FloatTensor(measures[curve_index[measure]]).to(device)
            diff = abs(measures_index - selected_measure[measure]/10)
            best = diff.argmin()
            all_positions.append(positions[curve_index[measure]][best])
            all_measures.append(measures_index[best])
            best_curves.append(curves[curve_index[measure]][best])

        derived_measures, derived_positions = CurveUtils.calculate_derived(body)
        all_measures.extend(derived_measures)
        all_positions.extend(derived_positions)

        selected_measure_tensor = tc.FloatTensor(selected_measure[curve_names+['stature']])
        selected_measure_tensor = selected_measure_tensor.to(device).cpu().numpy()
        all_measures = tc.FloatTensor(all_measures).to(device).cpu().numpy()

        data = {
            'original': selected_measure_tensor/10,
            'measured': all_measures,
            'error(mm)': abs(selected_measure_tensor/10 - all_measures)*10
        }
        data = pd.DataFrame(data)
        return best_curves, data
