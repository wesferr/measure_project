import torch as tc

def load_template(file_path, device):
    faces = []
    normals = []
    with_normals = False
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            if line[0] == "f":
                line_split = line[1:].split()
                line_split = [i.split("//") for i in line_split]
                if len(line_split[0]) == 2:
                    line_split = [ ( int(i[0]), int(i[1]) ) for i in line_split ]
                else:
                    line_split = [ [int(i[0]),] for i in line_split ]
                line_split = tc.LongTensor(line_split).to(device)

                faces.append(line_split[:, 0])
                if len(line_split[0]) > 1:
                    normals.append(line_split[:, 1])
                    with_normals = True

    if with_normals:
        return tc.row_stack(faces)-1, tc.row_stack(normals)
    else:
        return tc.row_stack(faces)-1, tc.LongTensor().to(device)

def load_mesh(file_path, device):
    vertex = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line[0] == 'v':
                line.replace('\n', ' ')
                tmp = list(map(float, line[1:].split()))
                vertex.append(tmp)
            else:
                continue
    return tc.FloatTensor(vertex).to(device)

def save_obj(path, pontos, faces=tc.LongTensor()):
    with open(path, "w", encoding="utf-8") as file:
        for ponto in pontos:
            file.write(f"v {ponto[0]} {ponto[1]} {ponto[2]}\n")
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")
