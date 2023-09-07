from vincenty import vincenty
import numpy as np
import json
import requests
import numpy as np
def get_json_data(url):
    r = requests.get(url)
    if r.status_code == 200:
        print('JSON файл успешно получен ')
    else:
        print('Ошибка при получений JSON файла ')
    return r

def convert_data_distance(data):
    if isinstance(data, str):
        data = json.loads(data)
    distance = []
    for item in data:
        coordX = item["coordX"]
        coordY = item["coordY"]
        distance.append([coordX, coordY])
    distance_arr = np.array(distance)
    return distance_arr
def copy_matrix(x,y, coord):

    coord_new = np.zeros((coord.shape[0]+1,coord.shape[1]))
    coord_new[0,0],coord_new[0,1] = x,y
    for i in range(1,coord_new.shape[0]):
        for j in range(coord_new.shape[1]):
            coord_new[i,j] = coord[i-1,j]

    return coord_new



def marix(data_coord):
    matrix_mas = np.zeros((len(data_coord), len(data_coord)))
    for i in range(len(matrix_mas)):
        for j in range(len(matrix_mas)):
            if i > j:
                first = (data_coord[i,0], data_coord[i,1])
                second = (data_coord[j,0], data_coord[j,1])
                matrix_mas[i, j] = int(vincenty(first, second))
                matrix_mas[j, i] = int(vincenty(first, second))
            if i == j:
                matrix_mas[i, j] = 0
    return matrix_mas


def start_matrix_create():
    url = 'http://localhost:3007/api/clients/'
    x_coord = 54.735152
    y_coord = 55.958736
    client = get_json_data(url)
    client = client.text
    print(client)
    dist = convert_data_distance(client)
    dist = copy_matrix(x_coord,y_coord,dist)
    res = marix(dist)


    # print(dist)
    #
    # dir_path = pathlib.Path.cwd()
    # path_result = Path(dir_path, 'C:\AllPycharn\SDVRP\data\Result.xlsx')
    # res = marix(dist)
    # print(res)
    # df = pd.DataFrame(res)
    # df.to_excel(excel_writer = path_result, index=False)
    return res
