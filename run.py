import json

import numpy as np
from flask import Flask, request,jsonify
from split_route_run import *
from create_matrix import *
import requests
app = Flask(__name__)


def check_params_middleware(func):
    def wrapper(*args, **kwargs):
        avg_speed = request.args.get('speed')
        fuel_consum = request.args.get('consumption')
        price_fuel = request.args.get('cost')
        unload_time = request.args.get('unloading')
        late_penalty = request.args.get('fine')
        if not avg_speed or not fuel_consum or not price_fuel or not unload_time or not late_penalty:
            return jsonify({'error': 'Не все параметры указаны'})
        try:
            avg_speed = int(avg_speed)
            fuel_consum = int(fuel_consum)
            price_fuel = int(price_fuel)
            unload_time = int(unload_time)
            late_penalty = int(late_penalty)
        except ValueError:
            return jsonify({'error': 'Неправильный формат данных'})
        return func(avg_speed, fuel_consum, price_fuel, unload_time, late_penalty, *args, **kwargs)
    return wrapper

def copy_window(start_hour,start_min, end_hour,end_min,data_time_start,data_time_end):
    datatimestartnew = np.zeros((data_time_start.shape[0]+1,data_time_start.shape[1]))
    datatimeendnew = np.zeros((data_time_end.shape[0] + 1, data_time_end.shape[1]))
    datatimestartnew[0,0],datatimestartnew[0,1]=start_hour,start_min
    datatimeendnew[0, 0], datatimeendnew[0, 1] = end_hour, end_min
    for i in range(1,datatimestartnew.shape[0]):
        for j in range(datatimestartnew.shape[1]):
            datatimestartnew[i,j] = data_time_start[i-1,j]
    for i in range(1, datatimeendnew.shape[0]):
        for j in range(datatimeendnew.shape[1]):
            datatimeendnew[i, j] = datatimeendnew[i - 1, j]
    return datatimestartnew, datatimeendnew
@app.route('/routes_with_splits')
@check_params_middleware
def run_split(avg_speed, fuel_consum, price_fuel, unload_time, late_penalty):
    print("START PROG")
   # initial_temperature = 20000  # Начальная температура
    initial_temperature = 20000  # Начальная температура
    final_temperature = 0.11  # конечная температура
    # кол-во поколений
    count_generation = 100
    # процент мутации
    procent_mutation = 0.4
    truck_capacity = 1500
    count_vehicle = 25
    start_hour = 9
    start_min = 0
    end_hour = 17
    end_min = 0
    # размер популяций
    size_pop = 40

    adj_matrix = start_matrix_create()

    print("Матрица расстояний: \n", adj_matrix, "\n")

    url = 'http://localhost:3007/api/clients/'
    client = get_json_data(url)
    client = client.text
    client = convert_data_client(client)
    time_window = get_json_data(url)
    time_window = time_window.text
    data_time_start, data_time_end = convert_data_windows(time_window)
    data_time_start,data_time_end = copy_window(start_hour,start_min, end_hour,end_min,data_time_start,data_time_end)

    url = 'http://localhost:3007/api/cargos/'
    cargos = get_json_data(url)
    cargos = cargos.text
    cargos = convert_data_demand(cargos)
    print(cargos)
    needs_corteg = create_need(client,cargos)
    count_client = len(needs_corteg)

    print(needs_corteg)

    all_route_list = []
    part_route = []
    flag = 0
    one_part_curr_corteg = []
    two_part_curr_corteg = []
    # while flag == 0:
    #
    #     current_corteg = copy.deepcopy(needs_corteg)
    #
    #     rand_ind1 = random.randint(0, len(current_corteg) - 1)
    #     rand_ind2 = random.randint(0, len(current_corteg) - 1)
    #     val_ind1 = current_corteg[rand_ind1]
    #     val_ind2 = current_corteg[rand_ind2]
    #     current_corteg[rand_ind1] = val_ind2
    #     current_corteg[rand_ind2] = val_ind1
    #     # current_corteg = needs_corteg.copy()
    while flag == 0:

        current_corteg = copy.deepcopy(needs_corteg)

        rand_ind1 = random.randint(0, len(current_corteg) - 1)
        rand_ind2 = random.randint(0, len(current_corteg) - 1)
        val_ind1 = current_corteg[rand_ind1]
        val_ind2 = current_corteg[rand_ind2]
        current_corteg[rand_ind1] = val_ind2
        current_corteg[rand_ind2] = val_ind1
        # current_corteg = needs_corteg.copy()
        len_gap = len(current_corteg) // 2
        for i in range(len_gap):
            one_part_curr_corteg.append(current_corteg[i])
        count_one = len(one_part_curr_corteg)
        route_1 = SplitDelivery(count_vehicle, count_one, truck_capacity, one_part_curr_corteg)

        for i in range(len_gap, count_client):
            two_part_curr_corteg.append(current_corteg[i])
        count_two = len(two_part_curr_corteg)
        route_2 = SplitDelivery(count_vehicle, count_two, truck_capacity, two_part_curr_corteg)
        part_route = route_1 + route_2

        json_veh = len(part_route)
        json_route = convert_to_json(part_route, json_veh)
        json_route = json.loads(json_route)
        print(json_route)
        r = requests.post('http://localhost:5007/api/routes', json=json_route)
        print(r.status_code)

        #r = requests.get('http://localhost:5007/api/pack?who=1')
        r = requests.get('http://localhost:5007/api/pack?save=1&who=1')
        print(r.status_code)
        packdata = r.text
        packdata1 = json.loads(packdata)
        success = packdata1["success"]
        if success == 1:
            all_route_list = part_route
            volume = packdata1["volume"]
            flag = 1
        else:
            flag = 0



        #С помощью раздельной доставки распределяем клиентов по грузовикам и получаем марщрутный лист
    need_size_route = len(all_route_list)
    route_lis = [[] for i in range(need_size_route)]


    GA_route_list = [[] for i in range(need_size_route)]
    SA_route_list = [[] for i in range(need_size_route)]
    all_cost_list = [0 for i in range(need_size_route)]
    all_cost_list_SA = [0 for i in range(need_size_route)]

    print(all_route_list)
    for k in range(need_size_route):
        route_lis[k] = all_route_list[k]
    print(route_lis)
    route_lis = route_lis
    need_size_route = len(route_lis)
    print(route_lis)
    for k in range(need_size_route):

        route = route_lis[k]
        print(f"route {route}")
        need_count_vehicle = len(route_lis)


        profit_route_by_GA = []
        cost_route_by_GA = []
        profit_route_by_SA = []
        cost_route_by_SA = []

        profit_route_by_GA, cost_route_by_GA = GeneticAlgorythm(count_generation,procent_mutation, route,adj_matrix,size_pop,
                                                                      fuel_consum, price_fuel, avg_speed, start_hour,
                                                                      start_min,
                                                                      end_hour, end_min, unload_time, late_penalty,
                                                                      data_time_start, data_time_end
                                                                      )
        #print(profit_route_by_GA,cost_route_by_GA)
        profit_route_by_SA, cost_route_by_SA = simulated_annealing(route, initial_temperature,
                                                                         final_temperature, adj_matrix,
                                                                         fuel_consum, price_fuel, avg_speed,
                                                                         start_hour,
                                                                         start_min,
                                                                         end_hour, end_min, unload_time,
                                                                         late_penalty,
                                                                         data_time_start, data_time_end
                                                                         )
        GA_route_list[k] = profit_route_by_GA
        SA_route_list[k] = profit_route_by_SA
        all_cost_list[k] = cost_route_by_GA
        all_cost_list_SA[k] = cost_route_by_SA

    final_cost_GA = sum(all_cost_list)
    final_cost_SA = sum(all_cost_list_SA)
    print(f"Стоимость по ГА:{final_cost_GA}")
    print(f"Стоимость по ИО:{final_cost_SA}")
    if final_cost_GA <= final_cost_SA:
        # final = final_GA
        final_route = GA_route_list
        final_cost = final_cost_GA
        print("Выгодный маршрут по генетическому алгоритму")
        print(final_route)
        print(final_cost)
    else:
        # final = final_SA
        final_route = SA_route_list
        final_cost = final_cost_SA
        print("Выгодный маршрут по алгоритму имитации отжига ")
        print(final_route)
        print(final_cost)

    json_cont=len(final_route)
    json_post_final = convert_to_json(final_route,json_cont)
    json_post_final = json.loads(json_post_final)
    print(json_post_final)

    r = requests.post('http://localhost:5007/api/routes', json=json_post_final)
    print(r.status_code)

    # r = requests.get('http://localhost:5007/api/pack?save=1&who=1')
    r = requests.get('http://localhost:5007/api/pack?who=1')
    print(f"Штраф за упаковку: {volume}")
    print(r.status_code)
    packdata = r.text
    packdata1 = json.loads(packdata)
    success = packdata1["success"]
    print(success)

    final_cost = final_cost + volume
    json_final = convert_final_json(final_route, json_cont,final_cost)
    json_final = json.loads(json_final)
    print(f"Итоговая стоимость c учетом штрафа: {final_cost}")
    # r = requests.post('http://localhost:5007/api/routes', json=json_final)
    # print(json_final)
    # print(r.status_code)

    return json_final


if __name__ == '__main__':
    app.run(port = 5005)
