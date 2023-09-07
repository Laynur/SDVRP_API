import random
import copy
import math
import numpy as np
import json
import requests
import sys
#функция для генерации потребностей с использованием кортежей
def create_tuple(count_client, data_customer_need):
    tuples = []
    for i in range(count_client):
        for j in range(1):
            number = i + 1
            value = data_customer_need[i][j]
            tuple_element = (number, value)
            tuples.append(tuple_element)
    return tuples

#функция для подсчета стоимости пути
def calculate_path_cost(adj_matrix, path):
    cost = 0
    for i in range(1,len(path)):
        first = path[i-1]
        second = path[i]
        cost += adj_matrix[first][second]
    return cost

def route_length(data_dist,path):
    distnce = 0
    for i in range(1,len(path)):
        first = path[i-1]
        second = path[i]
        distnce += data_dist[int(first), int(second)]
    return distnce

def route_cost(population, data_dist, fuel_consum, price_fuel, avg_speed, start_hour, start_min, end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end):
    #distance_pop = route_length(population, data_dist)
    penalty_pop, distance_pop = penalty(population, data_dist, avg_speed, start_hour, start_min, end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
    price = (distance_pop / 100 * fuel_consum * price_fuel) + penalty_pop
    #price = (distance_pop / 100 * fuel_consum * price_fuel)
    return price

def sum_time(hour, minuts, time):
    minuts = minuts + time
    while minuts >= 60:
        minuts = minuts - 60
        hour = hour + 1
    return hour, minuts
def diff_time(time, data_time, index):
    hour = data_time[index, 0]
    minuts = data_time[index, 1] - time
    while minuts < 0:
        minuts = 60 + minuts
        hour = hour - 1
    return hour, minuts

# Функция расчета штрафа на маршрут
def penalty(population, data_dist, avg_speed, start_hour, start_min, end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end):
    # p_to_p_time = np.zeros((population.shape[1] - 1,1))
   # distance_pop = calculate_path_cost(data_dist, population)
    distance_pop = route_length(data_dist,population)
    penalty_pop = 0
    hour = start_hour
    minuts = start_min

    for j in range(1, len(population)):
        time = (data_dist[population[j - 1]][population[j]] / avg_speed) * 60
        time = int(time)

        if hour == start_hour and minuts == start_min:
            hour, minuts = sum_time(hour,minuts,time)
            if (hour < data_time_start[int(population[j]),0]) or (hour == data_time_start[int(population[j]),0] and minuts < data_time_start[int(population[j]),1]):
                hour, minuts = diff_time(time, data_time_start, int(population[j]))
        if (hour < end_hour) or (hour == end_hour and minuts < end_min):
            hour, minuts = sum_time(hour, minuts, time)
            if (hour < data_time_start[int(population[j]), 0]) or (hour == data_time_start[int(population[j]), 0] and minuts < data_time_start[int(population[j]), 1]):
                penalty_pop = penalty_pop + late_penalty
                hour = data_time_start[int(population[j]), 0]
                minuts = data_time_start[int(population[j]), 1]
                hour, minuts = sum_time(hour, minuts, unload_time)
            elif (hour > data_time_end[int(population[j]), 0]) or (hour == data_time_end[int(population[j]), 0] and minuts > hour > data_time_end[int(population[j]), 1]):
                penalty_pop = penalty_pop + late_penalty
            else:
                hour, minuts = sum_time(hour, minuts, unload_time)
        else:
            if population[j] != 0:
                penalty_pop = penalty_pop + late_penalty
                distance_pop = distance_pop - data_dist[population[j - 1]][population[j]]

        # hour = start_hour
        # minuts = start_min
    point1 = 0

    return  penalty_pop, distance_pop




#функция возравщающая индекс маршрута, с наибольшим(невыгодным) значением
def get_index_max_value_of_population(adj_matrix,population, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end):
    max_val = 0
    index_val = 0
    for i in range(len(population)):
        current_val = route_cost(population[i], adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
        if current_val >= max_val:
            max_val = current_val
            index_val = i
    return index_val
#Функция для добавления депо в маршрутный лист
def sel_elem(list_of_val,count_client,count_vehicle):
    new_list = []
    new_complet_list = [[] for i in range(count_vehicle)]
    for i in range(count_vehicle):
        new_list.clear()
        new_list.append(0)
        for j in range(count_client):
            if list_of_val[i][j]!=0:
                new_list.append(list_of_val[i][j])
        new_list.append(0)
        b = new_list.copy()
        # b= copy(new_list)
        new_complet_list[i] = b

    return new_complet_list
#Функция для удаления пустых клиентов из маршрутного листа
def del_null_clients(head_list):
    new_list = []
    for i in range(len(head_list)):
        if len(head_list[i])!=2:
            new_list.append(head_list[i])
    return new_list

def create_pop(size_pop, route):
    population = [[] for i in range(size_pop)]
    print(route)
    size_route = len(route)
    #copyroute = route.copy()
    for i in range(size_pop):
        #route = copyroute
        random_point_1 = random.randint(1,size_route-2)
        random_point_2 = random.randint(1,size_route-2)
        value_1 = route[random_point_1]
        value_2 = route[random_point_2]
        route[random_point_1] = value_2
        route[random_point_2] = value_1
        population[i] = route.copy()

    return population

def convert_to_json(routes_list,json_veh):
    routes_json = {"routes": [], "vehicle":[]}
    for route in routes_list:
        route_str = '-'.join(str(num) for num in route)
        routes_json["routes"].append(route_str)
    idx = []
    for i in range(json_veh):
        idx.append(i+1)
    routes_json["vehicle"] = [str(v) for v in idx]
    json_data = json.dumps(routes_json)
    return json_data


def convert_final_json(routes_list,json_veh,json_cost):
    routes_json = {"routes": [], "vehicle": [], "price": json_cost}
    for route in routes_list:
        route_str = '-'.join(str(num) for num in route)
        routes_json["routes"].append(route_str)
    idx = []
    for i in range(json_veh):
        idx.append(i + 1)
    routes_json["vehicle"] = [str(v) for v in idx]
    json_data = json.dumps(routes_json)
    return json_data

def send_json_data(url, data):
    r = requests.post(url, json=data)
    if r.status_code == 200:
        print('JSON файл успешно отправлен на сервер')
    else:
        print('Ошибка при отправке JSON файла на сервер')

def get_json_data(url):
    r = requests.get(url)
    if r.status_code == 200:
        print('JSON файл успешно получен ')
    else:
        print('Ошибка при получений JSON файла ')
    return r

def GeneticAlgorythm(count_generation,procent_mutation,clients,adj_matrix,size_popul, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end):

    new_population = create_pop(size_popul, clients)
    if len(clients) > 3:

        for k in range(count_generation):
            parent1,parent2 = Selection_perfect_to_random(new_population,adj_matrix,fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
            child1 = Crossing(parent1,parent2)

            mutation_child = Mutation(procent_mutation,child1)

            child1_val = route_cost(child1, adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
            mutch1_val = route_cost(mutation_child, adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)

            if child1_val < mutch1_val:
                newchild = child1
            else:
                newchild = mutation_child

            val_index = get_index_max_value_of_population(adj_matrix,new_population,fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
            new_population[val_index] = newchild

    profit = sys.maxsize
    for i in range(len(new_population)):
        path = new_population[i]
        elem = route_cost(path, adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
        if elem < profit:
            profit = elem
            profit_index = i
    profit_route = new_population[profit_index]
    return profit_route, profit

def Selection_perfect_to_random(population,adj_matrix,fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end): #Лучший в популяции + случайно выбранный
    #min_fitness_val = 1000000000
    min_fitness_val = sys.maxsize
    for i in range(len(population)):
        curr_fitness = route_cost(population[i], adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
        if curr_fitness <= min_fitness_val:
            min_fitness_val = curr_fitness
            parent1 = population[i]
    parent2 = population[random.randint(0,len(population)-1)]
    if parent1 == parent2:
        parent2 = population[random.randint(0,len(population)-1)]
    return parent1,parent2


def Crossing(parent_one,parent_two): #Оператор скрещивания.

    gap = len(parent_one) // 3
    count = len(parent_one)
    child = [0 for i in range(count)]
    for i in range(gap):
        child[i] = parent_one[i]
    for i in range(gap, count):
        child[i] = parent_two[i]

    for i in range(1, count - 1):
        elem_p1 = parent_one[i]
        elem_p2 = parent_two[i]
        final_elem = None
        if elem_p1 not in child:
            final_elem = elem_p1
        elif elem_p2 not in child:
            final_elem = elem_p2
        seen = set()
        for j in range(1, count - 1):
            if final_elem is not None and child[j] in seen:
                child[j] = final_elem
            else:
                seen.add(child[j])
    return child


def Mutation(p_ver, gen):
    start_mutation = random.randint(0,1)
    if p_ver > start_mutation:
        random_val_1 = random.randint(1,len(gen)-2)
        random_val_2 = random.randint(1,len(gen)-2)
        if random_val_2 == random_val_1:
            random_val_2 = random.randint(1,len(gen)-2)
        val1 = gen[random_val_1]
        val2 = gen[random_val_2]
        gen[random_val_1] = val2
        gen[random_val_2] = val1
    else:
        val1 = gen[1]
        val2 = gen[len(gen)-2]
        gen[1] = val2
        gen[len(gen)-2] = val1
    return gen




def simulated_annealing(path, initial_temperature, final_temperature, adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end):
    current_path = path.copy()
    current_cost = route_cost(current_path, adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)
    curr_iter_val = 1
    temperature = initial_temperature
    alph_temp = 0.89
    all_solution = []
    while temperature > final_temperature:
        # Generate a new candidate solution by swapping two random points in the path
        new_path = current_path.copy()
        random_point_index1 = random.randint(1, len(new_path) - 2)
        random_point_index2 = random.randint(1, len(new_path) - 2)
        random_point1 = new_path[random_point_index1]
        random_point2 = new_path[random_point_index2]
        new_path[random_point_index1] = random_point2
        new_path[random_point_index2] = random_point1

        # Calculate the cost of the new solution
        new_cost = route_cost(new_path, adj_matrix, fuel_consum, price_fuel, avg_speed, start_hour, start_min,
                                                end_hour, end_min, unload_time, late_penalty, data_time_start, data_time_end)

        # находим разницу между стоимостями нового и предыдущего решения;
        delta = new_cost - current_cost

        if delta <= 0:
            probability = 1
        else:
            probability = math.exp(-delta / temperature)
        if random.randint(0,1) <= probability:
            # Accept the new solution with a probability determined by temperature and delta
            current_path = new_path.copy()
            current_cost = new_cost
            tuples = (current_path, current_cost)
            all_solution.append(tuples)
        #  понижаем температуру
        #print(f"текущее решение {current_path} и его стоимость {current_cost}")
        #temperature = temperature/(curr_iter_val + 1)
        temperature *= alph_temp
        curr_iter_val = curr_iter_val + 1
    best_solution = min(all_solution, key=lambda x:x[1])
    current_path = best_solution[0]
    current_cost = best_solution[1]
    return current_path, current_cost

def SplitDelivery(count_vehicle, count_clients, truck_capacity, list_of_need):
    current_route = [[0] * (count_clients) for i in range(count_vehicle)]
    vehicle = 1

    current_capacity = truck_capacity
    k = 0
    q = list_of_need[0][1]
    for i in range(count_clients):
        while sum(list_of_need[i][1])!=0:
            if current_capacity == 0:
                vehicle = vehicle + 1
                current_capacity = truck_capacity
                k = k + 1
            if vehicle >= count_vehicle:
                print("Недостаточно ТС")
                return -1
            if sum(list_of_need[i][1]) <= current_capacity:
                q_need = list_of_need[i][1]
                q_need.sort()
                for l in range(len(q_need)):
                    if q_need[l]<=current_capacity:
                        q = q_need[l]
                        current_capacity = current_capacity - q
                        q_need[l] = 0
                    else:
                        current_capacity = 0
                index_clients = list_of_need[i][0]
                current_route[k][i] = index_clients
            else:
                if sum(list_of_need[i][1]) >= current_capacity:
                    q_need = list_of_need[i][1]
                    q_need.sort()
                    for l in range(len(q_need)):
                        if q_need[l] <=current_capacity:
                            q = q_need[l]
                            current_capacity = current_capacity - q
                            q_need[l] = 0
                            index_clients = list_of_need[i][0]
                            current_route[k][i] = index_clients
                        else:
                            current_capacity = 0

    #print(f"needs vehicle = {vehicle}")
    print(f"список маршрутов: {current_route} для кол-во клиентов = {count_clients}")
    current_route = sel_elem(current_route, count_clients, count_vehicle)
    print(f"новый список маршрутов: {current_route}")
    # Удаляем из маршрутного листа пустых клиентов и получаем необходимое кол-во ТС для данных маршрутов
    current_route = del_null_clients(current_route)
    print(f"новый список маршрутов: {current_route}")
    return current_route
#width, legth, height, weight,

def sum_last_elements(tpl):
    total_sum = 0
    lst = list(tpl[1]) # Преобразуем кортеж в список
    for sublist in lst:
        last_element = sublist[-1]
        total_sum += last_element
    return total_sum



def convert_data_client(data):
    if isinstance(data, str):
        data = json.loads(data)
    result = []
    for item in data:
        item_id = item["id"]
        result.append(item_id)
    return result
def convert_data_demand(data):
    if isinstance(data, str):
        data = json.loads(data)
    result = {}
    for item in data:
        id_client = item["idClient"]
        weight = item["weight"]
        if id_client not in result:
            result[id_client] = []
        result[id_client].append(weight)
    return list(result.values())


def create_need(clients,cargos):
    list_need = []
    for i in range(len(cargos)):
        list_need.append((clients[i],cargos[i]))
    return list_need

def convert_data_windows(data):
    if isinstance(data, str):
        data = json.loads(data)
    open_window = []
    close_window = []
    for item in data:
        open_time = item["openWindow"].split(":")
        close_time = item["closeWindow"].split(":")
        open_window.append([int(open_time[0]), int(open_time[1])])
        close_window.append([int(close_time[0]), int(close_time[1])])
    open_window_arr = np.array(open_window)
    close_window_arr = np.array(close_window)
    return open_window_arr, close_window_arr
