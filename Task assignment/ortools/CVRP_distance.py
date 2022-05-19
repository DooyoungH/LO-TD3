"""Capacitied Vehicles Routing Problem (CVRP)."""

'''VRP는 모든 차량 중 가장 이동경로가 긴 차량의 경로를 최소화 .'''

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import math

delivery_points = [ [0, 0], [1, 2], [3, 4], [5, 6] ]

def distance_generator(delivery_points):
    distance_array = []

    for i in range(len(delivery_points)):
        distances = []
        for j in range(len(delivery_points)):
            distance = euclidean(delivery_points[i], delivery_points[j])
            distances.append(distance)

        distance_array.append(distances)

    return distance_array

def euclidean(point1, point2):
    return math.sqrt( pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2) + pow(point1[2] - point2[2], 2) )


def create_data_model():
    ''' Stores the data for the problem.'''
    data = {} #Dictionary
    # 자기자신 + 각 노드까지의 거리
    data['distance_matrix'] = distance_generator(delivery_points)

    # Demands  = 각 위치는 어떤 양에 대해서 수요를 갖음 (아이템의 무게 및 볼륨)
    # Capacities = 각 차량이 보유할수 있는 최대 무게

    # Delivery order 의 개수리
    data['demands'] = [2, 3, 1, 2]
    data['vehicle_capacities'] = [5, 5, 5, 5]
    data['num_vehicles'] = 4
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))



def main():
    """Solve the CVRP problem"""
    data = create_data_model()
    print(data)

    # Create the routing index manager.

    # pywrapcp.RoutingIndexManager 함수는 인덱스를 관리하는데
    # (1) depot+방문지 개수 (depot+방문지를 list로 따졌을때 인덱스 수가 됨) (2) 차량 대수 (3) depot 에 해당하는 인덱스임
    # 각 요소의 길이 [17, 4, 0]
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

    # Create and Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    # distance callback 함수는 특정 지점의 index 를 넣어주면 실제 두점 사이의 거리를 output 으로 내보내주는 함수임
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        # IndexToNode 함수는 index 에서 노드로 변환시켜주는건데, 이건 자세하게 모르겠네

        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc. (acr 코스트, cost of travel 도 정의해준데 아마 전체의 거리인듯?)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)

    # 이 routing solver 는 'dimension' 이라는 오브젝트를 필요로 하는데,
    # distance dimension 은 각 차량이 이동한 경로를 따라 누적되는 거리/시간을 계산하는데 필요함.
    # 라우팅 프로그램은 이 dimension 을 사용하여 차량 경로에 누적되는 수량을 추적하는 (아마 수량 제한이 있는 경우일듯?)
    # CVRP 의 경우 드론의 load 가 넘치지 않게끔 제약을 거는데 dimension 을 사용.

    # AddDimension 이라는 함수는 5개의 input 을 포함하는데,
    # callback_index: 수량을 돌려주는 callback 의 인덱스를 의미, damand의 callback 함수를 인자로 사용하는데 누적 load 가 차량 용량을 넘지 않도록 해줌
    # slack_max: 특정 장소에서의 대기시간의 maximum 값, scheduling 에 관련하지 않으므로 0으로 설정
    # capacity: 차량의 용량, heterogenious 형태로 가능한 것으로 보임
    # fix_start_cumul_to_zero: (Boolean) 시작값이 0인지 혹은 다른 초기값이 있는지를 판단.
    # Dimension_name: 나중에 접근할 수 있도록 dimension에 이름을 붙임
    # CVRP 의 경우 드론의 load 가 넘치지 않게끔 제약을 거는데 dimension 을 사용. => Capacity


    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')



    # 차량 용량의 관점에서 보면 차량이 i 에서 j 로 이동할때 누적된 load의 변화는 i에서의 수요와 같다 라고 설명하고 있는데..
    # depot(0) -> Load(8), 여기서 load 의 변화는 i 에서의 수요와 같음 (?? 이거 설명을 모르겠네)

    # Dimension 오브젝트는 차량의 route 를 따라 이동함에 따라 '누적되는 수량' 에 관련한 두가지 타입의 변수를 저장하는데
    # Transit variables: 방문지 i 에서 j의 이동이 하나의 step 이라고 할 때, 각 스텁에서 증가하거나 감소하는 수량의 숫자
    # Cumulative variables: 각 장소에서 누적된 수량.

    #slack(i) = cumul(j) - cumul(i) - transit(i, j) 라고 한다면
    # 다음장소에서 누적된 수량,현재 장소에서 누적된 수량 - 그 스탭에서의 차이는 당연히 0이 되는게 맞음

    #Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    print(solution)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)


if __name__ == '__main__':
    main()