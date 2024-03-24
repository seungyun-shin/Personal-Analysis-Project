from solution import Solution
import itertools
import bisect
import random


class Ant:

    def __init__(self, alpha, beta):
        self.alpha = alpha # 페로몬 영향
        self.beta = beta # 엣지 가중치 영향
        self.sales = None # 경로 수
        self.graph = None
        self.n = None # 노드 수

    # 솔루션 생성 역할
    def tour(self, graph, sales, start, opt2):
        self.graph = graph
        self.sales = sales
        self.n = len(graph.nodes)

        solutions = [Solution(graph, start, self) for _ in range(sales)]

        # as possible,  salesman travels vertexes as same
        # 그래프를 세그먼트로 나누고 각 개미가 세그먼트를 횡단하도록 할당 (균등하게)
        saleses = [(self.n - 1) // sales for i in range(sales)]
        for i in range((self.n - 1) % sales):
            saleses[i] += 1

        unvisited = [i for i in range(1, self.n + 1) if i != start]
        for i in range(sales):
            for j in range(saleses[i]):
                next_node = self.choose_destination(solutions[i].current, unvisited)
                solutions[i].add_node(next_node)
                unvisited.remove(next_node)
            solutions[i].close()

        if opt2:
            self.opt2_update(graph, opt2, sales, saleses, solutions)

        return solutions

    # 솔루션을 최적화하기 위해 2-opt 휴리스틱을 구현
    def opt2_update(self, graph, opt2, sales, saleses, solutions):
        # 경로의 솔루션 반복
        for i in range(sales):
            # 각 솔루션 내에서 2-opt 최적화 반복
            for j in range(opt2):
                k = saleses[i] + 1
                # 두 개의 가장자리를 무작위로 선택하고 이를 교체하면 솔루션이 개선되는지 확인
                while True:
                    a = random.randint(0, k - 1)
                    b = random.randint(0, k - 1)
                    if a != b:
                        break
                # 교환을 통해 솔루션이 개선되면 교환이 적용되고 솔루션 비용이 업데이트됩니다.
                if a > b:
                    a, b = b, a
                dist_a = graph.edges[solutions[i].nodes[a], solutions[i].nodes[a + 1]]['weight']
                dist_b = graph.edges[solutions[i].nodes[b], solutions[i].nodes[(b + 1) % k]]['weight']
                dist_c = graph.edges[solutions[i].nodes[a], solutions[i].nodes[b]]['weight']
                dist_d = graph.edges[solutions[i].nodes[a + 1], solutions[i].nodes[(b + 1) % k]]['weight']
                if dist_a + dist_b > dist_c + dist_d:
                    solutions[i].nodes[a + 1:b + 1] = reversed(solutions[i].nodes[a + 1: b + 1])
                    solutions[i].cost += (dist_c + dist_d - dist_a - dist_b)
                    solutions[i].path = []
                    for l in range(k):
                        solutions[i].path.append((solutions[i].nodes[l], solutions[i].nodes[(l + 1) % k]))

    def choose_destination(self, current, unvisited):
        # 현재 도시와 방문하지 않은 도시 목록을 고려
        if len(unvisited) == 1:
            return unvisited[0]
        # 현재 도시와 해당 도시를 연결하는 Edge의 바람직성을 기준으로 방문하지 않은 각 도시에 대한 점수를 계산
        scores = self.get_scores(current, unvisited)
        return self.choose_node(unvisited, scores)

    # 점수에 의해 결정된 확률 분포를 기반으로 이 방법은 개미가 방문할 다음 도시를 선택
    def choose_node(self, unvisited, scores):
        total = sum(scores)
        # 방문하지 않은 도시의 점수를 기준으로 확률분포를 계산
        cumdist = list(itertools.accumulate(scores))
        # 다음 도시를 선택하기 위해 이러한 확률에 가중치를 부여한 무작위 선택을 사용
        index = bisect.bisect(cumdist, random.random() * total)
        return unvisited[min(index, len(unvisited) - 1)]
    # 현재 도시와 잠재적인 다음 도시 사이의 바람직한 가장자리를 기반으로 가능한 각 다음 도시에 대한 점수를 계산
    def get_scores(self, current, unvisited):
        scores = []
        # 엣지의 페로몬 수준과 엣지 무게의 역수 같은 요소를 고려
        for node in unvisited:
            edge = self.graph.edges[current, node]
            score = self.score_edge(edge)
            scores.append(score)
        return scores
# 메서드는 페로몬 수준(phe)과 에지 가중치의 역수를 기반으로 에지의 점수를 계산
    def score_edge(self, edge):
        weight = edge.get('weight', 1)
        if weight == 0:
            return 1e200
        phe = edge['pheromone']
        return phe ** self.alpha * (1 / weight) ** self.beta

def optimize_route(path, coordinates, num_rows):
    optimized_path = [path[0]]  # 출발 지점 추가
    remaining_coordinates = coordinates.copy()
    remaining_coordinates.remove(path[0])
    current_position = path[0]
    count = 0
    min_distance_sum = float('inf')
    best_path = None
    
    while remaining_coordinates:
        min_distance_sum = float('inf')
        best_next_coordinate = None
        for next_coordinate in remaining_coordinates:
            count += 1
            temp_path = optimized_path + [next_coordinate]
            distance_sum = calculate_distance_sum(temp_path, num_rows)
            if distance_sum < min_distance_sum:
                min_distance_sum = distance_sum
                best_next_coordinate = next_coordinate
        optimized_path.append(best_next_coordinate)
        remaining_coordinates.remove(best_next_coordinate)
        current_position = best_next_coordinate
        print(f"Current position {count}: {current_position}, Distance: {min_distance_sum}") 
        
        # 현재 경로가 더 좋은 경우 최적 경로 업데이트
        if min_distance_sum < calculate_distance_sum(best_path, num_rows):
            best_path = optimized_path.copy()
    
    print(f"Best path: {best_path}, Distance: {calculate_distance_sum(best_path, num_rows)}")
    
    return best_path

def calculate_distance_sum(path, num_rows):
    distance_sum = 0
    for i in range(len(path) - 1):
        distance_sum += calculate_distance([path[i], path[i + 1]], num_rows)
    return distance_sum