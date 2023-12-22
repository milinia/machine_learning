import random
import copy

def start_population():
    first_generation = []
    for i in range(5):
        free_indexes = [1, 2, 3, 4, 5, 6]
        generation = []
        for i in range(6):
            random_number = random.randint(0, len(free_indexes) - 1)
            generation.append(free_indexes[random_number])
            free_indexes.pop(random_number)
        first_generation.append(generation)
    return first_generation

def crossover(parents):
    offspring = []
    offspring.append(make_child(parents[0], parents[1]))
    offspring.append(make_child(parents[1], parents[0]))
    offspring.append(make_child(parents[2], parents[3]))
    offspring.append(make_child(parents[3], parents[2]))
    offspring.append(make_child(parents[1], parents[3]))
    offspring.append(make_child(parents[3], parents[1]))
    return offspring

def make_child(parent1, parent2):
    offspring = copy.deepcopy(parent1)
    element_set = set()
    for n in range(0, 3):
        element_set.add(parent1[n])
    for k in range(3, 6):
        for value in parent2:
            if value not in element_set:
                offspring[k] = value
                element_set.add(value)
                break
    return offspring

def mutation(generations):
    number_of_mutations = random.randint(0, len(generations))
    print("Всего будет", number_of_mutations, "мутаций:")
    taken_indexes = set()
    for i in range(number_of_mutations):
        number_of_generation = random.randint(1, len(generations))
        while number_of_generation in taken_indexes:
            number_of_generation = random.randint(1, len(generations))
        generation = generations[number_of_generation - 1]
        taken_indexes.add(number_of_generation)
        first_index = random.randint(0, len(generation) - 1)
        second_index = random.randint(0, len(generation) - 1)
        while first_index == second_index:
            first_index = random.randint(0, len(generation) - 1)
            second_index = random.randint(0, len(generation) - 1)
        print("     Мутирование произошло в", number_of_generation, "гене")
        temp = generation[first_index]
        generation[first_index] = generation[second_index]
        generation[second_index] = temp
    return generations

def selection(path_cost, generations):
    selected_generations = []
    total_cost = sum(path_cost)
    costs = list(enumerate(path_cost, start=0))
    for i in range(4):
        selected_index = roulette_wheel_selection(total_cost, costs)
        selected_generations.append(generations[costs[selected_index][0]])
        costs.pop(selected_index)
    return selected_generations

def roulette_wheel_selection(total_cost, path_cost):
    probabilities = [(cost[1]) / total_cost for cost in path_cost]
    swapped_probabilities = [1.0 - prob for prob in probabilities]
    sum_swapped_probabilities = sum(swapped_probabilities)
    normalized_probabilities = [prob / sum_swapped_probabilities for prob in swapped_probabilities]
    r = random.uniform(0, 1)
    cumulative_prob = 0.0

    for i, prob in enumerate(normalized_probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return i

def evaluation(generations, distance_matrix):
    path_costs = []
    for i in range(len(generations)):
        generation = generations[i]
        path_cost = 0
        for i in range(len(generation) - 1):
            path_cost += distance_matrix[generation[i] - 1][generation[i + 1] - 1]
        path_cost += distance_matrix[generation[len(generation) - 1] - 1][generation[0] - 1]
        path_costs.append(path_cost)
    return path_costs

if __name__ == '__main__':
    # 6 городов
    distance_matrix = [
        [0, 10, 15, 20, 25, 30],
        [10, 0, 35, 25, 30, 15],
        [15, 35, 0, 30, 20, 10],
        [20, 25, 30, 0, 40, 5],
        [25, 30, 20, 40, 0, 10],
        [30, 15, 10, 5, 10, 0]
    ]
    first_generation = start_population()
    print("Первое поколение - ", first_generation)
    max_generations = 1000
    for i in range(max_generations):
        path_costs = evaluation(first_generation, distance_matrix)
        print("Приспособленность особей - ", path_costs)
        parents = selection(path_costs, first_generation)
        print("Выбранные особи - ", parents)
        children = crossover(parents)
        print("Полученное новое поколение из родителей - ", children)
        print("Мутация началась")
        first_generation = mutation(children)
        print("Последствия мутации", first_generation)
        print("-------------------------------------------------------------------")
    print("РЕЗУЛЬТАТ", first_generation)
    path_costs = evaluation(first_generation, distance_matrix)
    print("Приспособленность особей - ", path_costs)