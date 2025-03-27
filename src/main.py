import os.path
import random
import numpy as np
import csv
import time
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """Class for logging the progress of optimization algorithms."""

    def __init__(self, filename: str = "results", directory: str = "../results/"):
        self.data = []
        self.directory = directory
        self.filename = self._generate_filename(filename)

    def _generate_filename(self, filename: str) -> str:
        date_str = datetime.today().strftime("%Y%m%d-%H%M")
        return os.path.join(self.directory, f"{filename}_{date_str}")

    def log(self, generation: int, best_fitness: float, avg_fitness: float, worst_fitness: float):
        """Log statistics for a generation."""
        self.data.append((generation, best_fitness, avg_fitness, worst_fitness))

    def save(self):
        """Save logged data to a CSV file."""
        with open(f"{self.filename}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['generation', 'best_fitness', 'avg_fitness', 'worst_fitness'])
            for row in self.data:
                writer.writerow(row)

    def plot(self, title: str = "Algorithm Performance"):
        """Plot the performance of the algorithm."""
        generations = [row[0] for row in self.data]
        best_fitness = [row[1] for row in self.data]
        avg_fitness = [row[2] for row in self.data]
        worst_fitness = [row[3] for row in self.data]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label='Best Fitness')
        plt.plot(generations, avg_fitness, label='Average Fitness')
        plt.plot(generations, worst_fitness, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.filename)
        plt.close()


@dataclass
class Node:
    """Class representing a node (location) in the CVRP."""
    id: int
    x: float
    y: float
    demand: float
    is_depot: bool = False


class CVRPProblem:
    """Class representing a CVRP instance."""

    def __init__(self):
        self.name = ""
        self.comment = ""
        self.dimension = 0
        self.edge_weight_type = ""
        self.capacity = 0
        self.nodes = []
        self.depot_index = 0
        self.distance_matrix = None

    def load_from_file(self, filename: str):
        """Load CVRP instance from a file."""
        with open(filename, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("NAME"):
                self.name = line.split(':')[1].strip()
            elif line.startswith("COMMENT"):
                self.comment = line.split(':')[1].strip()
            elif line.startswith("TYPE"):
                # Verify this is a CVRP instance
                problem_type = line.split(':')[1].strip()
                if problem_type != "CVRP":
                    raise ValueError(f"Expected CVRP, got {problem_type}")
            elif line.startswith("DIMENSION"):
                self.dimension = int(line.split(':')[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                self.edge_weight_type = line.split(':')[1].strip()
            elif line.startswith("CAPACITY"):
                self.capacity = int(line.split(':')[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                i += 1
                for _ in range(self.dimension):
                    node_data = lines[i].strip().split()
                    node_id = int(node_data[0])
                    x = float(node_data[1])
                    y = float(node_data[2])
                    self.nodes.append(Node(node_id, x, y, 0))
                    i += 1
                i -= 1  # Adjust for the end of the loop increment
            elif line.startswith("DEMAND_SECTION"):
                i += 1
                for _ in range(self.dimension):
                    demand_data = lines[i].strip().split()
                    node_id = int(demand_data[0])
                    demand = float(demand_data[1])
                    # Node IDs in the file are 1-indexed, but our list is 0-indexed
                    self.nodes[node_id - 1].demand = demand
                    if demand == 0:
                        self.nodes[node_id - 1].is_depot = True
                        self.depot_index = node_id - 1
                    i += 1
                i -= 1  # Adjust for the end of the loop increment
            elif line.startswith("DEPOT_SECTION"):
                i += 1
                depot_id = int(lines[i].strip())
                if depot_id != -1:
                    self.depot_index = depot_id - 1  # Convert from 1-indexed to 0-indexed
            i += 1

        # Calculate distance matrix
        self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """Calculate the distance matrix for the nodes."""
        n = len(self.nodes)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Euclidean distance
                    dx = self.nodes[i].x - self.nodes[j].x
                    dy = self.nodes[i].y - self.nodes[j].y
                    self.distance_matrix[i, j] = round(np.sqrt(dx * dx + dy * dy))

    def get_distance(self, i: int, j: int) -> float:
        """Get the distance between nodes i and j."""
        return self.distance_matrix[i, j]

    def get_total_distance(self, route: List[int]) -> float:
        """Calculate the total distance of a route."""
        total = 0
        for i in range(len(route) - 1):
            total += self.get_distance(route[i], route[i + 1])
        return total

    def is_valid_route(self, route: List[int]) -> bool:
        """Check if a route is valid (respects capacity constraints)."""
        if not route:
            return True

        total_demand = 0
        for node_idx in route:
            if node_idx != self.depot_index:
                total_demand += self.nodes[node_idx].demand

        return total_demand <= self.capacity


class CVRPSolution:
    """Class representing a solution to a CVRP instance."""

    def __init__(self, problem: CVRPProblem):
        self.problem = problem
        self.routes = []  # List of routes, each route is a list of node indices
        self.fitness = float('inf')

    def copy(self):
        """Create a deep copy of the solution."""
        new_solution = CVRPSolution(self.problem)
        new_solution.routes = [route.copy() for route in self.routes]
        new_solution.fitness = self.fitness
        return new_solution

    def evaluate(self):
        """Evaluate the fitness (total distance) of the solution."""
        total_distance = 0
        depot_idx = self.problem.depot_index

        for route in self.routes:
            # Add depot at the start and end of each route if not already there
            if route[0] != depot_idx:
                route.insert(0, depot_idx)
            if route[-1] != depot_idx:
                route.append(depot_idx)

            total_distance += self.problem.get_total_distance(route)

        self.fitness = total_distance
        return self.fitness

    def is_valid(self) -> bool:
        """Check if the solution is valid."""
        # Check that all routes are valid
        for route in self.routes:
            if not self.problem.is_valid_route(route):
                return False

        # Check that all nodes (except depot) are visited exactly once
        visited_nodes = set()
        for route in self.routes:
            for node_idx in route:
                if node_idx != self.problem.depot_index:
                    if node_idx in visited_nodes:
                        return False
                    visited_nodes.add(node_idx)

        # Check that all nodes (except depot) are visited
        all_nodes = set(range(len(self.problem.nodes)))
        all_nodes.remove(self.problem.depot_index)
        if visited_nodes != all_nodes:
            return False

        return True

    def display(self):
        """Display the solution."""
        print(f"Total distance: {self.fitness}")
        for i, route in enumerate(self.routes):
            print(f"Route #{i + 1}: {' '.join(map(str, [idx + 1 for idx in route]))}")


class RandomSolver:
    """Class for generating random solutions to a CVRP instance."""

    def __init__(self, problem: CVRPProblem):
        self.problem = problem

    def solve(self) -> CVRPSolution:
        """Generate a random solution to the CVRP instance."""
        solution = CVRPSolution(self.problem)

        # Get all non-depot nodes
        nodes = [i for i in range(len(self.problem.nodes)) if i != self.problem.depot_index]
        random.shuffle(nodes)

        depot_idx = self.problem.depot_index
        current_route = []
        current_load = 0

        for node_idx in nodes:
            node_demand = self.problem.nodes[node_idx].demand

            # If adding this node would exceed capacity, start a new route
            if current_load + node_demand > self.problem.capacity:
                if current_route:
                    # Add depot at the start and end
                    current_route.insert(0, depot_idx)
                    current_route.append(depot_idx)
                    solution.routes.append(current_route)
                current_route = [node_idx]
                current_load = node_demand
            else:
                current_route.append(node_idx)
                current_load += node_demand

        # Add the last route if not empty
        if current_route:
            current_route.insert(0, depot_idx)
            current_route.append(depot_idx)
            solution.routes.append(current_route)

        solution.evaluate()
        return solution


class GreedySolver:
    """Class for generating greedy solutions to a CVRP instance."""

    def __init__(self, problem: CVRPProblem):
        self.problem = problem

    def solve(self, start_node: int = None) -> CVRPSolution:
        """Generate a greedy solution to the CVRP instance."""
        if start_node is None:
            start_node = self.problem.depot_index

        solution = CVRPSolution(self.problem)

        # Get all non-depot nodes
        unvisited = set(range(len(self.problem.nodes)))
        unvisited.remove(self.problem.depot_index)

        depot_idx = self.problem.depot_index

        while unvisited:
            current_route = [depot_idx]
            current_load = 0

            while unvisited:
                # Find the closest unvisited node
                current_node = current_route[-1]
                min_distance = float('inf')
                next_node = None

                for node in unvisited:
                    if current_load + self.problem.nodes[node].demand <= self.problem.capacity:
                        distance = self.problem.get_distance(current_node, node)
                        if distance < min_distance:
                            min_distance = distance
                            next_node = node

                if next_node is None:
                    # No more nodes can be added to this route
                    break

                current_route.append(next_node)
                current_load += self.problem.nodes[next_node].demand
                unvisited.remove(next_node)

            # Complete the route by returning to the depot
            current_route.append(depot_idx)
            solution.routes.append(current_route)

        solution.evaluate()
        return solution


class GeneticAlgorithm:
    """Class implementing a Genetic Algorithm for CVRP."""

    def __init__(
            self,
            problem: CVRPProblem,
            population_size: int = 100,
            crossover_rate: float = 0.7,
            mutation_rate: float = 0.1,
            max_generations: int = 500,
            tour_n: int = 5,
            elitism: int = 2
    ):
        self.problem = problem
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.tour_n = tour_n
        self.elitism = elitism
        self.logger = Logger(f"{self.problem.name}_genetic_results")
        self.population = []

    def initialize_population(self):
        """Initialize population using random and greedy solutions."""
        self.population = []
        # Use 50% random and 50% greedy solutions for diversity
        for _ in range(self.population_size // 2):
            solver = RandomSolver(self.problem)
            self.population.append(solver.solve())
        for _ in range(self.population_size // 2):
            solver = GreedySolver(self.problem)
            self.population.append(solver.solve())

    def tournament_selection(self) -> CVRPSolution:
        """Select a parent using tournament selection."""
        participants = random.sample(self.population, self.tour_n)
        return min(participants, key=lambda x: x.fitness)

    def ordered_crossover(self, parent1: CVRPSolution, parent2: CVRPSolution) -> CVRPSolution:
        """Ordered Crossover (OX) adapted for CVRP."""
        # Flatten routes and remove depots
        flat_parent1 = [node for route in parent1.routes for node in route if node != self.problem.depot_index]
        flat_parent2 = [node for route in parent2.routes for node in route if node != self.problem.depot_index]

        # Perform OX
        size = len(flat_parent1)
        start, end = sorted(random.sample(range(size), 2))
        child_sequence = flat_parent1[start:end]
        remaining = [node for node in flat_parent2 if node not in child_sequence]
        child_sequence += remaining

        # Rebuild routes respecting capacity
        child = CVRPSolution(self.problem)
        current_route = []
        current_load = 0

        for node in child_sequence:
            demand = self.problem.nodes[node].demand
            if current_load + demand > self.problem.capacity:
                child.routes.append([self.problem.depot_index] + current_route + [self.problem.depot_index])
                current_route = [node]
                current_load = demand
            else:
                current_route.append(node)
                current_load += demand

        if current_route:
            child.routes.append([self.problem.depot_index] + current_route + [self.problem.depot_index])

        child.evaluate()
        return child

    def mutate(self, solution: CVRPSolution) -> CVRPSolution:
        """Swap mutation between two nodes in the same route."""
        mutated = solution.copy()
        route_idx = random.randint(0, len(mutated.routes) - 1)
        route = mutated.routes[route_idx]

        if len(route) > 3:  # At least one node between depots
            i, j = random.sample(range(1, len(route) - 1), 2)
            route[i], route[j] = route[j], route[i]

        mutated.evaluate()
        return mutated

    def evolve(self):
        """Run the genetic algorithm."""
        self.initialize_population()

        for generation in range(self.max_generations):
            # Evaluate population
            for solution in self.population:
                if solution.fitness == float('inf'):
                    solution.evaluate()

            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness)

            # Log statistics
            best_fitness = self.population[0].fitness
            avg_fitness = sum(s.fitness for s in self.population) / len(self.population)
            worst_fitness = self.population[-1].fitness
            self.logger.log(generation, best_fitness, avg_fitness, worst_fitness)

            # Create next generation
            new_population = []

            # Elitism: keep the best solutions
            new_population.extend(self.population[:self.elitism])

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                if random.random() < self.crossover_rate:
                    child = self.ordered_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            self.population = new_population

        self.logger.save()
        self.logger.plot("Genetic Algorithm Performance")
        return min(self.population, key=lambda x: x.fitness)


class TabuSearch:
    """Class implementing Tabu Search for CVRP."""

    def __init__(self, problem: CVRPProblem, tabu_tenure: int = 20, max_iterations: int = 1000):
        self.problem = problem
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.best_solution = None
        self.logger = Logger(f"{self.problem.name}_tabu_search_results")

    def get_neighbors(self, solution: CVRPSolution) -> List[Tuple[CVRPSolution, Tuple]]:
        """Generate neighbors of a solution using swap and relocate moves."""
        neighbors = []

        # Swap move: exchange two nodes between routes
        for i in range(len(solution.routes)):
            for j in range(i, len(solution.routes)):
                route1 = solution.routes[i]
                route2 = solution.routes[j]

                for pos1 in range(1, len(route1) - 1):  # Skip depot at start and end
                    for pos2 in range(1, len(route2) - 1):
                        if i == j and abs(pos1 - pos2) <= 1:
                            continue  # Skip adjacent nodes in the same route

                        new_solution = solution.copy()
                        # Swap the nodes
                        node1 = new_solution.routes[i][pos1]
                        node2 = new_solution.routes[j][pos2]

                        new_solution.routes[i][pos1] = node2
                        new_solution.routes[j][pos2] = node1

                        # Check if the new solution is valid
                        route1_valid = self.problem.is_valid_route(
                            [n for n in new_solution.routes[i] if n != self.problem.depot_index])
                        route2_valid = self.problem.is_valid_route(
                            [n for n in new_solution.routes[j] if n != self.problem.depot_index])

                        if route1_valid and route2_valid:
                            new_solution.evaluate()
                            move = ("swap", i, pos1, j, pos2)
                            neighbors.append((new_solution, move))

        # Relocate move: move a node from one route to another
        for i in range(len(solution.routes)):
            for j in range(len(solution.routes)):
                if i == j:
                    continue

                route1 = solution.routes[i]
                route2 = solution.routes[j]

                for pos1 in range(1, len(route1) - 1):  # Skip depot at start and end
                    for pos2 in range(1, len(route2)):
                        new_solution = solution.copy()
                        # Remove node from route1 and insert into route2
                        node = new_solution.routes[i][pos1]
                        new_solution.routes[i].pop(pos1)
                        new_solution.routes[j].insert(pos2, node)

                        # Check if the new solution is valid
                        route1_valid = new_solution.routes[i] and self.problem.is_valid_route(
                            [n for n in new_solution.routes[i] if n != self.problem.depot_index])
                        route2_valid = self.problem.is_valid_route(
                            [n for n in new_solution.routes[j] if n != self.problem.depot_index])

                        if route1_valid and route2_valid:
                            new_solution.evaluate()
                            move = ("relocate", i, pos1, j, pos2)
                            neighbors.append((new_solution, move))

        return neighbors

    def solve(self, initial_solution: CVRPSolution = None) -> CVRPSolution:
        """Solve the CVRP using Tabu Search."""
        if initial_solution is None:
            # Generate an initial solution using the greedy algorithm
            greedy_solver = GreedySolver(self.problem)
            current_solution = greedy_solver.solve()
        else:
            current_solution = initial_solution.copy()

        current_solution.evaluate()
        self.best_solution = current_solution.copy()

        # Tabu list to store forbidden moves
        tabu_list = {}  # Move -> Tabu tenure expiration

        # For statistics logging
        iteration_best = []
        iteration_current = []

        for iteration in range(self.max_iterations):
            # Generate neighbors
            neighbors = self.get_neighbors(current_solution)

            if not neighbors:
                break

            # Find the best non-tabu neighbor or the best neighbor that improves the best solution
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_move = None

            for neighbor, move in neighbors:
                # Check if the move is tabu
                is_tabu = move in tabu_list and tabu_list[move] > iteration

                # Accept the move if it's not tabu or if it improves the best solution
                if (not is_tabu) or neighbor.fitness < self.best_solution.fitness:
                    if neighbor.fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = neighbor.fitness
                        best_move = move

            if best_neighbor is None:
                break

            # Update the current solution
            current_solution = best_neighbor

            # Update the best solution if needed
            if current_solution.fitness < self.best_solution.fitness:
                self.best_solution = current_solution.copy()

            # Add the move to the tabu list
            tabu_list[best_move] = iteration + self.tabu_tenure

            # Log statistics
            iteration_best.append(self.best_solution.fitness)
            iteration_current.append(current_solution.fitness)

            # For plotting purposes, log every 10 iterations
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                avg_fitness = sum(iteration_current[-10:]) / min(10, len(iteration_current))
                worst_fitness = max(iteration_current[-10:])
                self.logger.log(iteration, self.best_solution.fitness, avg_fitness, worst_fitness)

        self.logger.save()
        self.logger.plot("Tabu Search Performance")

        return self.best_solution


def main():
    # Load the problem
    problem = CVRPProblem()
    problem.load_from_file(os.path.join("..", "data", "A-n32-k5.vrp"))

    print("Problem loaded:")
    print(f"Name: {problem.name}")
    print(f"Dimension: {problem.dimension}")
    print(f"Capacity: {problem.capacity}")
    print(f"Depot: {problem.depot_index + 1}")

    # Solve with Genetic Algorithm
    start_time = time.time()
    ga = GeneticAlgorithm(problem,
                          population_size=100,
                          max_generations=200,
                          crossover_rate=0.6,
                          mutation_rate=0.3,
                          tour_n=5)
    ga_solution = ga.evolve()
    ga_time = time.time() - start_time

    print("\nGenetic Algorithm Solution:")
    ga_solution.display()
    print(f"Time: {ga_time:.2f} seconds")

    # Solve with Tabu Search
    start_time = time.time()
    tabu_search = TabuSearch(problem, tabu_tenure=20, max_iterations=200)
    tabu_solution = tabu_search.solve()
    tabu_time = time.time() - start_time

    print("\nTabu Search Solution:")
    tabu_solution.display()
    print(f"Time: {tabu_time:.2f} seconds")

    # Compare with Random and Greedy solutions
    random_solver = RandomSolver(problem)
    greedy_solver = GreedySolver(problem)

    # Random solutions - run multiple times
    random_results = []
    for _ in range(10**4):
        start_time = time.time()
        random_solution = random_solver.solve()
        random_time = time.time() - start_time
        random_results.append((random_solution.fitness, random_time))

    random_best = min(random_results, key=lambda x: x[0])
    random_worst = max(random_results, key=lambda x: x[0])
    random_avg = sum(x[0] for x in random_results) / len(random_results)
    random_std = (sum((x[0] - random_avg) ** 2 for x in random_results) / len(random_results)) ** 0.5

    # Greedy solutions - run from different starting points
    greedy_results = []
    for start_node in range(problem.dimension):
        start_time = time.time()
        greedy_solution = greedy_solver.solve(start_node)
        greedy_time = time.time() - start_time
        greedy_results.append((greedy_solution.fitness, greedy_time))

    greedy_best = min(greedy_results, key=lambda x: x[0])
    greedy_worst = max(greedy_results, key=lambda x: x[0])
    greedy_avg = sum(x[0] for x in greedy_results) / len(greedy_results)
    greedy_std = (sum((x[0] - greedy_avg) ** 2 for x in greedy_results) / len(greedy_results)) ** 0.5

    # Print comparison
    print("\nComparison:")
    print("Random Solver:")
    print(f"  Best: {random_best[0]}")
    print(f"  Worst: {random_worst[0]}")
    print(f"  Average: {random_avg:.2f}")
    print(f"  Std Dev: {random_std:.2f}")
    print(f"  Average Time: {sum(x[1] for x in random_results) / len(random_results):.4f} seconds")

    print("Greedy Solver:")
    print(f"  Best: {greedy_best[0]}")
    print(f"  Worst: {greedy_worst[0]}")
    print(f"  Average: {greedy_avg:.2f}")
    print(f"  Std Dev: {greedy_std:.2f}")
    print(f"  Average Time: {sum(x[1] for x in greedy_results) / len(greedy_results):.4f} seconds")

    print("Genetic Algorithm:")
    print(f"  Best: {ga_solution.fitness}")
    print(f"  Time: {ga_time:.4f} seconds")

    print("Tabu Search:")
    print(f"  Best: {tabu_solution.fitness}")
    print(f"  Time: {tabu_time:.4f} seconds")


if __name__ == "__main__":
    main()