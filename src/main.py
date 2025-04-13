import os
import os.path
import random
import numpy as np
import csv
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import statistics


class Logger:
    """Class for logging the progress of optimization algorithms."""

    # ZMIANA: Konstruktor przyjmuje teraz docelowy katalog (folder serii)
    def __init__(self, directory: str, filename_base: str = "log_data"):
        self.data = []
        self.directory = directory # Przechowuje katalog serii
        self.filename_base = filename_base

    def log(self, x_value: int, best_fitness: float, *args):
        """Log statistics for a generation or iteration."""
        self.data.append((x_value, best_fitness) + args)

    def save(self, run_identifier: str):
        """
        Saves logged data to a CSV file inside the directory provided during initialization.
        The filename is based on the run_identifier.

        Args:
            run_identifier (str): Identifier for the run (e.g., "run_1", "run_final").
                                  '.csv' extension will be added.
        """
        filename = f"{run_identifier}.csv"
        filepath = os.path.join(self.directory, filename)

        try:
            # Upewnij się, że folder istnieje (choć main powinien to zrobić)
            os.makedirs(self.directory, exist_ok=True)

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                header = ['x_value', 'best_fitness']
                if self.data and len(self.data[0]) > 2:
                    header.extend([f'metric_{i}' for i in range(len(self.data[0]) - 2)])
                writer.writerow(header)
                for row in self.data:
                    writer.writerow(row)
            # print(f"Log saved to {filepath}")
        except Exception as e:
            print(f"Error saving log to {filepath}: {e}")

    @staticmethod
    def plot_summary(run_histories: List[List[Tuple]],
                     filename_prefix: str,
                     directory: str = "../results/",
                     x_axis_label: str = "Step",
                     title_suffix: str = "Performance Summary"):
        """
        Generuje zbiorczy wykres dla wielu uruchomień algorytmu.

        Args:
            run_histories: Lista list. Każda wewnętrzna lista to historia jednego uruchomienia
                           z loggera (lista krotek: [x_value, best_fitness, ...]).
            filename_prefix: Podstawa nazwy pliku wyjściowego (np. "problem_ga_summary").
            directory: Katalog zapisu wykresu.
            x_axis_label: Etykieta osi X (np. "Generation", "Iteration").
            title_suffix: Dodatek do tytułu wykresu (np. "Genetic Algorithm", "Tabu Search").
        """
        if not run_histories:
            print("No run histories provided for summary plot.")
            return
        os.makedirs(directory, exist_ok=True)
        aggregated_data: Dict[int, List[float]] = {}
        max_x_value = 0
        num_runs = len(run_histories)

        # Agregacja z obsługą potencjalnych błędów indeksowania
        for run_index, run_data in enumerate(run_histories):
            if not run_data: continue
            run_processed_x = set()
            for step_data in run_data:
                try:
                    x_val = int(step_data[0])
                    best_fitness = float(step_data[1])
                    if x_val in run_processed_x: continue
                    run_processed_x.add(x_val)
                    if x_val not in aggregated_data:
                        aggregated_data[x_val] = [None] * num_runs
                    if run_index < num_runs:  # Dodatkowe zabezpieczenie
                        aggregated_data[x_val][run_index] = best_fitness
                    max_x_value = max(max_x_value, x_val)
                except (IndexError, ValueError, TypeError) as e:
                    print(f"Warning: Skipping invalid data point {step_data} in run {run_index}. Error: {e}")
                    continue  # Pomiń błędny punkt

        # Przygotowanie danych do wykresu (z wypełnianiem braków)
        plot_x_values = []
        plot_best_star, plot_avg_of_bests, plot_worst_of_bests = [], [], []
        filled_aggregated_data = {}
        last_known_bests = [None] * num_runs

        for x_val in range(max_x_value + 1):
            current_bests = aggregated_data.get(x_val)  # Pobierz dane dla x_val
            # Jeśli brak danych dla x_val, użyj ostatnio znanych
            if current_bests is None:
                if x_val > 0 and x_val - 1 in filled_aggregated_data:
                    filled_bests = filled_aggregated_data[x_val - 1][:]  # Kopiuj poprzednie
                else:
                    filled_bests = [None] * num_runs  # Brak danych na początku
            else:
                # Zaktualizuj ostatnio znane i wypełnij None
                filled_bests = []
                for i in range(num_runs):
                    if i < len(current_bests) and current_bests[i] is not None:
                        last_known_bests[i] = current_bests[i]
                        filled_bests.append(current_bests[i])
                    else:
                        filled_bests.append(last_known_bests[i])  # Użyj ostatnio znanej

            filled_aggregated_data[x_val] = filled_bests

            # Oblicz statystyki, jeśli są dostępne dane
            valid_bests_at_x = [b for b in filled_bests if b is not None]
            if valid_bests_at_x:
                plot_x_values.append(x_val)
                plot_best_star.append(min(valid_bests_at_x))
                plot_avg_of_bests.append(statistics.mean(valid_bests_at_x))
                plot_worst_of_bests.append(max(valid_bests_at_x))

        # Generowanie wykresu
        if not plot_x_values:
            print("No data points collected for the summary plot after processing.")
            return
        plt.figure(figsize=(12, 7))
        plt.plot(plot_x_values, plot_best_star, label='Best Overall Fitness (min across runs)', marker='.',
                 linestyle='-', markersize=3)
        plt.plot(plot_x_values, plot_avg_of_bests, label='Average Best Fitness (mean across runs)', marker='.',
                 linestyle='-', markersize=3)
        plt.plot(plot_x_values, plot_worst_of_bests, label='Worst Best Fitness (max across runs)', marker='.',
                 linestyle='-', markersize=3)
        plt.xlabel(x_axis_label)
        plt.ylabel('Fitness (Best Solution)')
        plt.title(f'{title_suffix} ({num_runs} runs)')
        plt.legend()
        plt.grid(True)
        date_str = datetime.today().strftime("%Y%m%d-%H%M%S")
        random_suffix = random.randint(100, 999)
        filepath = os.path.join(directory, f"{filename_prefix}_{date_str}_{random_suffix}.png")
        try:
            plt.savefig(filepath)
            print(f"Summary plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving summary plot to {filepath}: {e}")
        plt.close()

    # plot() dla indywidualnego wykresu - zapisuje w folderze przekazanym do konstruktora
    def plot(self, run_identifier: str = "individual_run"):
        """Plot the performance of a single algorithm run and save it."""
        if not self.data: return
        plot_filename = f"{run_identifier}_plot.png"
        filepath = os.path.join(self.directory, plot_filename)
        try:
            x_values = [row[0] for row in self.data]
            best_fitness = [row[1] for row in self.data]
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, best_fitness, label='Best Fitness')
            plt.xlabel('Step (Generation/Iteration)')
            plt.ylabel('Fitness')
            plt.title(f"Run: {run_identifier}") # Użyj nazwy bazowej w tytule
            plt.legend()
            plt.grid(True)
            plt.savefig(filepath)
            # print(f"Individual run plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving individual plot to {filepath}: {e}")
        finally:
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
    def __init__(
            self,
            problem: CVRPProblem,
            population_size: int = 100,
            crossover_rate: float = 0.7,
            mutation_rate: float = 0.1,
            max_evaluations: int = 10000,
            tour_n: int = 5,
            elitism: int = 2,
            log_directory: str = "../results/"
    ):
        # ... (reszta konstruktora bez zmian) ...
        self.problem = problem
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_evaluations = max_evaluations
        self.tour_n = tour_n
        self.elitism = elitism
        log_filename_base = f"{self.problem.name}_ga_internal"
        self.logger = Logger(directory=log_directory, filename_base=log_filename_base)
        self.population = []
        self.evaluations_count = 0


    def initialize_population(self):
        """Initialize population using 1 greedy solution and the rest random."""
        self.population = []
        self.evaluations_count = 0  # Resetuj licznik przy inicjalizacji

        if self.population_size <= 0:
            return  # Nie rób nic, jeśli rozmiar populacji jest zerowy lub ujemny

        # 1. Dodaj jednego osobnika z GreedySolver
        print("Initializing population: Adding 1 Greedy solution...")
        try:
            greedy_solver = GreedySolver(self.problem)
            # Uruchom z domyślnego punktu startowego (depot) dla spójności
            greedy_solution = greedy_solver.solve()
            # Lub ewentualnie z losowego:
            # start_node_greedy = random.randrange(self.problem.dimension)
            # greedy_solution = greedy_solver.solve(start_node_greedy)

            if greedy_solution and greedy_solution.fitness != float('inf'):
                self.population.append(greedy_solution)
                self.evaluations_count += 1
            else:
                print("Warning: Greedy solver failed to produce a valid solution during initialization.")
                # Jeśli greedy zawiedzie, możemy potrzebować fallbacku, np. dodać losowego zamiast niego
                # lub po prostu kontynuować z mniejszą populacją początkową.
                # Na razie kontynuujemy, zakładając że Random zadziała.

        except Exception as e:
            print(f"Error during Greedy initialization: {e}")
            # Handle error, maybe add a random one instead

        # 2. Dodaj pozostałych osobników (population_size - 1) z RandomSolver
        num_random_to_add = self.population_size - len(self.population)  # Ile jeszcze brakuje

        if num_random_to_add > 0:
            print(f"Initializing population: Adding {num_random_to_add} Random solutions...")
            random_solver = RandomSolver(self.problem)
            for i in range(num_random_to_add):
                try:
                    random_solution = random_solver.solve()
                    if random_solution and random_solution.fitness != float('inf'):
                        self.population.append(random_solution)
                        self.evaluations_count += 1
                    else:
                        print(
                            f"Warning: Random solver failed to produce a valid solution ({i + 1}/{num_random_to_add}).")
                        # Można spróbować ponownie lub pominąć
                except Exception as e:
                    print(f"Error during Random initialization ({i + 1}/{num_random_to_add}): {e}")

        # Komunikat końcowy o inicjalizacji
        print(f"Population initialized with {len(self.population)} individuals ({self.evaluations_count} evaluations).")

        # Opcjonalnie: przetasuj populację początkową
        random.shuffle(self.population)

    def tournament_selection(self) -> CVRPSolution:
        """Select a parent using tournament selection."""
        participants = random.sample(self.population, self.tour_n)
        return min(participants, key=lambda x: x.fitness)

    # --- Metoda 1: Ordered Crossover (OX) ---
    def ordered_crossover(self, parent1: CVRPSolution, parent2: CVRPSolution) -> CVRPSolution:
        """Ordered Crossover (OX) adapted for CVRP."""
        # 1. Flatten routes and remove depots
        flat_parent1 = [node for route in parent1.routes for node in route if node != self.problem.depot_index]
        flat_parent2 = [node for route in parent2.routes for node in route if node != self.problem.depot_index]

        size = len(flat_parent1)
        child_sequence = [-1] * size

        # 2. Select crossover points
        start, end = sorted(random.sample(range(size), 2))

        # 3. Copy the segment from parent1 to child
        child_sequence[start:end + 1] = flat_parent1[start:end + 1]  # OX includes the end point typically

        # 4. Fill remaining positions from parent2
        parent2_idx = 0
        child_idx = 0
        # Nodes from parent2 already in the child segment
        segment_nodes = set(child_sequence[start:end + 1])

        while child_idx < size:
            # Skip over the segment copied from parent1
            if start <= child_idx <= end:
                child_idx += 1
                continue

            # Find the next node in parent2 that is not in the child's segment
            node_from_p2 = flat_parent2[parent2_idx]
            while node_from_p2 in segment_nodes:
                parent2_idx += 1
                # Handle wrap around if needed, though lists should be same size
                parent2_idx %= size
                node_from_p2 = flat_parent2[parent2_idx]

            # Place the node in the child
            if child_sequence[child_idx] == -1:  # Ensure position is empty
                child_sequence[child_idx] = node_from_p2

            parent2_idx += 1
            child_idx += 1

        # 5. Rebuild routes respecting capacity (common function or duplicated code)
        child = CVRPSolution(self.problem)
        current_route = []
        current_load = 0
        depot_idx = self.problem.depot_index

        for node in child_sequence:
            # Handle potential errors where node might be -1 if logic failed
            if node == -1:
                print("Warning: OX resulted in unassigned node (-1). Skipping.")
                continue
            demand = self.problem.nodes[node].demand
            if current_load + demand > self.problem.capacity:
                if current_route:
                    child.routes.append([depot_idx] + current_route + [depot_idx])
                current_route = [node]
                current_load = demand
            else:
                current_route.append(node)
                current_load += demand
        if current_route:
            child.routes.append([depot_idx] + current_route + [depot_idx])

        if not child.routes and size > 0:
            print("Warning: OX resulted in no routes. Generating a fallback.")
            return parent1.copy() if random.random() < 0.5 else parent2.copy()

        child.evaluate()
        # Zlicz ocenę potomka
        self.evaluations_count += 1
        return child

    # --- Koniec OX ---

    # --- Metoda 2: Partially Mapped Crossover (PMX) ---
    def pmx_crossover(self, parent1: CVRPSolution, parent2: CVRPSolution) -> CVRPSolution:
        """Partially Mapped Crossover (PMX) adapted for CVRP."""
        flat_parent1 = [node for route in parent1.routes for node in route if node != self.problem.depot_index]
        flat_parent2 = [node for route in parent2.routes for node in route if node != self.problem.depot_index]
        size = len(flat_parent1)
        if size == 0:  # Handle empty parents
            return parent1.copy()
        child_sequence = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child_sequence[start:end] = flat_parent1[start:end]
        mapping = {flat_parent1[i]: flat_parent2[i] for i in range(start, end)}

        for i in range(size):
            if start <= i < end: continue
            candidate = flat_parent2[i]
            resolved_candidate = candidate
            # Keep track of visited nodes during conflict resolution to detect cycles
            visited_in_resolve = set()
            while resolved_candidate in child_sequence[start:end] and resolved_candidate not in visited_in_resolve:
                visited_in_resolve.add(resolved_candidate)
                if resolved_candidate in mapping:
                    resolved_candidate = mapping[resolved_candidate]
                else:
                    # Try reverse mapping if direct mapping doesn't exist
                    reverse_mapped = False
                    for p1_val, p2_val in mapping.items():
                        if p2_val == resolved_candidate:
                            resolved_candidate = p1_val
                            reverse_mapped = True
                            break
                    if not reverse_mapped:
                        # Cannot resolve, potential issue or complex cycle
                        print(
                            f"Warning/Error in PMX mapping: Cannot resolve conflict for {candidate} -> {resolved_candidate}")
                        all_nodes_set = set(flat_parent1)
                        child_set = set(filter(lambda x: x != -1, child_sequence))
                        available = list(all_nodes_set - child_set - set(visited_in_resolve))
                        if available:
                            resolved_candidate = random.choice(available)
                        else:
                            resolved_candidate = -2  # Error marker
                        break

            # Check for cycle detection
            if resolved_candidate in visited_in_resolve:
                print(f"Warning: Cycle detected during PMX conflict resolution for {candidate}. Using fallback.")
                all_nodes_set = set(flat_parent1)
                child_set = set(filter(lambda x: x != -1, child_sequence))
                available = list(all_nodes_set - child_set)
                if available:
                    resolved_candidate = random.choice(available)
                else:
                    resolved_candidate = -2

            if resolved_candidate != -2 and child_sequence[i] == -1:
                child_sequence[i] = resolved_candidate

        unassigned_indices = [i for i, x in enumerate(child_sequence) if x == -1]
        if unassigned_indices:
            print(f"Warning: PMX resulted in unassigned positions: {unassigned_indices}. Filling randomly.")
            all_nodes_set = set(flat_parent1)
            child_set = set(filter(lambda x: x != -1, child_sequence))
            available_nodes = list(all_nodes_set - child_set)
            random.shuffle(available_nodes)
            fill_idx = 0
            for i in unassigned_indices:
                if fill_idx < len(available_nodes):
                    child_sequence[i] = available_nodes[fill_idx]
                    fill_idx += 1
                else:  # Should not happen
                    print(f"Error: Not enough available nodes to fill unassigned PMX positions.")
                    child_sequence[i] = -3  # Another error marker

        # Rebuild routes (common function or duplicated code)
        child = CVRPSolution(self.problem)
        current_route = []
        current_load = 0
        depot_idx = self.problem.depot_index
        for node in child_sequence:
            if node < 0:  # Skip error markers
                print(f"Warning: Skipping node {node} during route rebuilding after PMX.")
                continue
            demand = self.problem.nodes[node].demand
            if current_load + demand > self.problem.capacity:
                if current_route:
                    child.routes.append([depot_idx] + current_route + [depot_idx])
                current_route = [node]
                current_load = demand
            else:
                current_route.append(node)
                current_load += demand
        if current_route:
            child.routes.append([depot_idx] + current_route + [depot_idx])

        if not child.routes and size > 0:
            print("Warning: PMX resulted in no routes. Generating a fallback.")
            return parent1.copy() if random.random() < 0.5 else parent2.copy()

        child.evaluate()
        # Zlicz ocenę potomka
        self.evaluations_count += 1
        return child

    # --- Koniec PMX ---

    def apply_crossover(self, parent1: CVRPSolution, parent2: CVRPSolution) -> CVRPSolution:
        """Randomly selects and applies either OX or PMX crossover."""
        if random.random() < 0.5:
            # print("Applying OX Crossover") # Debug line
            return self.ordered_crossover(parent1, parent2)
        else:
            # print("Applying PMX Crossover") # Debug line
            return self.pmx_crossover(parent1, parent2)

    def swap_mutate(self, solution: CVRPSolution) -> CVRPSolution:
        """Swap mutation between two nodes in the same route."""
        mutated = solution.copy()
        if not mutated.routes: return mutated # No routes to mutate

        route_idx = random.randrange(len(mutated.routes))
        route = mutated.routes[route_idx]

        # Need at least 2 non-depot nodes to swap
        if len(route) > 3:
            # Sample two distinct indices *between* depots (indices 1 to len(route)-2)
            idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
            mutated.evaluate()
            self.evaluations_count += 1 # Zmiana: Zlicz ocenę zmutowanego osobnika
        return mutated

    def inversion_mutate(self, solution: CVRPSolution) -> CVRPSolution:
        """Inversion mutation: reverses a subsegment of a route."""
        mutated = solution.copy()
        if not mutated.routes: return mutated

        route_idx = random.randrange(len(mutated.routes))
        route = mutated.routes[route_idx]

        # Need at least 2 non-depot nodes to invert a segment of size >= 2
        if len(route) > 3:
             # Select two distinct indices between depots (1 to len(route)-2)
             idx1, idx2 = sorted(random.sample(range(1, len(route) - 1), 2))
             # Reverse the sub-list in place
             route[idx1 : idx2+1] = route[idx1 : idx2+1][::-1]
             mutated.evaluate()
             self.evaluations_count += 1 # Zmiana: Zlicz ocenę zmutowanego osobnika
        return mutated

    def mutate(self, solution: CVRPSolution) -> CVRPSolution:
        """Applies one of the available mutation operators randomly."""
        if random.random() < 0.5: # 50% chance for swap
            return self.swap_mutate(solution)
        else: # 50% chance for inversion
            return self.inversion_mutate(solution)

    def evolve(self):
        """Run the genetic algorithm."""
        self.initialize_population()

        generation = 0
        # Pętla główna oparta na liczbie ocen
        while self.evaluations_count < self.max_evaluations:
            # Ocena i sortowanie populacji
            for sol in self.population:
                if sol.fitness == float('inf'):
                    sol.evaluate()
            self.population.sort(key=lambda x: x.fitness)

            # Loguj statystyki w KAŻDEJ generacji
            if self.population:  # Upewnij się, że populacja nie jest pusta
                best_fitness = self.population[0].fitness
                avg_fitness = sum(s.fitness for s in self.population) / len(self.population)
                worst_fitness = self.population[-1].fitness
                # Logowanie: generation, best_fitness (w tej populacji), avg_fitness (w populacji), worst_fitness (w populacji)
                self.logger.log(generation, best_fitness, avg_fitness, worst_fitness)

            # print(f"Gen: {generation}, Evals: {self.evaluations_count}/{self.max_evaluations}, Best: {best_fitness:.2f}") # Opcjonalny debug

            # Warunek stopu
            if self.evaluations_count >= self.max_evaluations:
                break

            # Tworzenie nowej populacji (elitism, crossover, mutation)
            new_population = []
            new_population.extend(self.population[:self.elitism])

            while len(new_population) < self.population_size:
                if self.evaluations_count >= self.max_evaluations: break  # Sprawdź ponownie przed tworzeniem dziecka

                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child = parent1.copy()

                if random.random() < self.crossover_rate:
                    child = self.apply_crossover(parent1, parent2)  # Ocena liczona wew.

                if random.random() < self.mutation_rate:
                    child = self.mutate(child)  # Ocena liczona wew.

                new_population.append(child)

            if not new_population and self.population_size > 0:
                print(f"Warning: New population became empty at generation {generation}. Reinitializing.")
                self.initialize_population()
                continue

            self.population = new_population
            generation += 1

            # Końcowe logowanie (może być powtórzeniem ostatniego logu z pętli)
        # Można rozważyć usunięcie tego bloku, jeśli log w pętli wystarcza
        if self.population:
            best_fitness = min(s.fitness for s in self.population)
            avg_fitness = sum(s.fitness for s in self.population) / len(self.population)
            worst_fitness = max(s.fitness for s in self.population)
            # Używamy ostatniej wartości 'generation' lub 'generation+1'? Dla spójności 'generation'.
            self.logger.log(generation, best_fitness, avg_fitness, worst_fitness)
        else:
            print("Warning: Final population is empty.")
            self.logger.log(generation, float('inf'), float('inf'), float('inf'))

        # Zapis logów JEDNEGO uruchomienia (dla debugowania lub szczegółów)
        # self.logger.save() # Zapis CSV może być nadal przydatny
        # self.logger.plot(f"Genetic Algorithm ({self.problem.name}) - Run Result") # Wyłączymy domyślne rysowanie

        if self.population:
            return min(self.population, key=lambda x: x.fitness)
        else:
            print("Error: No solution found, returning empty solution.")
            return CVRPSolution(self.problem)


class TabuSearch:
    """Class implementing Tabu Search for CVRP."""

    def __init__(self,
                 problem: CVRPProblem,
                 tabu_tenure: int = 20,
                 max_iterations: int = 1000,
                 neighborhood_limit: int = None,
                 log_directory: str = "../results/"
                 ):
        self.problem = problem
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighborhood_limit = neighborhood_limit # Zmiana
        self.best_solution = None
        log_filename_base = f"{self.problem.name}_ts_internal"
        self.logger = Logger(directory=log_directory, filename_base=log_filename_base)

    def get_neighbors(self, solution: CVRPSolution) -> List[Tuple[CVRPSolution, Tuple]]:
        """Generate neighbors of a solution using swap and relocate moves."""
        neighbors = []
        depot_idx = self.problem.depot_index

        # --- Swap move ---
        # (Logika jak wcześniej, ale upewnij się, że poprawnie tworzy `move` tuple)
        for i in range(len(solution.routes)):
            for j in range(i, len(solution.routes)):  # Optymalizacja: range(i, ...)
                route1 = solution.routes[i]
                route2 = solution.routes[j]

                for pos1 in range(1, len(route1) - 1):
                    # Jeśli ta sama trasa, zacznij od pos1 + 1, aby uniknąć zamiany z samym sobą i sąsiadami
                    start_pos2 = pos1 + 1 if i == j else 1
                    for pos2 in range(start_pos2, len(route2) - 1):

                        new_solution = solution.copy()
                        node1 = new_solution.routes[i][pos1]
                        node2 = new_solution.routes[j][pos2]

                        # Swap
                        new_solution.routes[i][pos1] = node2
                        new_solution.routes[j][pos2] = node1

                        # Check validity (consider implementing a more efficient check)
                        # Re-calculate demands might be faster than full check
                        route1_valid = self.problem.is_valid_route(
                            [n for n in new_solution.routes[i] if n != depot_idx])
                        route2_valid = True if i == j else self.problem.is_valid_route(
                            [n for n in new_solution.routes[j] if n != depot_idx])

                        if route1_valid and route2_valid:
                            new_solution.evaluate()
                            # Create a canonical move representation (e.g., sort nodes involved)
                            # move = ("swap", tuple(sorted((node1, node2)))) # Example attribute-based move
                            move = ("swap", i, pos1, j, pos2, node1, node2)  # Example detailed move
                            neighbors.append((new_solution, move))

        # --- Relocate move ---
        # (Logika jak wcześniej, ale upewnij się, że poprawnie tworzy `move` tuple)
        for i in range(len(solution.routes)):
            route1 = solution.routes[i]
            if len(route1) <= 2: continue  # Cannot relocate from empty/depot-only route

            for pos1 in range(1, len(route1) - 1):  # Node to relocate
                node_to_move = route1[pos1]

                for j in range(len(solution.routes)):  # Target route
                    # if i == j: continue # Allow relocation within the same route

                    route2 = solution.routes[j]
                    # Try inserting at every possible position (including start/end for intra-route)
                    for pos2 in range(1, len(route2)):

                        # Create new solution state *efficiently* if possible
                        new_solution = solution.copy()

                        # Perform relocation
                        moved_node = new_solution.routes[i].pop(pos1)
                        new_solution.routes[j].insert(pos2, moved_node)

                        # Prune empty routes (important!)
                        if len(new_solution.routes[i]) <= 2:  # Only depots left
                            # Check if it's the *only* route left - don't delete if so
                            if len(new_solution.routes) > 1:
                                del new_solution.routes[i]
                                # Adjust target route index if it was after the deleted one
                                if j > i: j -= 1
                            else:
                                # Cannot make the only route empty, invalid move
                                continue  # Skip this neighbor

                        # Check validity (efficiently if possible)
                        # Need to check route i (if not deleted) and route j
                        route_i_valid = True
                        if i < len(new_solution.routes) and  len(new_solution.routes[i]) > 2:
                            route_i_valid = self.problem.is_valid_route(
                                [n for n in new_solution.routes[i] if n != depot_idx])

                    route_j_valid = self.problem.is_valid_route([n for n in new_solution.routes[j] if n != depot_idx])

                    if route_i_valid and route_j_valid:
                        new_solution.evaluate()
                        # move = ("relocate", node_to_move, j, pos2) # Example attribute-based
                        move = ("relocate", i, pos1, j, pos2, node_to_move)  # Example detailed move
                        neighbors.append((new_solution, move))

                        # Need to break from inner loops and recalculate neighbors
                        # if indices change due to route deletion? Or handle carefully.
                        # Current approach recalculates neighbors fully each iteration, so it's safe.

                    # Important: Since we modify routes in the copy,
                    # the simple loop structure might have issues if routes are deleted.
                    # A safer approach generates moves without modifying the list during iteration
                    # or fully recalculates neighbors each time (as done here).

        return neighbors

    def solve(self, initial_solution: CVRPSolution = None) -> CVRPSolution:
        if initial_solution is None:
            greedy_solver = GreedySolver(self.problem)
            current_solution = greedy_solver.solve()
        else:
            current_solution = initial_solution.copy()

        # Upewnij się, że rozwiązania mają obliczony fitness
        if current_solution.fitness == float('inf'): current_solution.evaluate()
        self.best_solution = current_solution.copy()

        tabu_list: Dict[Tuple, int] = {}

        # --- ZMIANA: Logowanie początkowego stanu ---
        self.logger.log(0, self.best_solution.fitness, current_solution.fitness)

        for iteration in range(1, self.max_iterations + 1):  # Pętla od 1 do max_iterations włącznie
            all_neighbors = self.get_neighbors(current_solution)

            if self.neighborhood_limit is not None and len(all_neighbors) > self.neighborhood_limit:
                considered_neighbors = random.sample(all_neighbors, self.neighborhood_limit)
            else:
                considered_neighbors = all_neighbors

            if not considered_neighbors:
                print(f"TS: No valid neighbors found at iteration {iteration}. Stopping.")
                break

            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_move = None
            found_admissible = False

            for neighbor, move in considered_neighbors:
                is_tabu = move in tabu_list and tabu_list[move] > iteration
                aspiration_met = is_tabu and neighbor.fitness < self.best_solution.fitness

                if (not is_tabu) or aspiration_met:
                    if neighbor.fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = neighbor.fitness
                        best_move = move
                        found_admissible = True

            # Obsługa sytuacji braku dopuszczalnego ruchu
            if not found_admissible:
                # Zamiast przerywać, można wybrać najlepszego sąsiada (nawet tabu), jeśli nie ma aspiracji
                if not best_neighbor:  # Jeśli żaden sąsiad nie został wybrany (nawet tabu)
                    print(
                        f"TS: No improving or non-tabu neighbor found at iteration {iteration}. Selecting best tabu neighbor (if any).")
                    # Wybierz najlepszego sąsiada (nawet jeśli jest tabu i nie spełnia kryterium aspiracji)
                    # To pozwala algorytmowi kontynuować, potencjalnie pogarszając rozwiązanie
                    neighbors_sorted = sorted(considered_neighbors, key=lambda item: item[0].fitness)
                    if neighbors_sorted:
                        best_neighbor = neighbors_sorted[0][0]  # Weź najlepszego sąsiada
                        best_move = neighbors_sorted[0][1]  # i jego ruch
                        print(f"TS: Selected best (potentially tabu) neighbor with fitness {best_neighbor.fitness:.2f}")
                    else:  # To nie powinno się zdarzyć, jeśli considered_neighbors nie było puste
                        print(f"TS: No neighbors available at all at iteration {iteration}. Stopping.")
                        break
                        # Jeśli best_neighbor został znaleziony (np. przez aspirację), kontynuuj normalnie
                elif not best_move:  # Jeśli best_neighbor jest, ale best_move nie (nie powinno się zdarzyć)
                    print(f"TS: Logic error - best_neighbor found but no best_move at iter {iteration}. Stopping.")
                    break

            # Jeśli nadal nie ma sąsiada (np. gdy considered_neighbors było puste i neighbors_sorted też)
            if best_neighbor is None:
                print(f"TS: No neighbor could be selected at iteration {iteration}. Stopping.")
                break

            # Aktualizacja
            current_solution = best_neighbor

            # Aktualizacja najlepszego globalnego
            if current_solution.fitness < self.best_solution.fitness:
                self.best_solution = current_solution.copy()
                # print(f"TS Iter {iteration}: New best found: {self.best_solution.fitness:.2f}") # Opcjonalny log

            # Zarządzanie listą tabu
            if best_move:
                tabu_list[best_move] = iteration + self.tabu_tenure
            if iteration % 50 == 0:
                current_iter = iteration
                tabu_list = {move: expiry for move, expiry in tabu_list.items() if expiry > current_iter}

            # --- ZMIANA: Logowanie w każdej iteracji ---
            self.logger.log(iteration, self.best_solution.fitness, current_solution.fitness)
            # --- Koniec zmiany ---

        # --- ZMIANA: Wyłącz indywidualne rysowanie wykresu ---
        # self.logger.save() # Zapis CSV może być nadal przydatny
        # self.logger.plot(f"Tabu Search ({self.problem.name}) - Run Result")
        # --- Koniec zmiany ---

        return self.best_solution


try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_results_dir = os.path.join(project_root, "results")
except NameError:
    print("Warning: Could not determine script directory. Using relative path '../results/'.")
    base_results_dir = "../results/"

def main():
    # Load the problem
    problem = CVRPProblem()
    data_file_name = "A-n32-k5.vrp"
    num_runs = 10
    ga_params = {
        "population_size": 100,
        "max_evaluations": 20000,
        "crossover_rate": 0.7,
        "mutation_rate": 0.2,
        "tour_n": 5,
        "elitism": 2
    }
    ts_params = {
        "tabu_tenure": 15,
        "max_iterations": 500,
        "neighborhood_limit": 50
    }

    data_file = os.path.join("..", "data", data_file_name)
    try:
        problem.load_from_file(data_file)
    except FileNotFoundError:
        print(f"Error: Problem file not found at {data_file}")
        # Spróbuj w bieżącym katalogu jako fallback
        data_file = data_file_name
        try:
            problem.load_from_file(data_file)
        except FileNotFoundError:
            print(f"Error: Problem file also not found at {data_file}")
            return  # Zakończ, jeśli nie można załadować danych

    print("Problem loaded:")
    print(f"Name: {problem.name}")
    print(f"Dimension: {problem.dimension}")
    print(f"Capacity: {problem.capacity}")
    print(f"Depot: {problem.depot_index + 1}")

    os.makedirs(base_results_dir, exist_ok=True)

    ## --- Przygotowanie folderu dla serii GA ---
    timestamp_str = datetime.today().strftime("%Y%m%d-%H%M%S")
    ga_series_folder_name = f"{problem.name}_GA_{timestamp_str}"
    ga_series_folder_path = os.path.join(base_results_dir, ga_series_folder_name)
    try:
        os.makedirs(ga_series_folder_path, exist_ok=True)
        print(f"Created GA series folder: {ga_series_folder_path}")
    except OSError as e:
        print(f"Error creating GA series folder {ga_series_folder_path}: {e}")
        ga_series_folder_path = base_results_dir # Fallback

    # --- Genetic Algorithm Runs ---
    print(f"\nRunning Genetic Algorithm ({num_runs} runs)...")
    ga_results_fitness = []
    ga_times = []
    best_ga_solution = None
    ga_run_histories = []
    for i in range(num_runs):
        run_id = f"run_{i + 1}"
        print(f"  GA {run_id}/{num_runs}", end='\r')
        start_time = time.time()

        # --- Przekaż folder serii do konstruktora GA ---
        ga_params_run = ga_params.copy()
        ga_params_run["log_directory"] = ga_series_folder_path
        ga = GeneticAlgorithm(problem, **ga_params_run)

        ga_solution = ga.evolve()  # evolve nie powinno już zapisywać ani rysować
        ga_time = time.time() - start_time

        run_history = ga.logger.data
        ga_run_histories.append(run_history)

        # --- Zapisz log, używając loggera z instancji GA ---
        if run_history:
            ga.logger.save(run_identifier=run_id)  # Logger już wie, gdzie zapisać
            # ga.logger.plot(run_identifier=run_id) # Opcjonalny wykres indywidualny

        # Zbieranie wyników (bez zmian)
        if ga_solution.fitness != float('inf'):
            ga_results_fitness.append(ga_solution.fitness)
            if best_ga_solution is None or ga_solution.fitness < best_ga_solution.fitness:
                best_ga_solution = ga_solution
        else:
            print(f"\n  GA {run_id} failed.")
        ga_times.append(ga_time)

    print(f"\nGA Runs finished.")

    # Generowanie zbiorczego wykresu GA
    if ga_run_histories:
        Logger.plot_summary(ga_run_histories,
                            filename_prefix=f"{problem.name}_ga_summary",
                            directory=base_results_dir,
                            x_axis_label="Generation",
                            title_suffix=f"Genetic Algorithm ({problem.name})")
    else:
        print("No successful GA runs recorded, skipping summary plot.")

    # Statystyki GA (obliczane jak poprzednio, na podstawie `ga_results_fitness`)
    if ga_results_fitness:
        ga_best_overall = min(ga_results_fitness)
        ga_worst_of_bests = max(ga_results_fitness)
        ga_avg_of_bests = statistics.mean(ga_results_fitness)
        ga_std_dev = statistics.stdev(ga_results_fitness) if len(ga_results_fitness) > 1 else 0
        ga_avg_time = statistics.mean(ga_times)
        print("\nGenetic Algorithm Results (across runs):")
        print(f"  Best fitness found: {ga_best_overall:.2f}")
        print(f"  Worst best fitness: {ga_worst_of_bests:.2f}")
        print(f"  Average best fitness: {ga_avg_of_bests:.2f}")
        print(f"  Std Dev of best fitness: {ga_std_dev:.2f}")
        print(f"  Average execution time: {ga_avg_time:.2f} seconds")
        if best_ga_solution:
            print("  Best GA Solution found:")
            best_ga_solution.display()
    else:
        print("\nGenetic Algorithm did not produce any valid results.")

    # --- Przygotowanie folderu dla serii TS ---
    ts_series_folder_name = f"{problem.name}_TS_{timestamp_str}"
    ts_series_folder_path = os.path.join(base_results_dir, ts_series_folder_name)
    try:
        os.makedirs(ts_series_folder_path, exist_ok=True)
        print(f"Created TS series folder: {ts_series_folder_path}")
    except OSError as e:
        print(f"Error creating TS series folder {ts_series_folder_path}: {e}")
        ts_series_folder_path = base_results_dir  # Fallback

    # --- Tabu Search Runs ---
    print(f"\nRunning Tabu Search ({num_runs} runs)...")
    ts_results_fitness = []
    ts_times = []
    best_ts_solution = None
    ts_run_histories = []
    for i in range(num_runs):
        run_id = f"run_{i + 1}"
        print(f"  TS {run_id}/{num_runs}", end='\r')
        start_time = time.time()

        # --- Przekaż folder serii do konstruktora TS ---
        ts_params_run = ts_params.copy()
        ts_params_run["log_directory"] = ts_series_folder_path
        initial_sol_ts = None
        tabu_search = TabuSearch(problem, **ts_params_run)

        tabu_solution = tabu_search.solve()  # solve nie powinno już zapisywać ani rysować
        tabu_time = time.time() - start_time

        run_history = tabu_search.logger.data
        ts_run_histories.append(run_history)

        # --- Zapisz log, używając loggera z instancji TS ---
        if run_history:
            tabu_search.logger.save(run_identifier=run_id)  # Logger już wie, gdzie zapisać
            # tabu_search.logger.plot(run_identifier=run_id) # Opcjonalny wykres indywidualny

        # Zbieranie wyników (bez zmian)
        if tabu_solution.fitness != float('inf'):
            ts_results_fitness.append(tabu_solution.fitness)
            if best_ts_solution is None or tabu_solution.fitness < best_ts_solution.fitness:
                best_ts_solution = tabu_solution
        else:
            print(f"\n  TS {run_id} failed.")
        ts_times.append(tabu_time)

    print(f"\nTS Runs finished.")

    if ts_run_histories:
        Logger.plot_summary(ts_run_histories,
                            filename_prefix=f"{problem.name}_ts_summary",
                            directory=base_results_dir,
                            x_axis_label="Iteration",
                            title_suffix=f"Tabu Search ({problem.name})")
    else:
        print("No successful TS runs recorded, skipping summary plot.")

    # Statystyki TS (jak poprzednio)
    if ts_results_fitness:
        ts_best_overall = min(ts_results_fitness)
        ts_worst_of_bests = max(ts_results_fitness)
        ts_avg_of_bests = statistics.mean(ts_results_fitness)
        ts_std_dev = statistics.stdev(ts_results_fitness) if len(ts_results_fitness) > 1 else 0
        ts_avg_time = statistics.mean(ts_times)
        print("\nTabu Search Results (across runs):")
        print(f"  Best fitness found: {ts_best_overall:.2f}")
        print(f"  Worst best fitness: {ts_worst_of_bests:.2f}")
        print(f"  Average best fitness: {ts_avg_of_bests:.2f}")
        print(f"  Std Dev of best fitness: {ts_std_dev:.2f}")
        print(f"  Average execution time: {ts_avg_time:.2f} seconds")
        if best_ts_solution:
            print("  Best TS Solution found:")
            best_ts_solution.display()
    else:
        print("\nTabu Search did not produce any valid results.")

    # --- Porównanie z Random i Greedy ---
    print("\nReference Solvers:")

    # Random Solver
    random_solver = RandomSolver(problem)
    num_random_samples = 1000  # Mniej próbek niż poprzednio, dla szybkości
    random_results = [random_solver.solve().fitness for _ in range(num_random_samples)]
    random_best = min(random_results)
    random_worst = max(random_results)
    random_avg = statistics.mean(random_results)
    random_std = statistics.stdev(random_results) if len(random_results) > 1 else 0
    print("Random Solver:")
    print(f"  Best (of {num_random_samples}): {random_best:.2f}")
    print(f"  Worst (of {num_random_samples}): {random_worst:.2f}")
    print(f"  Average (of {num_random_samples}): {random_avg:.2f}")
    print(f"  Std Dev (of {num_random_samples}): {random_std:.2f}")

    # Greedy Solver (testowane wszystkie starty)
    greedy_solver = GreedySolver(problem)
    greedy_results = []
    # Uruchom z każdego węzła niebędącego depotem + raz z depotu
    customer_nodes = [i for i in range(problem.dimension) if i != problem.depot_index]
    start_nodes_greedy = customer_nodes + [problem.depot_index]
    for start_node in start_nodes_greedy:
        greedy_solution = greedy_solver.solve(start_node)
        greedy_results.append(greedy_solution.fitness)
    greedy_best = min(greedy_results)
    greedy_worst = max(greedy_results)
    greedy_avg = statistics.mean(greedy_results)
    greedy_std = statistics.stdev(greedy_results) if len(greedy_results) > 1 else 0
    print("Greedy Solver:")
    print(f"  Best (from all starts): {greedy_best:.2f}")
    print(f"  Worst (from all starts): {greedy_worst:.2f}")
    print(f"  Average (from all starts): {greedy_avg:.2f}")
    print(f"  Std Dev (from all starts): {greedy_std:.2f}")

    print("\nFinal Comparison (Best Found & Average Best):")
    print(f"  Random (Best of N): {random_best:.2f}")  # random_best zdefiniowane wcześniej
    print(f"  Greedy (Best of all starts): {greedy_best:.2f}")  # greedy_best zdefiniowane wcześniej
    if ga_results_fitness:
        print(f"  Genetic Algorithm: {ga_best_overall:.2f} (Avg Best: {ga_avg_of_bests:.2f} ± {ga_std_dev:.2f})")
    else:
        print("  Genetic Algorithm: No valid results.")
    if ts_results_fitness:
        print(f"  Tabu Search: {ts_best_overall:.2f} (Avg Best: {ts_avg_of_bests:.2f} ± {ts_std_dev:.2f})")
    else:
        print("  Tabu Search: No valid results.")


if __name__ == "__main__":
    main()