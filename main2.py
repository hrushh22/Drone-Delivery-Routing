###################### Complete Hybrid Optimization with All Required Functions ##############################

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime

# Create output directories if they don't exist
os.makedirs("Visual2", exist_ok=True)
os.makedirs("Output2", exist_ok=True)

# 1. Helper Functions
def path_distance(path):
    """Calculate total distance of a path including return to depot"""
    total = 0
    # Distance from depot to first point
    total += np.linalg.norm(delivery_points[path[0]] - depot)
    # Distance between delivery points
    for i in range(len(path)-1):
        total += distance_matrix[path[i], path[i+1]]
    # Distance from last point back to depot
    total += np.linalg.norm(delivery_points[path[-1]] - depot)
    return total

def save_output(text):
    """Save text output to file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Output2/results_{timestamp}.txt"
    with open(filename, 'a') as f:
        f.write(text + "\n")
    print(text)

def plot_route(route, title, save_name=None, show=False):
    """Generic function to plot any route"""
    plt.figure(figsize=(10, 8))
    plt.scatter(depot[0], depot[1], c='red', s=200, label='Depot')
    plt.scatter(delivery_points[:, 0], delivery_points[:, 1], c='blue', s=100, label='Delivery Points')
    
    for (x, y, w, h) in no_fly_zones:
        plt.gca().add_patch(Rectangle((x, y), w, h, color='gray', alpha=0.5, label='No-Fly Zone'))
    
    route_coords = [depot] + [delivery_points[i] for i in route] + [depot]
    for i in range(len(route_coords)-1):
        plt.plot([route_coords[i][0], route_coords[i+1][0]], 
                 [route_coords[i][1], route_coords[i+1][1]], 'r-')
    
    plt.title(f"{title}\nTotal Distance: {path_distance(route):.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    
    if save_name:
        plt.savefig(f"Visual2/{save_name}.png")
    if show:
        plt.show()
    plt.close()

# 2. Problem Setup
n_deliveries = 15
battery_limit = 100
time_windows = {i: (random.randint(0, 10), random.randint(10, 20)) for i in range(n_deliveries)}
no_fly_zones = [(20, 30, 10, 10), (60, 70, 15, 15)]

np.random.seed(42)
delivery_points = np.random.rand(n_deliveries, 2) * 100
depot = np.array([50, 50])

distance_matrix = np.zeros((n_deliveries, n_deliveries))
for i in range(n_deliveries):
    for j in range(n_deliveries):
        distance_matrix[i, j] = np.linalg.norm(delivery_points[i] - delivery_points[j])

# 3. Core Algorithm Components
def initialize_population(pop_size, num_cities):
    """Create initial population of random routes"""
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def crossover(parent1, parent2):
    """Ordered crossover for genetic algorithm"""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    remaining = [gene for gene in parent2 if gene not in child]
    child = [gene if gene != -1 else remaining.pop(0) for gene in child]
    return child

def mutate(route, mutation_rate=0.1):
    """Swap mutation for genetic algorithm"""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

class Particle:
    """Particle for PSO"""
    def __init__(self, route):
        self.route = route
        self.velocity = [0] * len(route)
        self.best_route = route.copy()
        self.best_fitness = path_distance(route)
    
    def update_position(self):
        """Update route based on velocity"""
        new_route = []
        for i in range(len(self.route)):
            new_pos = self.route[i] + self.velocity[i]
            new_pos = max(0, min(n_deliveries-1, round(new_pos)))
            new_route.append(new_pos)
        # Ensure all cities are visited exactly once
        self.route = list(dict.fromkeys(new_route))
        missing = set(range(n_deliveries)) - set(self.route)
        self.route.extend(missing)
        return self

def update_pheromones(pheromones, routes, fitness_values, decay=0.1):
    """Update pheromone matrix for ACO"""
    pheromones *= (1 - decay)
    for i, route in enumerate(routes):
        for j in range(len(route)-1):
            pheromones[route[j], route[j+1]] += 1 / (fitness_values[i] + 1e-10)
    return pheromones

def ant_colony_optimization(population, pheromones, iterations=5):
    """ACO implementation"""
    for _ in range(iterations):
        ant_routes = []
        for route in population:
            new_route = []
            current = route[0]
            new_route.append(current)
            remaining = set(route[1:])
            while remaining:
                probabilities = [
                    (pheromones[current, next_city] / (distance_matrix[current, next_city] + 1e-10))
                    if next_city in remaining else 0
                    for next_city in range(n_deliveries)
                ]
                total = sum(probabilities)
                if total == 0:
                    next_city = random.choice(list(remaining))
                else:
                    probabilities = [p/total for p in probabilities]
                    next_city = np.random.choice(range(n_deliveries), p=probabilities)
                new_route.append(next_city)
                remaining.remove(next_city)
                current = next_city
            ant_routes.append(new_route)
        pheromones = update_pheromones(pheromones, ant_routes, [path_distance(r) for r in ant_routes])
    return ant_routes, pheromones

# 4. Individual Algorithm Implementations
def genetic_algorithm(pop_size=50, generations=20):
    population = initialize_population(pop_size, n_deliveries)
    best_route = min(population, key=path_distance)
    
    for gen in range(generations):
        fitness = [1 / (path_distance(route) + 1e-10) for route in population]
        new_population = []
        for _ in range(pop_size // 2):
            parents = random.choices(population, weights=fitness, k=2)
            child1, child2 = crossover(parents[0], parents[1]), crossover(parents[1], parents[0])
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
        
        current_best = min(population, key=path_distance)
        if path_distance(current_best) < path_distance(best_route):
            best_route = current_best.copy()
    
    return best_route

def particle_swarm_optimization(pop_size=30, iterations=20):
    particles = [Particle(random.sample(range(n_deliveries), n_deliveries)) for _ in range(pop_size)]
    global_best_route = min([p.route for p in particles], key=path_distance)
    global_best_fitness = path_distance(global_best_route)
    
    for _ in range(iterations):
        for p in particles:
            # Update velocity
            r1, r2 = random.random(), random.random()
            cognitive = [1.5 * r1 * (b - c) for b, c in zip(p.best_route, p.route)]
            social = [1.5 * r2 * (g - c) for g, c in zip(global_best_route, p.route)]
            p.velocity = [0.7*v + c + s for v, c, s in zip(p.velocity, cognitive, social)]
            
            # Update position
            p.update_position()
            
            # Update personal best
            current_fitness = path_distance(p.route)
            if current_fitness < p.best_fitness:
                p.best_route = p.route.copy()
                p.best_fitness = current_fitness
                
                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_route = p.route.copy()
                    global_best_fitness = current_fitness
    
    return global_best_route

def ant_colony_optimization_only(pop_size=20, iterations=15):
    pheromones = np.ones((n_deliveries, n_deliveries))
    population = initialize_population(pop_size, n_deliveries)
    
    for _ in range(iterations):
        ant_routes = []
        for route in population:
            new_route = []
            current = route[0]
            new_route.append(current)
            remaining = set(route[1:])
            while remaining:
                next_node = max(remaining, key=lambda x: pheromones[current, x] / (distance_matrix[current, x] + 1e-10))
                new_route.append(next_node)
                remaining.remove(next_node)
                current = next_node
            ant_routes.append(new_route)
        pheromones = update_pheromones(pheromones, ant_routes, [path_distance(r) for r in ant_routes])
    
    return min(ant_routes, key=path_distance)

# 5. Visual2 of Each Algorithm's Progress
def visualize_algorithm_progress():
    # Initial random solution
    initial_route = random.sample(range(n_deliveries), n_deliveries)
    plot_route(initial_route, "Initial Random Route", "initial_route")
    
    # GA Visual2
    save_output("\nRunning Genetic Algorithm...")
    ga_route = genetic_algorithm()
    plot_route(ga_route, "GA Optimized Route", "ga_optimized")
    save_output(f"GA Best Distance: {path_distance(ga_route):.2f}")
    
    # PSO Visual2
    save_output("\nRunning Particle Swarm Optimization...")
    pso_route = particle_swarm_optimization()
    plot_route(pso_route, "PSO Optimized Route", "pso_optimized")
    save_output(f"PSO Best Distance: {path_distance(pso_route):.2f}")
    
    # ACO Visual2
    save_output("\nRunning Ant Colony Optimization...")
    aco_route = ant_colony_optimization_only()
    plot_route(aco_route, "ACO Optimized Route", "aco_optimized")
    save_output(f"ACO Best Distance: {path_distance(aco_route):.2f}")
    
    return initial_route, ga_route, pso_route, aco_route

# 6. Main Hybrid Algorithm
def hybrid_drone_optimization(pop_size=50, generations=20):
    # First visualize individual algorithms
    initial_route, ga_route, pso_route, aco_route = visualize_algorithm_progress()
    
    # Then run hybrid approach
    save_output("\n=== Starting Hybrid GA-PSO-ACO Optimization ===")
    population = initialize_population(pop_size, n_deliveries)
    pheromones = np.ones((n_deliveries, n_deliveries))
    global_best_route = min(population, key=path_distance)
    
    for gen in range(generations):
        # GA Phase
        fitness = [1 / (path_distance(route) + 1e-10) for route in population]
        new_population = []
        for _ in range(pop_size // 2):
            parents = random.choices(population, weights=fitness, k=2)
            child1, child2 = crossover(parents[0], parents[1]), crossover(parents[1], parents[0])
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
        
        # PSO Phase
        particles = [Particle(route) for route in population]
        for p in particles:
            # Update velocity
            r1, r2 = random.random(), random.random()
            cognitive = [1.5 * r1 * (b - c) for b, c in zip(p.best_route, p.route)]
            social = [1.5 * r2 * (g - c) for g, c in zip(global_best_route, p.route)]
            p.velocity = [0.7*v + c + s for v, c, s in zip(p.velocity, cognitive, social)]
            p.update_position()
        
        # ACO Phase
        population, pheromones = ant_colony_optimization(population, pheromones)
        
        # Update global best
        current_best = min(population, key=path_distance)
        if path_distance(current_best) < path_distance(global_best_route):
            global_best_route = current_best.copy()
            save_output(f"Generation {gen+1}: New best distance = {path_distance(global_best_route):.2f}")
    
    return initial_route, ga_route, pso_route, aco_route, global_best_route

# 7. Main Execution
if __name__ == "__main__":
    # Run the complete optimization
    initial, ga, pso, aco, hybrid = hybrid_drone_optimization(pop_size=30, generations=20)
    
    # Save comparison results
    save_output("\n=== Algorithm Comparison ===")
    save_output(f"Initial Random Route Distance: {path_distance(initial):.2f}")
    save_output(f"Genetic Algorithm Best Distance: {path_distance(ga):.2f}")
    save_output(f"Particle Swarm Optimization Best Distance: {path_distance(pso):.2f}")
    save_output(f"Ant Colony Optimization Best Distance: {path_distance(aco):.2f}")
    save_output(f"Hybrid Approach Best Distance: {path_distance(hybrid):.2f}")
    
    # Plot final comparison
    plt.figure(figsize=(15, 10))
    algorithms = ['Initial', 'GA', 'PSO', 'ACO', 'Hybrid']
    distances = [path_distance(initial), path_distance(ga), path_distance(pso), 
                path_distance(aco), path_distance(hybrid)]
    
    plt.bar(algorithms, distances, color=['red', 'blue', 'green', 'purple', 'orange'])
    plt.title("Algorithm Performance Comparison")
    plt.ylabel("Total Route Distance")
    plt.xlabel("Optimization Method")
    plt.grid(True)
    plt.savefig("Visual2/algorithm_comparison.png")
    plt.close()
    
    # Save final hybrid solution
    plot_route(hybrid, "Final Hybrid Optimized Route", "final_hybrid_route")