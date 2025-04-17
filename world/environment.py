import math
import random
from dna.genome import NeuralGenome
from dna.gene import Gene
def distance(pos1, pos2):
    """Calculate distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class Organism:
    def __init__(self, position, genome=None):
        """Initialize a new organism"""
        self.position = position
        self.age = 0
        self.alive = True
        self.energy = 100
        self.hydration = 100
        self.health = 100
        self.neural_genome = genome if genome else NeuralGenome()
        self.memory = []
        self.traits = self.initialize_traits()
        
    def initialize_traits(self):
        """Initialize organism traits based on genome"""
        return {
            "size": 0.5,  # 0 to 1 scale
            "speed": 0.5,
            "strength": 0.5,
            "perception": 0.5,
            "metabolism": 0.5
        }
        
    def calculate_trait(self, trait_name):
        """Get the current value of a trait, accounting for conditions"""
        base_value = self.traits[trait_name]
        
        # Apply modifiers based on organism state
        if trait_name == "speed":
            # Lower energy or hydration reduces speed
            energy_factor = max(0.2, self.energy / 100)
            hydration_factor = max(0.2, self.hydration / 100)
            return base_value * energy_factor * hydration_factor
            
        if trait_name == "strength":
            # Lower health reduces strength
            health_factor = max(0.2, self.health / 100)
            return base_value * health_factor
            
        return base_value
    
    def update_internal_states(self, world):
        """Update internal states based on metabolism and environment"""
        # Basic metabolism costs
        metabolism_rate = self.calculate_trait("metabolism")
        base_energy_cost = 0.5 + (metabolism_rate * 0.5)  # Higher metabolism = higher cost
        
        # Apply energy cost
        self.energy -= base_energy_cost
        
        # Apply hydration cost
        self.hydration -= 0.5 + (metabolism_rate * 0.3)
        
        # Check if organism should die
        if self.energy <= 0 or self.hydration <= 0 or self.health <= 0:
            self.alive = False
            return
        
        # Environmental effects
        current_terrain = world.get_terrain_at(self.position)
        
        # Extreme environments affect health
        if current_terrain == "desert":
            self.hydration -= 1.0  # Extra hydration loss in desert
        elif current_terrain == "snow":
            self.energy -= 1.0  # Extra energy loss in cold
            
        # Cap stats at maximum values
        self.energy = min(100, self.energy)
        self.hydration = min(100, self.hydration)
        self.health = min(100, self.health)
    
    def sense_environment(self, world):
        """Gather sensory data from environment"""
        sensory_data = {}
        perception_radius = 0.2 * (1 + self.calculate_trait("perception"))
        
        # Get terrain information
        sensory_data["current_terrain"] = world.get_terrain_at(self.position)
        
        # Find nearby resources
        nearby_resources = world.get_resources_in_radius(self.position, perception_radius)
        
        # Process resources by type
        food_resources = [r for r in nearby_resources if r["type"] == "food"]
        water_resources = [r for r in nearby_resources if r["type"] == "water"]
        
        # Find closest food and water if any
        if food_resources:
            closest_food = min(food_resources, key=lambda r: distance(self.position, r["position"]))
            sensory_data["food_distance"] = distance(self.position, closest_food["position"]) / perception_radius
            sensory_data["food_direction"] = math.atan2(
                closest_food["position"][1] - self.position[1],
                closest_food["position"][0] - self.position[0]
            ) / math.pi  # Normalize to -1 to 1
        else:
            sensory_data["food_distance"] = 1.0  # Maximum distance
            sensory_data["food_direction"] = 0.0  # Neutral direction
            
        if water_resources:
            closest_water = min(water_resources, key=lambda r: distance(self.position, r["position"]))
            sensory_data["water_distance"] = distance(self.position, closest_water["position"]) / perception_radius
            sensory_data["water_direction"] = math.atan2(
                closest_water["position"][1] - self.position[1],
                closest_water["position"][0] - self.position[0]
            ) / math.pi  # Normalize to -1 to 1
        else:
            sensory_data["water_distance"] = 1.0  # Maximum distance
            sensory_data["water_direction"] = 0.0  # Neutral direction
        
        # Find nearby organisms
        nearby_organisms = world.get_organisms_in_radius(self.position, perception_radius)
        nearby_organisms = [o for o in nearby_organisms if o is not self]
        
        if nearby_organisms:
            closest_organism = min(nearby_organisms, key=lambda o: distance(self.position, o.position))
            sensory_data["organism_distance"] = distance(self.position, closest_organism.position) / perception_radius
            sensory_data["organism_direction"] = math.atan2(
                closest_organism.position[1] - self.position[1],
                closest_organism.position[0] - self.position[0]
            ) / math.pi  # Normalize to -1 to 1
            sensory_data["organism_size"] = closest_organism.calculate_trait("size")
        else:
            sensory_data["organism_distance"] = 1.0
            sensory_data["organism_direction"] = 0.0
            sensory_data["organism_size"] = 0.0
        
        # Internal state sensors
        sensory_data["energy_level"] = self.energy / 100.0
        sensory_data["hydration_level"] = self.hydration / 100.0
        sensory_data["health_level"] = self.health / 100.0
        
        return sensory_data
    
    def update(self, world):
        """Main update function for organism"""
        self.age += 1
        
        # Update internal states based on environment
        self.update_internal_states(world)
        
        # If dead, stop processing
        if not self.alive:
            return False
        
        # Sense environment
        sensory_data = self.sense_environment(world)
        
        # Process through neural network
        decision_outputs = self.neural_genome.process_inputs(sensory_data)
        
        # Convert neural outputs to actions
        actions = self.convert_outputs_to_actions(decision_outputs)
        
        # Execute actions
        success_level = self.execute_actions(actions, world)
        
        # Learn from results
        self.neural_genome.learn_from_feedback(success_level)
        
        # Update memories
        self.update_memory(sensory_data, actions, success_level)
        
        return self.alive
    
    def convert_outputs_to_actions(self, outputs):
        """Convert neural outputs to concrete actions"""
        actions = {}
        
        # Movement direction (combine two outputs for x and y components)
        if "m_move_x" in outputs and "m_move_y" in outputs:
            # Convert activations to direction
            x_component = outputs["m_move_x"] * 2 - 1  # -1 to 1
            y_component = outputs["m_move_y"] * 2 - 1  # -1 to 1
            
            # Calculate direction angle
            actions["move_direction"] = math.atan2(y_component, x_component)
            
            # Calculate magnitude (speed)
            magnitude = math.sqrt(x_component**2 + y_component**2)
            actions["move_speed"] = min(1.0, magnitude)
        
        # Eating action
        if "m_eat" in outputs:
            actions["eat"] = outputs["m_eat"] > 0.5
        
        # Drinking action
        if "m_drink" in outputs:
            actions["drink"] = outputs["m_drink"] > 0.5
        
        # Attack action
        if "m_attack" in outputs:
            actions["attack"] = outputs["m_attack"] > 0.5
        
        # Flee action
        if "m_flee" in outputs:
            actions["flee"] = outputs["m_flee"] > 0.5
        
        # Reproduce action
        if "m_reproduce" in outputs:
            threshold = 0.7  # Higher threshold for reproduction
            actions["reproduce"] = outputs["m_reproduce"] > threshold
        
        # Rest action (reduces energy consumption)
        if "m_rest" in outputs:
            actions["rest"] = outputs["m_rest"] > 0.5
        
        return actions
    
    def execute_actions(self, actions, world):
        """Execute decided actions and return success level (-1 to 1)"""
        success_value = 0.0  # Neutral starting point
        energy_before = self.energy
        
        # Handle movement
        if "move_direction" in actions and "move_speed" in actions:
            # Calculate movement cost based on speed and organism size
            speed = actions["move_speed"] * 0.1  # Scale down for reasonable movement
            size_factor = self.calculate_trait("size")
            movement_cost = speed * (0.5 + size_factor * 2)  # Larger = more cost
            
            # Apply movement
            direction = actions["move_direction"]
            new_x = self.position[0] + math.cos(direction) * speed
            new_y = self.position[1] + math.sin(direction) * speed
            
            # Keep within world boundaries
            new_x = max(0, min(world.width, new_x))
            new_y = max(0, min(world.height, new_y))
            
            # Check for collisions or terrain effects
            terrain = world.get_terrain_at((new_x, new_y))
            if terrain == "mountain":
                # Mountains are harder to traverse
                movement_cost *= 2
                speed *= 0.5
            
            # Update position if enough energy left
            if self.energy > movement_cost:
                self.position = (new_x, new_y)
                self.energy -= movement_cost
            else:
                # Not enough energy to move
                success_value -= 0.2
        
        # Handle eating
        if actions.get("eat", False):
            nearby_food = world.get_resources_in_radius(self.position, 0.05)
            edible_food = [r for r in nearby_food if r["type"] == "food"]
            
            if edible_food:
                # Eat closest food
                food = min(edible_food, key=lambda r: distance(self.position, r["position"]))
                self.energy += food["energy"]
                self.energy = min(200, self.energy)  # Cap energy
                world.resources.remove(food)
                
                success_value += 0.8  # Successful eating is good
            else:
                # Failed eating attempt
                self.energy -= 2  # Small cost for failed attempt
                success_value -= 0.1
        
        # Handle drinking
        if actions.get("drink", False):
            nearby_water = world.get_resources_in_radius(self.position, 0.05)
            drinkable_water = [r for r in nearby_water if r["type"] == "water"]
            
            if drinkable_water:
                # Drink from closest water
                water = min(drinkable_water, key=lambda r: distance(self.position, r["position"]))
                self.hydration += 50
                self.hydration = min(100, self.hydration)
                
                # Reduce water source
                water["amount"] -= 5
                if water["amount"] <= 0:
                    world.resources.remove(water)
                    
                success_value += 0.7  # Successful drinking is good
            else:
                # Failed drinking attempt
                self.energy -= 1  # Small cost for failed attempt
                success_value -= 0.1
        
        # Handle attacking
        if actions.get("attack", False):
            nearby_organisms = world.get_organisms_in_radius(self.position, 0.1)
            nearby_organisms = [o for o in nearby_organisms if o is not self and o.alive]
            
            if nearby_organisms:
                # Attack closest organism
                target = min(nearby_organisms, key=lambda o: distance(self.position, o.position))
                
                # Calculate attack strength
                attack_strength = self.calculate_trait("strength") * 10
                target_defense = target.calculate_trait("size") * 5
                
                # Apply damage
                damage_dealt = max(0, attack_strength - target_defense)
                target.health -= damage_dealt
                
                # Attack costs energy
                self.energy -= 5 + (self.calculate_trait("strength") * 5)
                
                if damage_dealt > 0:
                    success_value += 0.5  # Successful attack
                    
                    # If target killed, get bonus energy
                    if target.health <= 0:
                        # Consume part of the target
                        energy_gain = 20 + (target.calculate_trait("size") * 30)
                        self.energy += energy_gain
                        self.energy = min(100, self.energy)
                        success_value += 0.5  # Extra success for kill
                else:
                    success_value -= 0.1  # Attack didn't do damage
            else:
                # No targets to attack
                self.energy -= 3  # Wasted energy
                success_value -= 0.2
        
        # Handle fleeing
        if actions.get("flee", False):
            nearby_organisms = world.get_organisms_in_radius(self.position, 0.2)
            threats = [o for o in nearby_organisms if o is not self and o.calculate_trait("strength") > self.calculate_trait("strength")]
            
            if threats:
                # Find closest threat
                threat = min(threats, key=lambda o: distance(self.position, o.position))
                
                # Calculate flee direction (away from threat)
                flee_direction = math.atan2(
                    self.position[1] - threat.position[1],
                    self.position[0] - threat.position[0]
                )
                
                # Move quickly away from threat
                flee_speed = 0.15 * self.calculate_trait("speed")
                flee_cost = flee_speed * 2.5  # Fleeing costs more energy
                
                # Apply movement
                new_x = self.position[0] + math.cos(flee_direction) * flee_speed
                new_y = self.position[1] + math.sin(flee_direction) * flee_speed
                
                # Keep within boundaries
                new_x = max(0, min(world.width, new_x))
                new_y = max(0, min(world.height, new_y))
                
                # Update position if enough energy
                if self.energy > flee_cost:
                    self.position = (new_x, new_y)
                    self.energy -= flee_cost
                    success_value += 0.3  # Successful fleeing
                else:
                    # Not enough energy
                    success_value -= 0.3
            else:
                # No threats to flee from
                self.energy -= 2
                success_value -= 0.2
        
        # Handle reproduction
        if actions.get("reproduce", False):
            # Check if organism has enough energy and is mature enough
            if self.energy > 60 and self.age > 100:
                # Create offspring
                offspring_position = (
                    self.position[0] + (random.random() * 0.1 - 0.05),
                    self.position[1] + (random.random() * 0.1 - 0.05)
                )
                
                # Clone genome with mutations
                offspring_genome = self.neural_genome.clone_with_mutations()
                
                # Create new organism
                offspring = Organism(offspring_position, offspring_genome)
                
                # Transfer some energy to offspring
                energy_transfer = self.energy * 0.3
                self.energy -= energy_transfer
                offspring.energy = energy_transfer
                
                # Add offspring to world
                world.add_organism(offspring)
                
                success_value += 1.0  # Reproduction is highest success
            else:
                # Failed reproduction attempt
                self.energy -= 5
                success_value -= 0.3
        
        # Handle resting
        if actions.get("rest", False):
            # Resting recovers some energy and health
            rest_recovery = 2
            self.energy += rest_recovery
            self.health += rest_recovery * 0.5
            
            # Cap values
            self.energy = min(100, self.energy)
            self.health = min(100, self.health)
            
            success_value += 0.1  # Small success for resting
        
        # Adjust success based on energy change
        energy_change = self.energy - energy_before
        if energy_change > 0:
            success_value += min(0.3, energy_change / 30)  # Reward energy gain
        elif energy_change < -10:  # Only penalize significant energy loss
            success_value -= min(0.2, abs(energy_change) / 50)
            
        return max(-1.0, min(1.0, success_value))  # Clamp between -1 and 1
    
    def update_memory(self, sensory_data, actions, success_level):
        """Update organism memory with recent experiences"""
        # Create memory entry
        memory_entry = {
            "sensory_data": sensory_data.copy(),
            "actions": actions.copy(),
            "success": success_level,
            "age": self.age
        }
        
        # Add to memory (limit size)
        self.memory.append(memory_entry)
        if len(self.memory) > 10:  # Keep only 10 most recent memories
            self.memory.pop(0)
    
    def reproduce_with(self, partner, world):
        """Sexual reproduction with another organism"""
        if self.energy < 50 or partner.energy < 50:
            return None  # Not enough energy
            
        # Create offspring position
        offspring_position = (
            (self.position[0] + partner.position[0]) / 2 + (random.random() * 0.1 - 0.05),
            (self.position[1] + partner.position[1]) / 2 + (random.random() * 0.1 - 0.05)
        )
        
        # Create new genome combining parent genomes
        offspring_genome = self.neural_genome.combine_with(partner.neural_genome)
        
        # Create new organism
        offspring = Organism(offspring_position, offspring_genome)
        
        # Transfer energy from parents
        energy_from_self = self.energy * 0.2
        energy_from_partner = partner.energy * 0.2
        self.energy -= energy_from_self
        partner.energy -= energy_from_partner
        offspring.energy = energy_from_self + energy_from_partner
        
        # Add to world
        world.add_organism(offspring)
        return offspring


class NeuralGenome:
    def __init__(self, hidden_layers=None):
        """Initialize neural network genome"""
        self.input_nodes = []
        self.hidden_layers = hidden_layers if hidden_layers else [8, 4]  # Default architecture
        self.output_nodes = []
        self.weights = {}
        self.biases = {}
        self.learning_rate = 0.1
        
        # Configure default nodes
        self.configure_default_nodes()
        
        # Initialize weights and biases randomly
        self.initialize_weights()
        
    def configure_default_nodes(self):
        """Set up default input and output nodes"""
        # Input nodes
        self.input_nodes = [
            "energy_level", "hydration_level", "health_level",
            "food_distance", "food_direction",
            "water_distance", "water_direction",
            "organism_distance", "organism_direction", "organism_size"
        ]
        
        # Output nodes
        self.output_nodes = [
            "m_move_x", "m_move_y",  # Movement components
            "m_eat", "m_drink",       # Resource gathering
            "m_attack", "m_flee",     # Combat behaviors
            "m_reproduce", "m_rest"   # Reproduction and conservation
        ] 
        
    def initialize_weights(self):
        """Initialize the neural network weights and biases randomly"""
        import random
        
        # Create a full network specification with layers
        network_structure = [len(self.input_nodes)] + self.hidden_layers + [len(self.output_nodes)]
        
        # Initialize weights between all layers
        for l in range(len(network_structure) - 1):
            for i in range(network_structure[l]):
                for j in range(network_structure[l + 1]):
                    # Weight key format: "from_layer:from_node:to_layer:to_node"
                    key = f"{l}:{i}:{l+1}:{j}"
                    self.weights[key] = random.uniform(-1.0, 1.0)
        
        # Initialize biases for all nodes except input layer
        for l in range(1, len(network_structure)):
            for i in range(network_structure[l]):
                # Bias key format: "layer:node"
                key = f"{l}:{i}"
                self.biases[key] = random.uniform(-0.5, 0.5)
        
    def process_inputs(self, input_data):
        """Process inputs through the neural network to get outputs"""
        # Convert input_data from dictionary to array
        inputs = [input_data.get(node, 0.0) for node in self.input_nodes]
        
        # Create a list of layers starting with inputs
        layers = [inputs]
        
        # Process through hidden layers
        for l in range(len(self.hidden_layers)):
            layer_num = l + 1  # Layer 0 is input layer
            layer_size = self.hidden_layers[l]
            
            # Create new layer
            new_layer = []
            
            # Calculate each node in this layer
            for j in range(layer_size):
                # Sum weighted inputs plus bias
                node_sum = 0.0
                
                # Add weighted connections from previous layer
                for i in range(len(layers[-1])):
                    weight_key = f"{layer_num-1}:{i}:{layer_num}:{j}"
                    node_sum += layers[-1][i] * self.weights.get(weight_key, 0.0)
                
                # Add bias
                bias_key = f"{layer_num}:{j}"
                node_sum += self.biases.get(bias_key, 0.0)
                
                # Apply activation function (ReLU)
                activated = max(0, node_sum)
                new_layer.append(activated)
            
            # Add this layer to our list
            layers.append(new_layer)
        
        # Process final output layer
        output_layer_num = len(self.hidden_layers) + 1
        output_layer = []
        
        for j in range(len(self.output_nodes)):
            # Sum weighted inputs plus bias
            node_sum = 0.0
            
            # Add weighted connections from previous layer
            for i in range(len(layers[-1])):
                weight_key = f"{output_layer_num-1}:{i}:{output_layer_num}:{j}"
                node_sum += layers[-1][i] * self.weights.get(weight_key, 0.0)
            
            # Add bias
            bias_key = f"{output_layer_num}:{j}"
            node_sum += self.biases.get(bias_key, 0.0)
            
            # Apply sigmoid activation for outputs
            activated = 1.0 / (1.0 + math.exp(-node_sum))
            output_layer.append(activated)
        
        # Convert output array to dictionary
        outputs = {self.output_nodes[i]: output_layer[i] for i in range(len(self.output_nodes))}
        
        return outputs
    
    def learn_from_feedback(self, success_level):
        """Simple reinforcement learning from success feedback"""
        # This is a simplified version - a real implementation would use
        # backpropagation or other proper learning algorithms
        if abs(success_level) < 0.1:
            # Small success level doesn't trigger learning
            return
            
        # Adjust all weights slightly in the direction of success
        for key in self.weights:
            # Add small random adjustment weighted by success
            self.weights[key] += random.uniform(-0.1, 0.1) * success_level * self.learning_rate
            
        # Similarly for biases
        for key in self.biases:
            self.biases[key] += random.uniform(-0.1, 0.1) * success_level * self.learning_rate
    
    def clone_with_mutations(self):
        """Create a copy of this genome with mutations"""
        clone = NeuralGenome(self.hidden_layers)
        
        # Copy and mutate weights
        for key, value in self.weights.items():
            # 20% chance of mutation per weight
            if random.random() < 0.2:
                # Apply mutation
                mutation = random.uniform(-0.2, 0.2)
                clone.weights[key] = value + mutation
            else:
                clone.weights[key] = value
                
        # Copy and mutate biases
        for key, value in self.biases.items():
            # 20% chance of mutation per bias
            if random.random() < 0.2:
                # Apply mutation
                mutation = random.uniform(-0.2, 0.2)
                clone.biases[key] = value + mutation
            else:
                clone.biases[key] = value
                
        return clone
    
    def combine_with(self, other_genome):
        """Combine this genome with another through crossover"""
        # Create a new genome with same architecture
        child = NeuralGenome(self.hidden_layers)
        
        # Crossover weights
        for key in self.weights:
            if key in other_genome.weights:
                # 50% chance to inherit from each parent
                if random.random() < 0.5:
                    child.weights[key] = self.weights[key]
                else:
                    child.weights[key] = other_genome.weights[key]
                    
                # Small chance of mutation
                if random.random() < 0.1:
                    child.weights[key] += random.uniform(-0.1, 0.1)
        
        # Crossover biases
        for key in self.biases:
            if key in other_genome.biases:
                # 50% chance to inherit from each parent
                if random.random() < 0.5:
                    child.biases[key] = self.biases[key]
                else:
                    child.biases[key] = other_genome.biases[key]
                    
                # Small chance of mutation
                if random.random() < 0.1:
                    child.biases[key] += random.uniform(-0.1, 0.1)
                    
        return child


class World:
    def __init__(self, width=1.0, height=1.0):
        """Initialize the simulation world"""
        self.width = width
        self.height = height
        self.organisms = []
        self.resources = []
        self.terrain_map = {}
        self.time = 0
        
        # Initialize terrain
        self.generate_terrain()
        
        # Add initial resources
        self.generate_resources()
    
    def generate_terrain(self):
        """Generate terrain for the world"""
        # Simple terrain generation
        terrain_types = ["plain", "forest", "mountain", "desert", "water"]
        
        # Create a grid of terrain
        grid_size = 0.1  # Size of each terrain cell
        
        for x in range(int(self.width / grid_size)):
            for y in range(int(self.height / grid_size)):
                # Select a terrain type with some neighborhood coherence
                if random.random() < 0.7 and x > 0:
                    # 70% chance to use the terrain to the left for continuity
                    left_pos = (x-1, y)
                    terrain = self.terrain_map.get(left_pos, random.choice(terrain_types))
                else:
                    terrain = random.choice(terrain_types)
                
                self.terrain_map[(x, y)] = terrain
    
    def get_terrain_at(self, position):
        """Get terrain type at a specific position"""
        grid_size = 0.1
        grid_x = int(position[0] / grid_size)
        grid_y = int(position[1] / grid_size)
        
        # Keep within bounds
        grid_x = max(0, min(int(self.width / grid_size) - 1, grid_x))
        grid_y = max(0, min(int(self.height / grid_size) - 1, grid_y))
        
        return self.terrain_map.get((grid_x, grid_y), "plain")
    
    def generate_resources(self):
        """Generate initial resources in the world"""
        # Add food sources
        for _ in range(50):
            self.resources.append({
                "type": "food",
                "position": (random.random() * self.width, random.random() * self.height),
                "energy": 20 + random.random() * 30
            })
            
        # Add water sources
        for _ in range(20):
            self.resources.append({
                "type": "water",
                "position": (random.random() * self.width, random.random() * self.height),
                "amount": 50 + random.random() * 100
            })
    
    def update(self):
        """Update the world for one time step"""
        self.time += 1
        
        # Update all organisms
        for organism in list(self.organisms):  # Copy list to allow modification during iteration
            if not organism.update(self):
                # Organism died, remove it
                self.organisms.remove(organism)
        
        # Replenish resources occasionally
        if self.time % 20 == 0:  # Every 20 time steps
            self.replenish_resources()
    
    def replenish_resources(self):
        """Add new resources to the world"""
        # Add some new food
        num_new_food = max(0, 50 - len([r for r in self.resources if r["type"] == "food"]))
        for _ in range(num_new_food // 10):  # Add 10% of deficit
            self.resources.append({
                "type": "food",
                "position": (random.random() * self.width, random.random() * self.height),
                "energy": 20 + random.random() * 30
            })
            
        # Add some new water
        num_new_water = max(0, 20 - len([r for r in self.resources if r["type"] == "water"]))
        for _ in range(num_new_water // 5):  # Add 20% of deficit
            self.resources.append({
                "type": "water",
                "position": (random.random() * self.width, random.random() * self.height),
                "amount": 50 + random.random() * 100
            })