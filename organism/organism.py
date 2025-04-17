import random
import math
class Organism:
    def __init__(self, genome=None, position=None):
        # Base attributes
        self.genome = genome if genome else self.generate_default_genome()
        self.position = position if position else (random.random(), random.random())
        
        # Internal states - must be maintained within ranges
        self.energy = 100        # Energy reserve
        self.hydration = 100     # Water level
        self.integrity = 100     # Physical health/integrity
        self.temperature = 37    # Internal temperature (°C)
        
        # Memory & experience
        self.memory = []         # Short-term memory buffer
        self.neural_state = {}   # Current activation state of neural network
        
        # Life tracking
        self.age = 0
        self.alive = True
        self.cause_of_death = None
        
    def update_internal_states(self, world):
        """Update all internal states based on environment and actions"""
        # Energy metabolism - baseline cost
        base_metabolism = 1.0
        
        # Adjust metabolism based on size (from genome)
        size_factor = self.calculate_trait("size")
        metabolism_cost = base_metabolism * (0.5 + size_factor)  # Larger = more energy use
        
        # Adjust metabolism based on temperature differential
        world_temp = world.get_temperature_at(self.position)
        temp_diff = abs(self.temperature - world_temp)
        temp_regulation_cost = temp_diff * 0.1
        
        # Apply metabolism costs
        self.energy -= (metabolism_cost + temp_regulation_cost)
        
        # Hydration decreases over time
        self.hydration -= 0.5 + (world.get_aridity_at(self.position) * 0.5)
        
        # Integrity regeneration if energy available
        if self.energy > 50 and self.integrity < 100:
            repair_amount = 0.1
            self.integrity = min(100, self.integrity + repair_amount)
            self.energy -= repair_amount * 2  # Repair costs energy
        
        # Temperature regulation (homeostasis)
        if world_temp != self.temperature:
            # Gradual temperature adjustment toward environment
            adjustment = (world_temp - self.temperature) * 0.1
            self.temperature += adjustment
        
        # Check vital signs
        self.check_vitals(world)
    
    def check_vitals(self, world):
        """Check if organism is still alive based on internal states"""
        if self.energy <= 0:
            self.alive = False
            self.cause_of_death = "starvation"
        elif self.hydration <= 0:
            self.alive = False
            self.cause_of_death = "dehydration"
        elif self.integrity <= 0:
            self.alive = False 
            self.cause_of_death = "injury"
        elif self.temperature < 10 or self.temperature > 45:
            self.alive = False
            self.cause_of_death = "temperature_extreme"
        
        return self.alive
def sense_environment(self, world):
    """Gather comprehensive information about surroundings"""
    sensory_data = {}
    
    # Vision - separate into sectors
    vision_range = self.calculate_trait("vision_range") * 0.2  # 0-0.2 units
    vision_sectors = 8  # 8 directional sectors
    
    for sector in range(vision_sectors):
        angle = sector * (2 * math.pi / vision_sectors)
        # Look in this direction
        sensory_data[f"vision_{sector}"] = self.look_in_direction(world, angle, vision_range)
    
    # Smell - detect resources
    smell_range = self.calculate_trait("smell_sensitivity") * 0.15
    nearby_resources = world.get_resources_in_radius(self.position, smell_range)
    
    sensory_data["smell_food"] = sum(1 for r in nearby_resources if r["type"] == "food")
    sensory_data["smell_water"] = sum(1 for r in nearby_resources if r["type"] == "water")
    sensory_data["smell_danger"] = sum(1 for r in nearby_resources if r["type"] == "danger")
    
    # Internal state sensing
    sensory_data["energy_level"] = self.energy / 100.0  # 0-1 normalized
    sensory_data["hydration_level"] = self.hydration / 100.0
    sensory_data["integrity_level"] = self.integrity / 100.0
    sensory_data["temperature_feel"] = (self.temperature - 20) / 30.0  # Normalized around optimal
    
    # Environmental sensing
    sensory_data["ground_type"] = world.get_terrain_at(self.position)
    sensory_data["ambient_temp"] = world.get_temperature_at(self.position) / 50.0  # Normalized
    sensory_data["light_level"] = world.get_light_at(self.position)
    
    return sensory_data

def look_in_direction(self, world, angle, range_limit):
    """Look in a specific direction and return what's seen"""
    # Calculate endpoint of vision ray
    end_x = self.position[0] + math.cos(angle) * range_limit
    end_y = self.position[1] + math.sin(angle) * range_limit
    
    # Check what's along this ray (simplified)
    found_objects = world.raycast(self.position, (end_x, end_y))
    
    if found_objects:
        closest = found_objects[0]  # Get closest object
        return closest["type"]  # Return type of object seen
    else:
        return "nothing"
    


#////////////////////////////////

class NeuralGenome:
    def __init__(self):
        # Layers of the neural network
        self.layers = {
            "sensory": {},       # Input neurons
            "association": {},   # Middle layer neurons
            "motor": {}          # Output neurons
        }
        self.connections = []    # Connections between neurons
        self.plasticity = 0.01   # How quickly connections strengthen/weaken
    
    def add_neuron(self, layer, neuron_id, activation=0.0):
        """Add a neuron to specified layer"""
        if layer in self.layers:
            self.layers[layer][neuron_id] = {
                "activation": activation,
                "threshold": random.uniform(0.2, 0.8),
                "decay": random.uniform(0.1, 0.5)
            }
    
    def add_connection(self, from_layer, from_id, to_layer, to_id, weight=None):
        """Connect two neurons"""
        if weight is None:
            weight = random.uniform(-1.0, 1.0)
            
        self.connections.append({
            "from_layer": from_layer,
            "from_id": from_id,
            "to_layer": to_layer,
            "to_id": to_id,
            "weight": weight,
            "strength": 1.0  # Connection strength (for learning)
        })
    
    def process_inputs(self, inputs):
        """Process sensory inputs through the neural network"""
        # Set sensory neuron activations from inputs
        for input_name, value in inputs.items():
            if input_name in self.layers["sensory"]:
                self.layers["sensory"][input_name]["activation"] = value
        
        # Process association layer
        self._process_layer("sensory", "association")
        
        # Process motor layer
        self._process_layer("association", "motor")
        
        # Return motor neuron activations as outputs
        return {n_id: neuron["activation"] 
                for n_id, neuron in self.layers["motor"].items()}
    
    def _process_layer(self, from_layer, to_layer):
        """Process signals from one layer to the next"""
        # Calculate input for each target neuron
        target_inputs = {n_id: 0.0 for n_id in self.layers[to_layer]}
        
        # Sum weighted inputs
        for conn in self.connections:
            if conn["from_layer"] == from_layer and conn["to_layer"] == to_layer:
                from_neuron = self.layers[from_layer].get(conn["from_id"])
                if from_neuron:
                    activation = from_neuron["activation"]
                    weighted_signal = activation * conn["weight"] * conn["strength"]
                    target_inputs[conn["to_id"]] += weighted_signal
        
        # Apply activation to each target neuron
        for n_id, input_sum in target_inputs.items():
            neuron = self.layers[to_layer][n_id]
            if input_sum >= neuron["threshold"]:
                # Fire the neuron
                neuron["activation"] = 1.0
            else:
                # Apply decay
                neuron["activation"] *= (1.0 - neuron["decay"])
    
    def learn_from_feedback(self, reward):
        """Update connection strengths based on feedback"""
        for conn in self.connections:
            from_neuron = self.layers[conn["from_layer"]].get(conn["from_id"])
            to_neuron = self.layers[conn["to_layer"]].get(conn["to_id"])
            
            if from_neuron and to_neuron:
                # Only strengthen connections between active neurons
                if from_neuron["activation"] > 0.5 and to_neuron["activation"] > 0.5:
                    # Positive reward strengthens connection
                    if reward > 0:
                        conn["strength"] = min(2.0, conn["strength"] + reward * self.plasticity)
                    # Negative reward weakens connection
                    elif reward < 0:
                        conn["strength"] = max(0.1, conn["strength"] + reward * self.plasticity)
    
    def generate_random(self, sensory_count=20, association_count=30, motor_count=10):
        """Generate a random neural genome"""
        # Create neurons in each layer
        for i in range(sensory_count):
            self.add_neuron("sensory", f"s{i}")
        
        for i in range(association_count):
            self.add_neuron("association", f"a{i}")
        
        for i in range(motor_count):
            self.add_neuron("motor", f"m{i}")
        
        # Create random connections
        # Sensory to association
        for s_id in self.layers["sensory"]:
            # Connect to random subset of association neurons
            for a_id in random.sample(list(self.layers["association"].keys()), 
                                     k=random.randint(1, 5)):
                self.add_connection("sensory", s_id, "association", a_id)
        
        # Association to motor
        for a_id in self.layers["association"]:
            # Connect to random subset of motor neurons
            for m_id in random.sample(list(self.layers["motor"].keys()),
                                    k=random.randint(1, 3)):
                self.add_connection("association", a_id, "motor", m_id)
        
        # Some association to association connections (recurrence)
        for _ in range(association_count * 2):
            from_id = random.choice(list(self.layers["association"].keys()))
            to_id = random.choice(list(self.layers["association"].keys()))
            if from_id != to_id:  # Avoid self-connections
                self.add_connection("association", from_id, "association", to_id)