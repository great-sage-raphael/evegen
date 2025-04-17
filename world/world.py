import random
import math
from math import dist as distance

class World:
    def __init__(self, width=1.0, height=1.0):
        self.width = width
        self.height = height
        self.time = 0
        self.day_cycle = 0  # 0-1 representing time of day
        self.seasons = 0    # 0-1 representing seasons
        
        # World resources
        self.resources = []  # Food, water, danger spots, etc.
        
        # Terrain and climate maps
        self.terrain_map = self._generate_terrain_map()
        self.temperature_map = self._generate_temperature_map()
        self.water_map = self._generate_water_map()
        
        # Organisms
        self.organisms = []
        
        # Initialize environment
        self._initialize_environment()
    
    def _generate_terrain_map(self):
        """Generate a procedural terrain map"""
        # Use simplex noise for natural-looking terrain
        # For simplicity, we'll just create a random map here
        terrain_types = ["plain", "forest", "mountain", "swamp"]
        map_size = 50  # 50x50 grid
        
        terrain = {}
        for x in range(map_size):
            for y in range(map_size):
                # Coordinates normalized to world size
                nx = x / map_size * self.width
                ny = y / map_size * self.height
                terrain[(nx, ny)] = random.choice(terrain_types)
        
        return terrain
    
    def _generate_temperature_map(self):
        """Generate temperature variation across the map"""
        # Higher in the south, lower in the north (simplified)
        map_size = 50
        temp_map = {}
        
        for x in range(map_size):
            for y in range(map_size):
                nx = x / map_size * self.width
                ny = y / map_size * self.height
                
                # Basic temperature gradient + noise
                base_temp = 20 + (ny * 30)  # 20-50°C range
                variation = random.uniform(-5, 5)
                temp_map[(nx, ny)] = base_temp + variation
        
        return temp_map
    
    def _generate_water_map(self):
        """Generate water sources and rivers"""
        water_map = {}
        map_size = 50
        
        # Place some water bodies
        water_bodies = random.randint(3, 7)
        for _ in range(water_bodies):
            center_x = random.randint(0, map_size-1)
            center_y = random.randint(0, map_size-1)
            radius = random.randint(2, 5)
            
            # Create water body
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:
                        x = center_x + dx
                        y = center_y + dy
                        if 0 <= x < map_size and 0 <= y < map_size:
                            nx = x / map_size * self.width
                            ny = y / map_size * self.height
                            water_map[(nx, ny)] = 1.0  # Water level
        
        return water_map
    
    def _initialize_environment(self):
        """Set up initial resources and conditions"""
        # Add food sources
        for _ in range(50):
            x = random.random() * self.width
            y = random.random() * self.height
            
            # Food value varies by terrain
            terrain = self.get_terrain_at((x, y))
            if terrain == "forest":
                energy = random.randint(30, 50)
            elif terrain == "plain":
                energy = random.randint(20, 40)
            else:
                energy = random.randint(10, 30)
                
            self.resources.append({
                "type": "food",
                "position": (x, y),
                "energy": energy,
                "created_at": self.time
            })
        
        # Add water sources near water bodies
        for pos in self.water_map:
            if random.random() < 0.1:  # Don't place too many
                self.resources.append({
                    "type": "water",
                    "position": pos,
                    "amount": random.randint(50, 100),
                    "created_at": self.time
                })
        
        # Add danger spots (predators, toxic areas, etc.)
        for _ in range(10):
            x = random.random() * self.width
            y = random.random() * self.height
            
            self.resources.append({
                "type": "danger",
                "position": (x, y),
                "damage": random.randint(10, 30),
                "created_at": self.time
            })
    
    def update(self):
        """Update world state for one time step"""
        self.time += 1
        
        # Update day/night cycle
        self.day_cycle = (self.day_cycle + 0.01) % 1.0
        
        # Update seasons very slowly
        self.seasons = (self.seasons + 0.001) % 1.0
        
        # Update organisms
        self._update_organisms()
        
        # Update resources
        self._update_resources()
        
        # Environmental events
        self._handle_environmental_events()
    
    def _update_organisms(self):
        """Update all organisms"""
        for organism in self.organisms[:]:  # Copy to allow removal
            # Organism updates itself and returns whether still alive
            if not organism.update(self):
                self.organisms.remove(organism)
    
    def _update_resources(self):
        """Update, add, and remove resources"""
        # Remove old resources
        self.resources = [r for r in self.resources if (self.time - r["created_at"]) < 300]
        
        # Add new resources periodically
        if self.time % 20 == 0:
            self._add_new_resources()
    
    def _add_new_resources(self):
        """Add new resources to the world"""
        # Add food - more in appropriate season
        food_chance = 0.3 + 0.2 * math.sin(self.seasons * 2 * math.pi)  # 0.1-0.5
        
        if random.random() < food_chance:
            for _ in range(random.randint(5, 15)):
                x = random.random() * self.width
                y = random.random() * self.height
                
                # Food depends on terrain and season
                terrain = self.get_terrain_at((x, y))
                season_factor = math.sin(self.seasons * 2 * math.pi)  # -1 to 1
                
                if terrain == "forest":
                    energy = random.randint(20, 40) + int(10 * season_factor)
                elif terrain == "plain":
                    energy = random.randint(15, 35) + int(5 * season_factor)
                else:
                    energy = random.randint(10, 25) + int(3 * season_factor)
                
                self.resources.append({
                    "type": "food",
                    "position": (x, y),
                    "energy": max(10, energy),  # Minimum energy value
                    "created_at": self.time
                })
    
    def _handle_environmental_events(self):
        """Handle random environmental events"""
        # Rare catastrophic events
        if random.random() < 0.001:  # 0.1% chance per step
            event_type = random.choice(["drought", "flood", "cold_snap", "heat_wave"])
            
            if event_type == "drought":
                # Reduce water levels
                for resource in self.resources:
                    if resource["type"] == "water":
                        resource["amount"] = max(10, resource["amount"] - random.randint(10, 30))
            
            elif event_type == "flood":
                # Increase water, but damage some food
                for resource in self.resources:
                    if resource["type"] == "water":
                        resource["amount"] += random.randint(20, 50)
                    elif resource["type"] == "food" and random.random() < 0.3:
                        self.resources.remove(resource)
            
            elif event_type == "cold_snap":
                # Temporarily decrease temperatures
                for key in self.temperature_map:
                    self.temperature_map[key] -= random.uniform(5, 15)
            
            elif event_type == "heat_wave":
                # Temporarily increase temperatures
                for key in self.temperature_map:
                    self.temperature_map[key] += random.uniform(5, 15)
    
    def get_terrain_at(self, position):
        """Get terrain type at the given position"""
        # Find closest grid point
        closest_pos = min(self.terrain_map.keys(), 
                         key=lambda p: distance(p, position))
        return self.terrain_map[closest_pos]
    
    def get_temperature_at(self, position):
        """Get temperature at the given position"""
        # Find closest grid point
        closest_pos = min(self.temperature_map.keys(), 
                         key=lambda p: distance(p, position))
        
        # Base temperature from map
        base_temp = self.temperature_map[closest_pos]
        
        # Adjust for time of day
        time_factor = math.sin(self.day_cycle * 2 * math.pi)  # -1 to 1
        day_night_variation = 10 * time_factor  # ±10°C
        
        # Adjust for season
        season_factor = math.sin(self.seasons * 2 * math.pi)  # -1 to 1
        season_variation = 15 * season_factor  # ±15°C
        
        return base_temp + day_night_variation + season_variation
    
    def get_aridity_at(self, position):
        """Get aridity (0-1) at position, higher means more water loss"""
        # Check if position is in water
        for pos, level in self.water_map.items():
            if distance(pos, position) < 0.05:
                return 0.0  # No water loss in water
        
        # Base aridity from terrain
        terrain = self.get_terrain_at(position)
        if terrain == "forest":
            base_aridity = 0.3
        elif terrain == "plain":
            base_aridity = 0.5
        elif terrain == "mountain":
            base_aridity = 0.7
        elif terrain == "swamp":
            base_aridity = 0.1
        else:
            base_aridity = 0.5
        
        # Adjust for temperature
        temp = self.get_temperature_at(position)
        temp_factor = max(0, min(1, (temp - 15) / 35))  # 0-1 scale
        
        # Adjust for time of day
        time_factor = self.day_cycle  # 0-1
        day_aridity = 0.2 if time_factor < 0.5 else 0.0  # Higher during day
        
        return min(1.0, base_aridity + temp_factor * 0.5 + day_aridity)
    
    def get_light_at(self, position):
        """Get light level (0-1) at position"""
        # Day/night cycle determines base light
        if self.day_cycle < 0.25:  # Dawn
            base_light = self.day_cycle * 4  # 0-1
        elif self.day_cycle < 0.75:  # Day
            base_light = 1.0
        else:  # Dusk/night
            base_light = max(0, 1 - (self.day_cycle - 0.75) * 4)  # 1-0
        
        # Adjust for terrain (forests are darker)
        terrain = self.get_terrain_at(position)
        if terrain == "forest":
            terrain_factor = 0.7  # 30% darker in forests
        else:
            terrain_factor = 1.0
            
        return base_light * terrain_factor
    
    def get_resources_in_radius(self, position, radius):
        """Find resources within given radius of position"""
        nearby = []
        for resource in self.resources:
            if distance(position, resource["position"]) <= radius:
                nearby.append(resource)
        return nearby
    
    def raycast(self, start_pos, end_pos):
        """Find what's along a ray from start to end position"""
        # Simple implementation - check resources along line
        direction = (
            end_pos[0] - start_pos[0],
            end_pos[1] - start_pos[1]
        )
        dir_length = math.sqrt(direction[0]**2 + direction[1]**2)
        
        if dir_length == 0:
            return []
            
        # Normalize direction
        direction = (direction[0]/dir_length, direction[1]/dir_length)
        
        # Check points along ray
        found_objects = []
        step_size = 0.01
        max_steps = int(dir_length / step_size)
        
        for step in range(1, max_steps + 1):
            check_pos = (
                start_pos[0] + direction[0] * step * step_size,
                start_pos[1] + direction[1] * step * step_size
            )
            
            # Check for resources
            for resource in self.resources:
                if distance(check_pos, resource["position"]) < 0.02:
                    found_objects.append({
                        "type": resource["type"],
                        "distance": step * step_size,
                        "object": resource
                    })
                    break
            
            # If something found, stop raycast
            if found_objects:
                break
        
        return found_objects
    