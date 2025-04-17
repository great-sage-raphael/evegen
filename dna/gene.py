# gene.py
import math
import random

class Gene:
    def __init__(self, gene_id, initial_value=None):
        self.id = gene_id
        # Start with random values unless specified
        self.value = initial_value if initial_value is not None else random.random()
        self.connections = {}  # {gene_id: influence_weight}
    
    def connect_to(self, gene_id, weight=None):
        """Create connection to another gene with random or specific weight"""
        # Random weight if none provided
        if weight is None:
            weight = random.uniform(-1.0, 1.0)
        self.connections[gene_id] = weight
    
    def update_value(self, network):
        """Update gene value based on connected genes' influence"""
        if not self.connections:
            return self.value  # Value remains static if no connections
            
        influence_sum = 0
        for gene_id, weight in self.connections.items():
            if gene_id in network:
                influence_sum += network[gene_id].value * weight
        
        # Sigmoid function to keep values between 0-1
        new_value = 1 / (1 + math.exp(-influence_sum))
        
        # Introduce some noise/stochasticity
        new_value += random.uniform(-0.05, 0.05)
        self.value = max(0, min(1, new_value))  # Keep within bounds
        
        return self.value