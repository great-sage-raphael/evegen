# genome.py
import math
import random
from gene import Gene
class Genome:
    def __init__(self):
        self.genes = {}  # {gene_id: Gene object}
        # No predefined trait mapping
    
    def add_gene(self, gene_id=None, initial_value=None):
        """Add a new gene to the genome"""
        if gene_id is None:
            gene_id = 0 if not self.genes else max(self.genes.keys()) + 1
            
        self.genes[gene_id] = Gene(gene_id, initial_value)
        return self.genes[gene_id]
    
    def generate_random_genome(self, gene_count=20, connection_density=0.2):
        """Generate a random genome with connections"""
        # Create genes
        for i in range(gene_count):
            self.add_gene(i)
        
        # Add random connections
        for gene in self.genes.values():
            connection_count = int(gene_count * connection_density)
            potential_targets = list(self.genes.keys())
            targets = random.sample(potential_targets, min(connection_count, len(potential_targets)))
            
            for target in targets:
                gene.connect_to(target)
    
    def update_gene_values(self, iterations=3):
        """Update all gene values based on their connections"""
        for _ in range(iterations):
            for gene in self.genes.values():
                gene.update_value(self.genes)
    
    def get_output_values(self):
        """Get values from all genes with no outgoing connections (outputs)"""
        outputs = {}
        for gene_id, gene in self.genes.items():
            # Use gene ID as key for output
            outputs[f"output_{gene_id}"] = gene.value
        return outputs
    
    def copy(self):
        """Create a copy of this genome"""
        new_genome = Genome()
        
        # Copy genes
        for gene_id, gene in self.genes.items():
            new_gene = new_genome.add_gene(gene_id, gene.value)
            
            # Copy connections
            for target_id, weight in gene.connections.items():
                new_gene.connect_to(target_id, weight)
                
        return new_genome