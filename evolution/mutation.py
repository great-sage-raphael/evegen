import random
# mutation.py
def mutate(genome, mutation_rate=0.05):
    """Apply random mutations to a genome"""
    genome_changed = False
    
    # 1. Gene value mutations
    for gene in genome.genes.values():
        if random.random() < mutation_rate:
            # Alter gene value slightly
            mutation_amount = random.gauss(0, 0.1)  # Normal distribution
            gene.value += mutation_amount
            gene.value = max(0, min(1, gene.value))  # Keep within bounds
            genome_changed = True
    
    # 2. Connection mutations
    for gene in genome.genes.values():
        # Add new connection
        if random.random() < mutation_rate / 2:
            target_id = random.choice(list(genome.genes.keys()))
            weight = random.gauss(0, 0.5)  # Normal distribution centered at 0
            gene.connect_to(target_id, weight)
            genome_changed = True
        
        # Modify existing connections
        for target_id in list(gene.connections.keys()):
            if random.random() < mutation_rate:
                # Change connection weight
                weight_change = random.gauss(0, 0.2)  # Normal distribution
                gene.connections[target_id] += weight_change
                genome_changed = True
                
                # Sometimes delete connections
                if random.random() < 0.1:
                    del gene.connections[target_id]
                    genome_changed = True
    
    # 3. Gene duplication - a key evolutionary mechanism
    if random.random() < mutation_rate / 10:
        if genome.genes:
            # Select a gene to duplicate
            source_id = random.choice(list(genome.genes.keys()))
            source_gene = genome.genes[source_id]
            
            # Create new gene with similar properties
            new_id = max(genome.genes.keys()) + 1
            new_gene = genome.add_gene(new_id, source_gene.value)
            
            # Copy connections
            for target_id, weight in source_gene.connections.items():
                # Add some variation
                new_weight = weight + random.gauss(0, 0.1)
                new_gene.connect_to(target_id, new_weight)
            
            genome_changed = True
    
    # 4. Rarely add completely new genes
    if random.random() < mutation_rate / 20:
        new_id = max(genome.genes.keys()) + 1 if genome.genes else 0
        new_gene = genome.add_gene(new_id, random.random())
        
        # Connect to some existing genes
        connect_count = random.randint(1, 3)
        if genome.genes:
            for _ in range(connect_count):
                target_id = random.choice(list(genome.genes.keys()))
                weight = random.gauss(0, 0.5)
                new_gene.connect_to(target_id, weight)
        
        genome_changed = True
    
    return genome_changed