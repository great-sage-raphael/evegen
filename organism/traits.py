def calculate_traits(gene_expressions):
    return {
        "speed": gene_expressions.get("A", 0.5),
        "vision": gene_expressions.get("B", 0.5),
        "aggression": gene_expressions.get("C", 0.5),
        "metabolism": gene_expressions.get("D", 0.5)
    }
