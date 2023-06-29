import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def compute_childgene_probs(inherit_prob):
    """
    ChildGene node is conditional on the number of copies of the gene inhirited from parents
    """
    # probs table has 9 rows and 3 columns. Each row represents a combination of mother & father genes.
    # The columns represent the number of copies of the gene the child has.
    num_rows = 3 * 3  # number of mother & father gene combinations
    num_cols = 3  # number of columns in probs table
    probs = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    for i in range(len(probs)):
        mother_gene = i // 3
        father_gene = i % 3
        probs[i][0] = inherit_prob[mother_gene][0] * \
            inherit_prob[father_gene][0]
        probs[i][2] = inherit_prob[mother_gene][1] * \
            inherit_prob[father_gene][1]
        probs[i][1] = 1 - (probs[i][0] + probs[i][2])
    return probs


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # conditional probability that a child inherits the gene from a parent
    # given the number of copies of the gene the parent has
    inherit_prob = {
        0: {
            0: 1 - PROBS["mutation"],
            1: PROBS["mutation"]
        },
        1: {
            0: 0.5,
            1: 0.5
        },
        2: {
            0: PROBS["mutation"],
            1: 1 - PROBS["mutation"]
        }
    }

    # conditional probability of a ChildGene node
    child_gene_probs = compute_childgene_probs(inherit_prob)

    # traverse each person in the family tree. Each person has 2 random variables - gene and trait to compute probabilities for.
    probability = 1

    for person, person_info in people.items():
        # get gene probabilities for the person
        gene_probs = {}
        if person_info["mother"] is None and person_info["father"] is None:
            gene_probs = PROBS["gene"]
        elif person_info["mother"] is not None and person_info["father"] is not None:
            mother_gene = 2 if person_info["mother"] in two_genes else 1 if person_info["mother"] in one_gene else 0
            father_gene = 2 if person_info["father"] in two_genes else 1 if person_info["father"] in one_gene else 0
            gene_probs = child_gene_probs[mother_gene * 3 + father_gene]
        else:
            raise Exception("Invalid family tree")

        # compute probability of the person having the gene and trait
        if person in two_genes:
            probability *= gene_probs[2]
            probability *= PROBS["trait"][2][True] if person in have_trait else PROBS["trait"][2][False]

        elif person in one_gene:
            probability *= gene_probs[1]
            probability *= PROBS["trait"][1][True] if person in have_trait else PROBS["trait"][1][False]

        else:
            probability *= gene_probs[0]
            probability *= PROBS["trait"][0][True] if person in have_trait else PROBS["trait"][0][False]

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person, person_info in probabilities.items():
        # gene
        person_info["gene"][1] += p if person in one_gene else 0
        person_info["gene"][2] += p if person in two_genes else 0
        person_info["gene"][0] += p if person not in one_gene and person not in two_genes else 0

        # trait
        person_info["trait"][True] += p if person in have_trait else 0
        person_info["trait"][False] += p if person not in have_trait else 0


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person, person_info in probabilities.items():
        # gene
        gene_sum = sum(person_info["gene"].values())
        for gene, gene_prob in person_info["gene"].items():
            probabilities[person]["gene"][gene] = gene_prob / gene_sum
        # trait
        trait_sum = sum(person_info["trait"].values())
        for trait, trait_prob in person_info["trait"].items():
            probabilities[person]["trait"][trait] = trait_prob / trait_sum


if __name__ == "__main__":
    main()
