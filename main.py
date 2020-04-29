import torch
from itertools import permutations, repeat, chain
from nsm import NSM, dummy

EMBD_DIM = 7
OUT_DIM = 4
BATCH = 10
N = 3

def embed(x):
    return dummy(x, EMBD_DIM)

if __name__ == "__main__":

    property_types = ['color', 'material']
    property_concepts = {
        'color': ['red', 'green', 'blue'],
        'material': ['cloth', 'rubber']
    }
    L = len(property_types)

    state_identities = ['cat', 'shirt']
    relationships = ['holding', 'behind']

    # extra identity connections
    property_types = ['identity'] + property_types
    property_types += ['relations']
    property_concepts['identity'] = state_identities
    property_concepts['relations'] = relationships

    D = torch.stack(list(map(embed, property_types)))

    ordered_C = [
        torch.stack(
            [embed(concept)
             for concept in property_concepts[property]]
        )
        for property in property_types
    ]

    C = torch.cat(ordered_C, dim=0)

    c_prime = torch.rand(1, EMBD_DIM, requires_grad=True)
    C = torch.cat([C, c_prime], dim=0)

    nodes = ['kitten', 'person', 'shirt']

    relations = {
        ('person', 'shirt'): 'wear',
        ('person', 'kitten'): 'holding',
        ('kitten', 'shirt'): 'bite'
    }

    S = torch.rand(BATCH, len(nodes), L+1, EMBD_DIM)

    adjacency_mask = torch.zeros(BATCH, len(nodes), len(nodes))
    E = torch.zeros(BATCH, len(nodes), len(nodes), EMBD_DIM)

    for idx_pair in permutations(range(len(nodes)), 2):
        pair = tuple(nodes[idx] for idx in idx_pair)
        if pair in relations:
            E[:, idx_pair[0], idx_pair[1]] = torch.rand(EMBD_DIM)   # (TODO)
            adjacency_mask[:, idx_pair[0], idx_pair[1]] = 1

    questions = list(repeat(['what', 'color', 'is', 'the', 'cat'], BATCH))

    nsm = NSM(nodes, EMBD_DIM, OUT_DIM, BATCH, N, L)
    output = nsm(questions, C, D, E, S, adjacency_mask)

    print(output)
