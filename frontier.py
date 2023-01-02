import jax
import jax.numpy as jnp
import jax.lax as jl
from Bio import SeqIO

# def read_seqs(path):
#     with open(path) as f:
#         records = list(SeqIO.parse(f, "fasta"))

#     seqs = []
#     lens = []
#     for rec in records:
#         seq = []
#         for c in rec.seq:
#             idx = ord(c) - ord('A')
#             seq.append(jnp.zeros((26,)).at[idx].set(1.0))
#         seqs.append(jnp.vstack(seq))
#         lens.append(len(rec.seq))
#     max_len = max(lens)

#     return jnp.stack([jl.pad(seq, -1., [(0, max_len - seq.shape[0], 0), (0,0,0)]) for seq in seqs], 0), lens

def read_profs(path):
    with open(path) as f:
        records = list(SeqIO.parse(f, "fasta"))

    profs = []
    for rec in records:
        prof = []
        for c in rec.seq:
            idx = ord(c) - ord('A')
            prof.append(jnp.zeros((26,)).at[idx].set(1.0))
        profs.append(jnp.vstack(prof))

    return profs

def _front_cells(k, seq_idx, dims):
    if 0 <= k < dims[0]:
        min_dim = jnp.minimum(dims[0], dims[1])
        return jnp.stack([
            jnp.full(jnp.minimum(k + 1, min_dim), seq_idx),
            jnp.arange(k, jnp.maximum(k - min_dim, -1), -1),
            jnp.arange(0, jnp.minimum(k + 1, min_dim), 1),
        ], 1)
    elif 0 <= k < (dims[0] + dims[1] - 1):
        i = k - dims[0] + 1
        steps = dims[1] - i
        return jnp.stack([
            jnp.full((jnp.minimum(dims[0], steps),), seq_idx),
            jnp.arange(dims[0] - 1, jnp.maximum(-1, dims[0] - 1 - steps), -1),
            jnp.arange(i, i + jnp.minimum(dims[0], steps), 1),
        ], 1)

    raise ValueError(f"requested front {k} out of bounds")

def _get_cell_dep(cell, s):
    cell_upleft = cell.at[1:3].add(-1)
    dep_upleft = jl.cond(
        (cell_upleft[1] >= 0) & (cell_upleft[2] >= 0),
        lambda cell, s: jnp.max(s[cell[0], cell[1], cell[2]]),
        lambda cell, s: -jnp.inf,
        cell_upleft, s
    )
    cell_up = cell.at[1].add(-1)
    dep_up = jl.cond(
        cell_up[1] >= 0,
        lambda cell, s: jnp.max(s[cell[0], cell[1], cell[2]]),
        lambda cell, s: -jnp.inf,
        cell_up, s
    )
    cell_left = cell.at[2].add(-1)
    dep_left = jl.cond(
        cell_left[2] >= 0,
        lambda cell, s: jnp.max(s[cell[0], cell[1], cell[2]]),
        lambda cell, s: -jnp.inf,
        cell_left, s
    )

    return jnp.stack([dep_upleft, dep_up, dep_left], 0)
_get_cells_dep = jax.vmap(_get_cell_dep, (0, None), 0)

def _resolve_cell(dep, cell, profs, score_mat, gap_penalty):
    def score_fn(profs, score_mat, cell):
        outer = jnp.outer(profs[cell[0], 0, cell[1]-1, :], profs[cell[0], 1, cell[2]-1, :])
        return jnp.sum(outer * score_mat)

    score = jl.cond(
        (cell[1] > 0) & (cell[2] > 0),
        score_fn,
        lambda profs, score_mat, cell: -jnp.inf,
        profs,
        score_mat,
        cell,
    )
    out = jnp.empty((3,)).at[:].set([dep[0] + score, dep[1] - gap_penalty, dep[2] - gap_penalty])

    return jl.cond(jnp.max(out) == -jnp.inf, (lambda x: x.at[:].set(0.)), (lambda x: x), operand=out)
_resolve_cells = jax.vmap(_resolve_cell, (0, 0, None, None, None), 0)

@jax.jit
def _resolve_cells_scatter(profs, score_mat, gap_penalty, cells, s):
    deps = _get_cells_dep(cells, s)
    outputs = _resolve_cells(deps, cells, profs, score_mat, gap_penalty)

    return s.at[cells[:, 0], cells[:, 1], cells[:, 2]].set(outputs)

def _combine_pairs(pairs):
    max_len = max(max(one.shape[0], two.shape[0]) for one, two in pairs)
    stacked = []
    for one, two in pairs:
        stacked.append(jnp.stack([
            jl.pad(one, -1., [(0, max_len - one.shape[0], 0), (0, 0, 0)]),
            jl.pad(two, -1., [(0, max_len - two.shape[0], 0), (0, 0, 0)]),
        ], 0))

    return jnp.stack(stacked, 0)


def needleman_wunsch(prof_pairs, score_mat, gap_penalty):
    height = max(one.shape[0] for one, _ in prof_pairs) + 1
    width = max(two.shape[0] for _, two in prof_pairs) + 1
    max_k = max(one.shape[0] + two.shape[0] + 1 for one, two in prof_pairs)

    s = jnp.full((len(prof_pairs), height, width, 3), jnp.nan)

    profs = _combine_pairs(prof_pairs)

    for k in range(max_k):
        fronts = []
        for i, (one, two) in enumerate(prof_pairs):
            if one.shape[0] + two.shape[0] + 1 <= k:
                continue
            fronts.append(_front_cells(k, i, (one.shape[0] + 1, two.shape[0] + 1)))

        cells = jnp.concatenate(fronts, 0)
        s = _resolve_cells_scatter(profs, score_mat, gap_penalty, cells, s)

    print(jnp.max(s[0, :, :, :], 2))
    print(jnp.max(s[1, :, :, :], 2))


def main():
    profs = read_profs("./data/input.fasta")

    align_pairs = [(profs[0], profs[1]), (profs[1], profs[0])]

    score_mat = jnp.eye(26, 26, dtype=jnp.float32)
    needleman_wunsch(align_pairs, score_mat, 1)

if __name__ == "__main__":
    main()