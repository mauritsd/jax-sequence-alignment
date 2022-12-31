import jax
import jax.numpy as jnp
import jax.lax as jl
from Bio import SeqIO

def read_seqs(path):
    with open(path) as f:
        records = list(SeqIO.parse(f, "fasta"))

    seqs = []
    lens = []
    for rec in records:
        seq = []
        for c in rec.seq:
            idx = ord(c) - ord('A')
            seq.append(jnp.zeros((26,)).at[idx].set(1.0))
        seqs.append(jnp.vstack(seq))
        lens.append(len(rec.seq))
    max_len = max(lens)

    return jnp.stack([jl.pad(seq, -1., [(0, max_len - seq.shape[0], 0), (0,0,0)]) for seq in seqs], 0), lens

def _front_cells(k, dims):
    if 0 <= k < dims[0]:
        return jnp.stack([
            jnp.arange(k, -1, -1),
            jnp.arange(0, k + 1, 1),
        ], 1)
    elif 0 <= k < (dims[0] + dims[1] - 1):
        i = k - dims[0] + 1
        steps = dims[1] - i
        return jnp.stack([
            jnp.arange(dims[0] - 1, jnp.maximum(-1, dims[0] - 1 - steps), -1),
            jnp.arange(i, i + jnp.minimum(dims[0], steps), 1),
        ], 1)

    raise ValueError(f"requested front {k} out of bounds")

def _get_cell_dep(cell, s):
    cell_upleft = cell.at[:].add(-1)
    dep_upleft = jl.cond(
        (cell_upleft[0] >= 0) & (cell_upleft[1] >= 0),
        lambda cell, s: jnp.max(s[cell[0], cell[1]]),
        lambda cell, s: -jnp.inf,
        cell_upleft, s
    )
    cell_up = cell.at[0].add(-1)
    dep_up = jl.cond(
        cell_up[0] >= 0,
        lambda cell, s: jnp.max(s[cell[0], cell[1]]),
        lambda cell, s: -jnp.inf,
        cell_up, s
    )
    cell_left = cell.at[1].add(-1)
    dep_left = jl.cond(
        cell_left[1] >= 0,
        lambda cell, s: jnp.max(s[cell[0], cell[1]]),
        lambda cell, s: -jnp.inf,
        cell_left, s
    )

    return jnp.stack([dep_upleft, dep_up, dep_left], 0)
_get_cells_dep = jax.vmap(_get_cell_dep, (0, None), 0)

def _resolve_cell(dep, cell, profs, score_mat, gap):
    score = jl.cond(
        (cell[0] > 0) & (cell[1] > 0),
        lambda profs, score_mat, cell: jnp.sum(jnp.outer(profs[0, cell[0]-1, :], profs[1, cell[1]-1, :]) * score_mat),
        lambda profs, score_mat, cell: -jnp.inf,
        profs,
        score_mat,
        cell,
    )
    out = jnp.empty((3,)).at[:].set([dep[0] + score, dep[1] - gap, dep[2] - gap])

    return jl.cond(jnp.max(out) == -jnp.inf, (lambda x: x.at[:].set(0.)), (lambda x: x), operand=out)
_resolve_cells = jax.vmap(_resolve_cell, (0, 0, None, None, None), 0)

@jax.jit
def _resolve_cells_scatter(profs, score_mat, gap, cells, s):
    print("Compiling")
    deps = _get_cells_dep(cells, s)
    outputs = _resolve_cells(deps, cells, profs, score_mat, gap)

    return s.at[cells[:, 0], cells[:, 1]].set(outputs)

def needleman_wunsch(lens, profs, score_mat, gap):
    height = lens[0] + 1
    width = lens[1] + 1
    dims = (height, width)

    s = jnp.full((height, width, 3), jnp.nan)

    for k in range(dims[0] + dims[1] - 1):
        cells = _front_cells(k, dims)
        pad_cells = jl.pad(cells, 0, [(0, 16 - cells.shape[0], 0), (0, 0, 0)])
        s = _resolve_cells_scatter(profs, score_mat, gap, pad_cells, s)

    print(jnp.max(s, 2))

def main():
    profs, lens = read_seqs("./data/input.fasta")
    score_mat = jnp.eye(26, 26, dtype=jnp.float32)
    needleman_wunsch(lens, profs, score_mat, 1)

if __name__ == "__main__":
    main()