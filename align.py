import jax
import jax.numpy as jnp
import jax.lax as jl
from Bio import SeqIO

def read_seqs(path):
    with open(path) as f:
        records = list(SeqIO.parse(f, "fasta"))

    seqs = []
    for rec in records:
        seq = []
        for c in rec.seq:
            idx = ord(c) - ord('A')
            seq.append(jnp.zeros((26,)).at[idx].set(1.0))
        seqs.append(jnp.vstack(seq))

    return seqs

@jax.jit
def calculate_scores(seq_one, seq_two, scores):
    return jnp.sum(seq_one[:, None, None, :] * seq_two[None, :, :, None] * scores[None, None, :, :], axis=(2, 3))

@jax.jit
def needleman_wunsch_unrolled(scored, gap):
    height = scored.shape[0] + 1
    width = scored.shape[1] + 1

    s = jnp.full((height, width, 3), -jnp.inf)

    for n in range(height):
        for m in range(width):
            if n == 0 and m == 0:
                s = s.at[n, m, :].set(0.)
                continue

            if n > 0 and m > 0:
                s = s.at[n, m, 0].set(jnp.max(s[n - 1, m - 1, :]) + scored[n-1, m-1])
            if m > 0:
                s = s.at[n, m, 1].set(jnp.max(s[n, m - 1, :]) - gap)
            if n > 0:
                s = s.at[n, m, 2].set(jnp.max(s[n - 1, m, :]) - gap)
    return s

def needleman_wunsch_loop(scored, gap):
    height = scored.shape[0] + 1
    width = scored.shape[1] + 1

    def loop(i, s):
        r = jnp.unravel_index(i, (height, width, 3))
        n = r[0]
        m = r[1]

        if n == 0 and m == 0:
            j = jnp.ravel_multi_index((0, 0, 0))
            s = s.at[j:j+3].set(0.)
            return s

        if n > 0 and m > 0:
            s = s.at[n, m, 0].set(jnp.max(s[n - 1, m - 1, :]) + scored[n-1, m-1])
        if m > 0:
            s = s.at[n, m, 1].set(jnp.max(s[n, m - 1, :]) - gap)
        if n > 0:
            s = s.at[n, m, 2].set(jnp.max(s[n - 1, m, :]) - gap)

        return s

    s = jnp.full((height * width * 3,), -jnp.inf)
    return jax.lax.fori_loop(0, height, loop, s)

def main():
    seqs = read_seqs("./data/input.fasta")

    scored = calculate_scores(seqs[0], seqs[1], jnp.eye(26, 26))
    # print(needleman_wunsch_unrolled(scored, 1.))
    print(needleman_wunsch_loop(scored, 1.))


if __name__ == "__main__":
    main()