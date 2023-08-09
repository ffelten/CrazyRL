def longueur_bloc(l, pos):
    elem = l[pos]
    count = 1
    pos += 1
    while pos < len(l) and l[pos] == elem:
        count += 1
        pos += 1
    return count


def transform(l):
    idx = 0
    t = []
    while idx < len(l):
        n = longueur_bloc(l, idx)
        t.append(n)
        t.append(l[idx])
        idx += n
    return t


l = [0, 0, 0, 0, 5, -2, -2, -2, 0, 0, 0, 0, 0]
print(transform(l))
