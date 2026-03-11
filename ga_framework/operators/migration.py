def ring_migration(islands, n=1):
    for i in range(len(islands)):
        src = islands[i].individuals[:n]
        dst = islands[(i + 1) % len(islands)]
        dst.individuals[-n:] = src