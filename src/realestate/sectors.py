# Bucharest's sectors 1-6 are wedges in a clockwise ring, so each borders the two either side.
SECTOR_WITH_NEIGHBOURS: dict[int, tuple[int, int, int]] = {
    1: (1, 2, 6),
    2: (1, 2, 3),
    3: (2, 3, 4),
    4: (3, 4, 5),
    5: (4, 5, 6),
    6: (1, 5, 6),
}
