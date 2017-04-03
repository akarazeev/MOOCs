def get_neighbrs(x, y, num_rows, num_cols):
    ans = []
    if x+1 < num_rows:
        ans.append((x+1, y))
    if y+1 < num_cols:
        ans.append((x, y+1))
    if x-1 >= 0:
        ans.append((x-1, y))
    if y-1 >= 0:
        ans.append((x, y-1))
    return ans


matrix = []

with open('input.txt', 'r') as f_in:
    n, m, k = list(map(lambda x: int(x), f_in.readline().split()))
    for line in f_in:
        matrix.append(list(map(lambda x: int(x), line.split())))

matrix_diffs = [list() for i in range (n)]

for i in range(len(matrix_diffs)):
    matrix_diffs[i] = [0 for j in range(m)]

for iterr in range(k):
    matrix_prev = [x[:] for x in matrix]

    for i in range(n):
        for j in range(m):
            tmp = [matrix_prev[tup[0]][tup[1]] for tup in get_neighbrs(i, j, n, m)]
            if sum(elem == 2 for elem in tmp) > 1:
                matrix[i][j] = 2
            elif sum(elem == 2 or elem == 3 for elem in tmp) >= 1:
                matrix[i][j] = 3
            else:
                matrix[i][j] = 1
            # note the differences
            if matrix[i][j] != matrix_prev[i][j]:
                matrix_diffs[i][j] += 1

for row in matrix_diffs:
    print(' '.join(map(lambda x: str(int(x)), row)))
