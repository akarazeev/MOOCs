def convex_hull(points):
    points = sorted(set(points))

    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


coords = []

with open('input.txt', 'r') as f_in:
    n = int(f_in.readline())
    for i in range(n):
        coords.append(list(map(lambda x: int(x), f_in.readline().split())))


def extnd(pts):
    if len(pts) == 1:
        tmp = pts
        tmp.append((9999, 9999))
        return tmp
    return pts


pts_good = []
pts_bad = []

for i in range(n):
    if coords[i][2] == 1:
        pts_good.append((coords[i][0], coords[i][1]))
    else:
        pts_bad.append((coords[i][0], coords[i][1]))

pts_good = extnd(pts_good)
pts_bad = extnd(pts_bad)

hull_good = convex_hull(pts_good)
hull_bad = convex_hull(pts_bad)


def build_proc_list(hull_list):
    ans = []

    for i in range(len(hull_list)-1):
        if i == 0:
            ans.append((hull_list[-1], hull_list[i]))
        ans.append((hull_list[i], hull_list[i+1]))

    return ans


to_process_good = build_proc_list(hull_good)
to_process_bad = build_proc_list(hull_bad)


def onSegment(p, q, r):
    if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
        return True

    return False


def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))

    if val == 0:
        return 0

    if val > 0:
        return 1
    else:
        return 2


def is_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and onSegment(p1, p2, q1):
        return True

    if o2 == 0 and onSegment(p1, q2, q1):
        return True

    if o3 == 0 and onSegment(p2, p1, q2):
        return True

    if o4 == 0 and onSegment(p2, q1, q2):
        return True

    return False


flag = True
for par1 in to_process_bad:
    for par2 in to_process_good:
        if is_intersect(par1[0], par1[1], par2[0], par2[1]):
            flag = False
            break

if flag is True:
    print('Yes')
else:
    print('No')
