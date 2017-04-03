with open('input.txt', 'r') as f_in:
    inp = f_in.read()

n, m, k = list(map(lambda x: int(x), inp.split()))

print(n * m * (1. - ((1. - (1./(n * m))) ** k)))
