import sys

if len(sys.argv)!=2:
    print("Specify file name!")
    exit(0)

fname = sys.argv[1]

with open(fname, 'r') as f:
    data = f.read().split('\n\n')

data = [d.split('\n') for d in data]
data = [' & '.join(d) for d in data]

for d in data:
    print(d, '\\\\')
