import sys

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("Specify fname!")
        exit(0)
    fname = sys.argv[1]
    with open(fname+'_raw.txt', 'r') as f:
        data = f.read().split('\n')
        if len(data[-1]) < 2:
            del data[-1]
    data = [d.split(',') for d in data]
    
    X = [d[:-1] for d in data]
    y = [d[-1] for d in data]
    dic = {}
    lset = set(y)
    i = 0
    for yy in lset:
        dic[yy] = str(i)
        i = i+1
    y = [dic[i] for i in y]
    
    X = [','.join(x) for x in X]
    data = [x+','+yy for (x,yy) in zip(X, y)]
    data = '\n'.join(data)

    with open(fname+'.txt', 'w') as f:
        f.write(data)
