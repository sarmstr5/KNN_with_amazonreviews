import operator
d = {}
z = 100
for i in range(z):
    d[i] = i*2
print(d)
d_sort = sorted(d.items(), key = lambda x: x[1], reverse = True)
new_d = {}
for t in d_sort[:int(len(d_sort)*.1)]:
    print(t)
    new_d[t[0]] = t[1] 

print(new_d)

