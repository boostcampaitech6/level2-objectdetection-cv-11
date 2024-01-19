import os

f = open('0000.txt', 'r')
g = open('submit.txt', 'w')

lines = f.readlines()
for line in lines:
    l = line.split(' ')
    label = l[0]
    score = l[5]
    minx = str((float(l[1]) * 1024) - ((1024 * float(l[3])) / 2))
    miny = str((float(l[2]) * 1024) - ((1024 * float(l[4])) / 2))
    maxx = str((float(l[1]) * 1024) + ((1024 * float(l[3])) / 2))
    maxy = str((float(l[2]) * 1024) + ((1024 * float(l[4])) / 2))
    w = label+' '+score+' '+minx+' '+miny+' '+maxx+' '+maxy+' '
    g.write(w)

f.close()
g.close()


