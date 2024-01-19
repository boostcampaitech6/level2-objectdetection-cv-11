import os

label_path = '/data/ephemeral/home/yolov5/runs/detect/yolov5x_AdamW_100epoch_high_test4/labels'

label_list = sorted(os.listdir(label_path))

g = open('submit.txt', 'w')

for i in range(len(label_list)):
    f = open(os.path.join(label_path, label_list[i]), 'r')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
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
    g.write(',test/'+label_list[i].split('.')[0]+'.jpg'+'\n')
    f.close()

g.close()
# f = open('0000.txt', 'r')
# g = open('submit.txt', 'w')

# lines = f.readlines()
# for line in lines:
#     l = line.split(' ')
#     label = l[0]
#     score = l[5]
#     minx = str((float(l[1]) * 1024) - ((1024 * float(l[3])) / 2))
#     miny = str((float(l[2]) * 1024) - ((1024 * float(l[4])) / 2))
#     maxx = str((float(l[1]) * 1024) + ((1024 * float(l[3])) / 2))
#     maxy = str((float(l[2]) * 1024) + ((1024 * float(l[4])) / 2))
#     w = label+' '+score+' '+minx+' '+miny+' '+maxx+' '+maxy+' '
#     g.write(w)

# f.close()
# g.close()