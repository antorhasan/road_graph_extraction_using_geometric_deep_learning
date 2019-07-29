import shapely
from shapely.geometry import LineString, Point

line = LineString([(0,0), (4,4)])

line1 = LineString([(0,4), (4,0)])
line2 = LineString([(2,0), (4,0)])
line3 = LineString([(1,0), (4,0)])
line4 = LineString([(3,0), (4,0)])

lis = [line2,line3,line3,line1]
for i in range(len(lis)):
    try:
        int_pt = line.intersection(lis[i])
        point_of_intersection = int_pt.x, int_pt.y
        
    except:
        continue

print(point_of_intersection)