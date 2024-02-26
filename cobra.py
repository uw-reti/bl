# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:17:56 2023

@author: b9801
"""

from pwr1 import get_pins,get_w
from heat_map import calc_P
import numpy as np
from heat_map import cplot
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


def chopped_cosine_distribution(x, a, b):
    """
    From ChatGPT
    Chopped cosine distribution with peak value of 1.55.
    
    Args:
        x (array-like): Input values to evaluate the distribution at.
        a (float): Lower bound of the distribution.
        b (float): Upper bound of the distribution.
        
    Returns:
        array-like: Values of the chopped cosine distribution evaluated at x.
    """
    y = np.zeros_like(x)
    peak = 1.55
    c = (a + b) / 2.0
    delta = (b - a) / 2.0
    mask = (x >= a) & (x <= b)
    y[mask] = peak * (0.5 * (1.0 + np.cos(np.pi * (x[mask] - c) / delta)))
    return y

def subchan_center(sc):
    px = np.mean(np.array([s[0] for s in sc]))
    py = np.mean(np.array([s[1] for s in sc]))
    return (px,py)

def get_subchan_coords(subchan):
    coords=[]
    p0=1.26
    gt_x,gt_y = get_w(p0)
    for sc in subchan:
        if isinstance(sc,int):
            coords.append([(pins[sc][0],pins[sc][1]),(21.5/2,pins[sc][1]),(21.5/2,21.5/2)])
        else:
            c=list()
            for s in sc:
                if s<0:
                    c.append((gt_x[-s-1],gt_y[-s-1]))
                else:
                    c.append((pins[s][0],pins[s][1]))
            if len(c)==2:
                c.append((21.5/2,c[1][1]))
                c.append((21.5/2,c[0][1]))    
            coords.append(c)
    return coords

def first_three_significant_figures(number):
    """
    ChatGPT
    Returns the first three significant figures of a number as a string.
    
    Args:
        number (float): The number to extract the first three significant figures from.
        
    Returns:
        str: The first three significant figures of the number, as a string.
    """
    formatted_number = "{:.3E}".format(number).replace(".","")
    first_three_digits = formatted_number[:3]
    return first_three_digits

def round_to_3sf(number):
    """
    Rounds a number to 3 significant figures.
    
    Args:
        number (float): The number to round.
        
    Returns:
        float: The rounded number.
    """
    if number == 0:
        return 0.0
    
    magnitude = np.floor(np.log10(abs(number))) + 1
    precision = int(3 - magnitude)
    rounded_number = round(number, precision)
    return rounded_number

def write_6digit(item):
    n=round_to_3sf(item)
    f3sf="{:.3E}".format(n).replace(".","")[:3]
    mystr=str(f3sf)+"E"
    exponent=np.log10(n/float(f3sf))
    if exponent >=0:
        mystr+="+"
    mystr+=str(round(exponent))
    return mystr




def distance_between_centers(pin1,pin2):
    return np.sqrt((pin1[0]-pin2[0])**2+(pin1[1]-pin2[1])**2)

def angle_between_three_points(point_a, point_b, point_c):
    """
    Calculates the angle between three points in radians.

    Args:
        point_a (tuple): The (x, y) coordinates of the first point.
        point_b (tuple): The (x, y) coordinates of the second point.
        point_c (tuple): The (x, y) coordinates of the third point.

    Returns:
        float: The angle between the three points in radians.
    """
    # Calculate the vectors formed by the three points
    vector_ab = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    
    #Negative sign put as as both lines must point TO the same point
    vector_bc = -np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])
    

    # Calculate the dot product and magnitude of the vectors
    dot_product = np.dot(vector_ab, vector_bc)
    magnitude_ab = np.linalg.norm(vector_ab)
    magnitude_bc = np.linalg.norm(vector_bc)

    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine = dot_product / (magnitude_ab * magnitude_bc)

    # Calculate the angle in radians using the arccosine function
    angle = np.arccos(cosine)
    
    #we want the internal angle
    return angle
    
def polygon_area(coords):
    """
    Calculates the area of a polygon given its coordinates using the Shoelace formula.
    
    Args:
        coords (list or numpy.ndarray): List of tuples of the vertices
    
    Returns:
        float: The area of the polygon.
    """
    # Ensure that x_coords and y_coords are numpy arrays
    x_coords = np.asarray([coord[0] for coord in coords])
    y_coords = np.asarray([coord[1] for coord in coords])
    
    # Apply the Shoelace formula
    area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
    
    return area

import unittest

class Testing(unittest.TestCase):
    def test_chopped_cosine_distribution(self):
        self.assertTrue(np.allclose([1.55],chopped_cosine_distribution(np.array([0.5]),0,1)))
    
    def test_subchan_center(self):
        s=[[0,0],[1,0],[0,1],[1,1]]
        x,y=subchan_center(s)
        self.assertEqual(x,0.5)
        self.assertEqual(y,0.5)
        
    
    def test_round_to_3sf(self):
        self.assertEqual(1.23,round_to_3sf(1.226))
        self.assertEqual(12.7,round_to_3sf(12.717171))
        
    def test_write_6digit(self):
        self.assertEqual("123E-5",write_6digit(0.001226))
        self.assertEqual("456E+1",write_6digit(4564))
        
    def test_distance_between_centers(self):
        self.assertEqual(5.0,distance_between_centers((0,0),(3,4)))
    
    def test_angle_between_three_points(self):
        a=[0,0]
        b=[0,1]
        c=[1,1]
        self.assertEqual(np.pi/2,angle_between_three_points(a,b,c))
    
    def test_polygon_area(self):
        coords=[(0,0),(0,1),(1,1),(1,0)]
        self.assertEqual(1.0,polygon_area(coords))
        
        coords=[(0,0),(0,1),(1,1)]
        self.assertEqual(0.5,polygon_area(coords))
    
T=Testing()
T.test_chopped_cosine_distribution()
T.test_subchan_center()
#T.test_get_subchan_coords() confirmed visually so no unit test needed
T.test_distance_between_centers()
T.test_write_6digit()
T.test_round_to_3sf()
T.test_angle_between_three_points()
T.test_polygon_area()

npins=324
calc_dnb=False #inlet +2K, 95% nominal flow, 112% power and 1.587 radial peaking factor

#want to run the following cases:
#increase raw power of the 'hot' assembly to recover the extra allowable power peaking under same core power
#case where the power and flow are both uprated in proportional to number of pins to see if MDNBR preserved
uprate_case = None #None, "pow "and "both"

if uprate_case is not None and npins == 264:
    raise ValueError

if uprate_case == "pow":
    pow_uprate = 1.095 #tune to original reference value
    flow_uprate = 1.0

elif uprate_case == 'both':
    pow_uprate = npins/264
    flow_uprate = npins/264 

elif uprate_case is None:
    pow_uprate = 1.0
    flow_uprate = 1.0

else:
    raise ValueError

if calc_dnb:
    pow_fac=pow_uprate*1.12*1.587
    flow_fac=0.95
else:
    pow_fac=pow_uprate
    flow_fac=1.0

f=np.sqrt(264/npins)


p0 = 1.26
p2 = 17/19*p0
pins = get_pins(p0,p2,f,npins)
P=calc_P(npins,pins)
gt_x,gt_y = get_w(p0)

sc=[]
if npins==264:
    sc.append((6,7,-1)) #guide tubes are negatives
    sc.append((6,7,16,15))
    sc.append((15,16,24,-7))
    sc.append((-7,24,31,30))
    sc.append((30,31,40,39))
    sc.append((39,40,47,-8))
    sc.append((47,-8,54,55))
    sc.append((54,55,64,63))
    sc.append((63,64))
    sc.append((7,16,17))
    sc.append((16,17,25,24))
    sc.append((24,25,32,31))
    sc.append((31,32,41,40))
    sc.append((40,41,48,47))
    sc.append((47,48,56,55))
    sc.append((55,56,65,64))
    sc.append((64,65))
    sc.append((17,25,-3))
    sc.append((25,-3,33,32))
    sc.append((32,33,42,41))
    sc.append((41,42,-9,48))
    sc.append((48,-9,57,56))
    sc.append((56,57,66,65))
    sc.append((65,66))
    sc.append((-3,33,34))
    sc.append((33,34,43,42))
    sc.append((42,43,49,-9))
    sc.append((-9,49,58,57))
    sc.append((57,58,67,66))
    sc.append((66,67))
    sc.append((34,43,-4))
    sc.append((43,-4,50,49))
    sc.append((49,50,59,58))
    sc.append((58,59,68,67))
    sc.append((59,60,69,68))
    sc.append((67,68))
    sc.append((-4,50,51))
    sc.append((50,51,60,59))
    sc.append((68,69))
    sc.append((51,60,61))
    sc.append((60,61,70,69))
    sc.append((69,70))
    sc.append((61,70,71))
    sc.append((70,71))
    sc.append((71))
elif npins==280:
    sc.append((-1,46,66))
    sc.append((66,46,47,68))
    sc.append((68,47,48,-7))
    sc.append((-7,48,49,70))
    sc.append((70,49,50,72))
    sc.append((72,50,51,-8))
    sc.append((-8,51,15,13))
    sc.append((13,15,16,14))
    sc.append((14,16))
    sc.append((46,47,54))
    sc.append((47,54,55,48))
    sc.append((48,55,56,49))
    sc.append((49,56,57,50))
    sc.append((50,57,58,51))
    sc.append((51,58,17,15))
    sc.append((15,17,18,16))
    sc.append((16,18))
    sc.append((54,55,-3))
    sc.append((55,-3,71,56))
    sc.append((56,71,73,57))
    sc.append((57,73,-9,58))
    sc.append((58,-9,19,17))
    sc.append((17,19,20,18))
    sc.append((18,20))
    sc.append((-3,71,63))
    sc.append((71,63,64,73))
    sc.append((73,64,65,-9))
    sc.append((-9,21,19))
    sc.append((19,21,22,20))
    sc.append((20,22))
    sc.append((-9,65,21))
    sc.append((65,23,21))
    sc.append((21,23,24,22))
    sc.append((22,24))
    sc.append((8,65,23))
    sc.append((8,23,25))
    sc.append((23,25,26,24))
    sc.append((24,26))
    sc.append((-4,2,8))
    sc.append((2,8,25,27))
    sc.append((25,27,28,26))
    sc.append((26,28))
    sc.append((2,9,27))
    sc.append((9,27,28,10))
    sc.append((10,28))
    sc.append((9,12,10))
    sc.append((10,12))
    sc.append((12))
    sc.append((63,64,-4))
    sc.append((64,65,8,-4))
elif npins==292:
    raise ValueError("Have disqualified this case")
    pass
elif npins==324:
    sc.append((-1,60,74))
    sc.append((60,74,76,62))
    sc.append((62,76,41,-7))
    sc.append((-7,41,43))
    sc.append((-7,43,64))
    sc.append((45,43,64))
    sc.append((64,45,66))
    sc.append((45,66,47))
    sc.append((66,47,-8))
    sc.append((47,49,-8))
    sc.append((49,-8,4,6))
    sc.append((6,4,5,7))
    sc.append((5,7))
    sc.append((74,77,76))
    sc.append((77,76,41,51))
    sc.append((51,41,43,53))
    sc.append((53,55,45,43))
    sc.append((45,55,57,47))
    sc.append((47,57,59,49))
    sc.append((49,59,8,6))
    sc.append((6,8,9,7))
    sc.append((7,9))
    sc.append((77,-3,51))
    sc.append((-3,51,53,78))
    sc.append((53,78,55))
    sc.append((55,80,78))
    sc.append((55,80,57))
    sc.append((80,-9,59,57))
    sc.append((59,-9,10,8))
    
    sc.append((78,82,84,-3))
    sc.append((78,82,80))
    sc.append((80,82,39,-9))
    sc.append((84,85,-4))
    sc.append((82,84,85))
    sc.append((82,85,39))
    sc.append((85,39,73,-4))
    sc.append((10,12,37,-9))
    sc.append((39,73,37,-9))
    sc.append((73,70,37))
    sc.append((37,12,14))
    sc.append((70,37,14))
    sc.append((70,14,16))
    sc.append((70,73,-4,71))
    sc.append((70,68,71))
    sc.append((70,68,18,16))
    sc.append((68,0,18))
    
    sc.append((8,10,11,9))
    sc.append((9,11))
    sc.append((10,12,13,11))
    sc.append((11,13))
    sc.append((12,14,15,13))
    sc.append((13,15))
    sc.append((14,16,17,15))
    sc.append((15,17))
    sc.append((16,18,19,17))
    sc.append((17,19))
    sc.append((18,0,1,19))
    sc.append((1,19))
    sc.append((0,1,3))
    sc.append((1,3))
    sc.append((3))
    
#Write nchan and nrod (noting octant symmetry except 280 pins)
nchan=len(sc)

#unique rods used in the problem
rods=list()
for s in sc:
    try:
        for item in s:
            if item>=0:
                rods.append(item)
    except TypeError:
        if s>=0:
            rods.append(s)
rods=list(set(rods))
nrod=len(rods)

subchan_coords=get_subchan_coords(sc)

cplot(np.zeros(len(pins))+0.5,0.475*f,p0,pins,None,subchan=subchan_coords) #Draws them

pin_pow=3411e6/193.0/npins/4


with open("COBRA/INPFILE","w") as IF:
    #1
    IF.write(str(npins)+"\n")
    #2
    IF.write("{:6d}{:6d}{:6d}{:6d}\n".format(1,2,2,0)) #last number is COBRA 3 eqn model vs 4 eqn and TWIGL
    
    #3
    nax=16 #number of axial layers
    nctyp=nchan #set each channel as own type
    ngrid=8 #From my thesis 
    ngtype=1 #From my thesis
    nrnode=10 #number of radial nodes in fuel pellet

    IF.write("{:6d}{:6d}{:6d}{:6d}{:6d}{:6d}{:6d}{:6d}{:6d}      {:6d}{:6d}\n".format(1,nchan,nrod,nax,nctyp,ngrid,ngtype,nrnode,0,1,1))
    
    #4
    axl = 3.66/nax #axial node length
    IF.write("{:12.5E}\n".format(-axl))
    
    #5
    IF.write("{:6d}\n".format(nax))
    xpoints=(np.arange(nax)+0.5)
    ypoints=chopped_cosine_distribution(xpoints,0,nax)
    for j in range(nax):
        this_ax=axl*(0.5+j)
        IF.write("{:12.5E}\n".format(this_ax))
        for k in range(nrod):
            IF.write("{:12.5E}".format(pin_pow*ypoints[j]*P[rods[k]]*pow_fac))
            if k%6==5 or k == nrod-1:
                IF.write("\n")

    #7
    
    

    for j in range(nchan):
        IF.write("{:6d}   ".format(j+1))
        
        for k in range(j+1,nchan):
            common_values = set(subchan_coords[j]) & set(subchan_coords[k])
            if len(common_values) == 2:
                #they share a border
                cv=list(common_values)
                length_of_gap=distance_between_centers(cv[0],cv[1])
                if j==0:
                    print(length_of_gap)
                #What to subtract depends on if guide tube, pin or neither
                for item in cv:
                    if tuple(list(item)+[f]) in pins:
                        length_of_gap-=0.475*f
                    for gx,gy in zip(gt_x,gt_y):
                        if item[0] == gx and item[1] == gy:
                            length_of_gap-=0.612
                centroid_to_centroid=distance_between_centers(subchan_center(subchan_coords[j]),subchan_center(subchan_coords[k]))
                IF.write("{:3d}".format(k+1))
                IF.write(write_6digit(length_of_gap/100))
                IF.write(write_6digit(centroid_to_centroid/100))

                        
        IF.write("\n") 
    IF.write("     0     0\n") #end of this set of cards

    #8
    for j,rod in enumerate(rods):
        IF.write("{:3d}".format(j+1))
        IF.write("{:6d}".format(1)) #fuel type
        IF.write("   ")
        
        for k,s in enumerate(sc):
            try:
                schans = list(s)
            except:
                schans=list([s])
            if rod in schans: #this rod borders this subchannel
                IF.write("{:3d}".format(k+1))
                r_idx = schans.index(rod) #position of rod within schans
                master_coord = subchan_coords[k][r_idx]
                coord1 = subchan_coords[k][r_idx-1]
                try:
                    coord2 = subchan_coords[k][r_idx+1]
                except IndexError:
                    coord2 = subchan_coords[k][0] #if r_idx is on the end go round to first coords
                
                angle = angle_between_three_points(coord1,master_coord,coord2)
                IF.write(write_6digit(angle/2/np.pi))
        IF.write("\n")
    
    IF.write("000\n")
    
    
    #10
    for j in range(nchan):
        IF.write("{:6d}{:6d}".format(1,1)) #friction indicator, multiplier to wetted and heated perimeters
        
        flow_area=polygon_area(subchan_coords[j])
        heated_perim=0
        wetted_perim=0
        poly=len(subchan_coords[j]) #is this a quadrilateral or a triangle
        
        if isinstance(sc[j],int):
            tmp_sc=[sc[j]]
        else:
            tmp_sc=sc[j]
        
        n_heated=0
        for k in range(len(tmp_sc)): #only compute for water rods and fuel rods, not added corners
            angle=angle_between_three_points(subchan_coords[j][k-1],subchan_coords[j][k],subchan_coords[j][(k+1)%poly])
            if tmp_sc[k]<0:
                #water rod
                wetted_perim+=0.61*angle
                flow_area-=0.5*0.61**2*angle
            else:
                wetted_perim+=0.475*f*angle
                heated_perim+=0.475*f*angle
                flow_area-=0.5*(0.475*f)**2*angle
                n_heated+=1

        IF.write("{:12.5E}".format(flow_area/100**2))
        IF.write("{:12.5E}".format(heated_perim/100))
        IF.write("{:12.5E}\n".format(wetted_perim/100))
        IF.write("{:12.5E}\n".format(1.0)) #grid spacer coefficient
        if j>0:
            IF.write("{:6d}{:6d}\n".format(j+1,0)) #channels part of the present type. Omit for first type. Terminate with zero
                
    #11
    IF.write(""" .0625     1 .1875     1 .3125     1 .4375     1 .5625     1 .6875     1
 .8125     1 .9375     1\n""")
        
    #12
    IF.write("{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}\n".format(0.4095/100*2,(0.475-0.418)/100,10400,6552,0.475/100*2,0))
    hgap=5e4 #https://www.sciencedirect.com/science/article/pii/S0149197020302353
    IF.write("{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}\n".format(-11,-11,-11,-11,hgap,0.0))
    
    #Rest of cards from thesis. Mflux set to give 30C T rise under nominal
    IF.write("""     1     0     1     1     0     1     1     0     1     0
  .038     0     0     0
     1
     1     1
     0     0     0     0     0     0
0.50   0.0  0.50  0.0
     2     0     0  0.954    0     0    0.    0.    0.
$                                                                 ISCHEM
     0   500    0.  0.     0.    0.     0.    0.     0    0.    0.     1.
     1 {0}  {1:.0f}  15.5   0.0     0   1.0 0.026     0    0.    1.    0.
    0.     0     0
     0     0     0     0     0
     0     0     3     0     0     0    -1
""".format("565.3" if not calc_dnb else "567.3",2952*flow_uprate*flow_fac))

import os
os.system("cd COBRA && cobra.exe")

#read the output file
with open("COBRA/OUTFILE","r") as OF:
    data=OF.readlines()

T_out=[]
for j,line in enumerate(data):
    if "AVERAGE PRESSURE DROP (Pa)" in line:
        print(line)
    
    if "CHANNEL EXIT SUMMARY RESULTS" in line:
        for k in range(j+13,j+13+nchan):
            T_out.append(float(data[k].split()[3])-293.15)
    
    if "MDNBR" in line:
        MDNBR=[]
        for k in range(j+3,j+3+nax):
            value = float(data[k].split()[2])
            if value > 0:
                MDNBR.append(value)
        print("MDNBR is "+str(min(MDNBR)))
            

T_max=np.max(T_out)
T_min=np.min(T_out)
print(T_max,T_min)
            
T_out = np.array(T_out)
T_out -= 300
T_out /= 3.0


#create a new subchan plot which draws the temperature as heat map and then has white pins over the top
cplot(T_out,0.475*f,p0,pins,None,subchan=subchan_coords,sc_out=True,colorbar=True,vbounds=[300,303])