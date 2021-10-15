# This is where I store various experimental codes and arcive stuff.

# Importing 3rd party libraries
import math
import time
import cmath
import timeit
import random
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as ml
import Payoff_Structures
# import pvp.Payoff_Structures
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import OrderedDict
from Payoff_Function_2D import sellist
from Payoff_Structures import MGPayoffs3
from Payoff_Structures import MGPayoffs9
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon
from matplotlib import animation
from matplotlib import colors
from scipy.stats import norm
from matplotlib import cm
import plotly.io as pio
# import collections as coll


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

pio.renderers.default = "browser"

time0 = timeit.default_timer()

def varname(var):
    v = f'{var=}'
    return v.split('=')[0]

"""These are the inputs to the distgen, hillcover, and histogram functions.  Equation is the shape of the density
function.  Nagents is the number of agents in the simulation.  Standard determines if the trinomeal is in standard
or vertex form.  Rand determines if the outputs are noisy or deterministic."""

nagents, numrounds = 1200, 40
equation = [-1,0.3,0.7]
powerequ = [0.01,0.9,1]
# selfequ = [0.01,1,0.01]
# otherequ = [0.01,1,0.01]
# selfequ = [0.01,-1,1] 
# otherequ = [0.01,-1,1]
selfequ = [0.01,0.01,1]
otherequ = [0.01,0.01,1]
envyequ = [0.01,0.01,1]
guiltequ = [0.01,0.01,1]
vtthetaequ = [180, 10]
egthetaequ = [180, 180]
standard, rand, showhist = True, True, False
showgalaxy, chronicle, normaxes = True, False, True
cirpay, circular, Angle = [90, 45], False, False
givetime, givetimes, RandPay = True, False, True
showposts, betview = True, False
# experimenter's input
Axes = {'X Axis' : 'Self', 'X Label' : 'Self Regard', 'Y Axis' : 'Other', 'Y Label' : 'Other Regard', 'Z Axis' : 'Final Points', 'Z Label' : 'Total Points'}
# Axes = {'X Axis' : 'Envy', 'X Label' : 'Level of Envy', 'Y Axis' : 'Guilt', 'Y Label' : 'Level of Guilt', 'Z Axis' : 'Final Points', 'Z Label' : 'Total Points'}
# Axes = {'X Axis' : 'VTtheta', 'X Label' : 'VT Angle', 'Y Axis' : 'EGtheta', 'Y Label' : 'EG Angle', 'Z Axis' : 'Final Points', 'Z Label' : 'Total Points'}
# Axes = {'X Axis' : 'Prior', 'X Label' : 'Prior Mean', 'Y Axis' : 'WtRatio', 'Y Label' : 'Learnability', 'Z Axis' : 'Final Points', 'Z Label' : 'Betting Points'}

sweight, oweight, gweight, eweight, vweight, age = 1, 1, 0, 0, 1, 1

if nagents % 2 > 0:
    nagents = nagents - 1

WorldParameters = {'p(switch)' : 0.5, 'p(catch)' : 0.5, 'p(simultanious)' : 0.0, 'p(multi step)' : 0.0}

def calcmixlist(mixlist):
    """This provides the sum, mean, and number of numbers and strings in a list."""
    numbs = 0
    for i in mixlist:
        numbs = numbs + 1 if isinstance(i, (int, float)) else numbs
    summer = sum([i for i in mixlist if isinstance(i, (int, float)) or i.isdigit()])
    average, omitted = round(summer/numbs,3), len(mixlist) - numbs
    return summer, average, numbs, omitted

def randexcept(not1=0.5,not2=0.5,not3=0.5,not4=0.5,not5=0.5):
    """This gives a random number between 0 and 1, except for whatever number(s) you specify."""
    rando, stop = random.random(), 0
    while rando == not1 and rando == not2 and rando == not3 and rando == not4 and rando == not5 and stop < 10:
        rando = random.random()
        stop = stop + 1
    return rando

def normround(array, rnd=20):
    Min = min(array)
    for i in range(len(array)):
        array[i] = array[i] - Min
    Max = max(array)
    for i in range(len(array)):
        array[i] = array[i] / Max
    for i in range(len(array)):
        array[i] = round(array[i], rnd)
    return array

def normcircle(mean = 180, stdev = 60, ammt = nagents, rnd = 5):
    # us = S(Cs-Rs) + O(Co-Ro)
    # S (-1,1) is self regard, O (-1,1) is regard of others
    count, normlist = 0, []
    while count < ammt:
        sample = np.random.normal(mean, stdev)
        quotient = sample // 360
        if quotient != 0:
            sample = sample - 360*quotient
        sample = (sample * math.pi) / 180
        normlist.append(round(sample, rnd))
        count = count + 1
    return normlist
    # a list of angles represent the angle between social utility function
    
# plt.hist(normcircle(),51), plt.show(), exit()

def circlepay():
    angle, payoffs = 0, []
    while angle < 360:
        PayID = 'Cir' + str(angle)
        x = round(math.cos((angle*math.pi) / 180), 5)
        y = round(math.sin((angle*math.pi) / 180), 5)
        a1, b1 = round(x + .5, 5), round(x - .5, 5)
        a2, b2 = round(y + .5, 5), round(y - .5, 5)
        paystr = [a1, b1, a2, b2, PayID]
        payoffs.append(paystr)
        angle = angle + 1   
    return payoffs

payoffs360 = circlepay()
angledist = normcircle(cirpay[0], cirpay[1])
angledist = [int(t*180/math.pi) for t in angledist]

sellist360 = []
for i in angledist:
    sellist360.append(payoffs360[i])

def angletheta(spin, Angle = True):
    """This converts an angle into theta."""
    if Angle: 
        quotient = spin // 360
        if quotient != 0:
            spin = spin - 360*quotient
        theta = (spin * math.pi) / 180
    else:
        theta = (spin * 180 ) / math.pi
    return theta

# print(angletheta(225, True)), exit()

def slopedegs(x, y, degrees = True):
    """This converts the slope into degrees or theta."""
    if x == 0 and y == 0:
        angle = 0
        theta = angletheta(angle)
    elif x > 0 and y == 0:
        angle = 0
        theta = angletheta(angle)
    elif x == 0 and y > 0:
        angle = 90
        theta = angletheta(angle)
    elif x < 0 and y == 0:
        angle = 180
        theta = angletheta(angle)
    elif x == 0 and y < 0:
        angle = 270
        theta = angletheta(angle)
    elif x > 0 and y > 0:
        theta = np.arctan(y/x)
        angle = round(theta*180/np.pi,5)
    elif x < 0 and y > 0:
        theta = np.arctan(y/x)
        angle = abs(theta*180/np.pi) + 90
        theta = round(math.radians(angle),5)
    elif x < 0 and y < 0:
        theta = np.arctan(y/x)
        angle = abs(theta*180/np.pi) + 180
        theta = round(math.radians(angle),5)
    elif x > 0 and y < 0:
        theta = np.arctan(y/x)
        angle = abs(theta*180/np.pi) + 270
        theta = round(math.radians(angle),5)
    else:
        theta = np.arctan(y/x)
        angle = theta*180/np.pi    
    # degs = int(round(angle,0)) if degrees else theta
    degs = round(angle,2) if degrees else theta
    return degs

# print(slopedegs(-2, 0, True))
# exit()
# agent function
def virtuechoice(VTangle=225, paystruct=[5, 9, 9, 1, 'MG01'], Angle = True, show = False, stdev = 1.5):
    #outputs a graph that output probability of choosing the options
    #ex: optionA 5,9
    #optionB 9,1
    # probability of agent making that choice
    # utility for self  Aself - Bself, utility for other Aothr - Bothr
    Aself, Aothr, Bself, Bothr = paystruct[0], paystruct[1], paystruct[2], paystruct[3]
     # utility for self  Aself - Bself, utility for other Aothr - Bothr
     # different from previous S and O (-1,1)
    S, O = Aself - Bself, Aothr - Bothr
    theta = angletheta(VTangle, Angle)
    probA = sp.stats.norm.cdf(S * math.cos(theta) - O * math.sin(theta), 0, stdev)  
    probpolicy = randexcept(probA)  
    choice = [Aself, Aothr] if probA > probpolicy else [Bself, Bothr]
    reject = [Bself, Bothr] if probA > probpolicy else [Aself, Aothr]
    if show:
        fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.suptitle('Virtue Threshold', fontdict={'fontname':'Calibri'}),
        ax.set_xlabel('Utility Self'), ax.set_ylabel('Utility Other'), ax.set_zlabel('Probability of Choice')
        X = np.arange(-9, 9, .1)
        Y = np.arange(-9, 9, .1)
        X, Y = np.meshgrid(X, Y)
        Z = sp.stats.norm.cdf(X * math.cos(theta) - Y * math.sin(theta), 0, stdev)
        ax.set_xlim(-10, 10), ax.set_ylim(-10, 10)
        ax.plot_surface(X, Y, Z, cmap=cm.inferno, linewidth=0, antialiased=False), plt.show() 
    return choice, reject

# Virtue = virtuechoice(225, [5, 9, 9, 1, 'MG01'], Angle = True, show = True)
# print(Virtue)
# exit()

def prior(mean = 180, stdev = 50, show = False, norm = True):
    FlatAngle, CirAngle = np.arange(-360, 720, 1), np.arange(0, 360, 1)
    FlatPrior, CirPrior = sp.stats.norm.pdf(FlatAngle, mean, stdev), []
    for i, v in enumerate(CirAngle):
        CirPrior.append(max(FlatPrior[i], FlatPrior[i+360], FlatPrior[i+720]))  
    if show:
        plt.rcParams['axes.linewidth'] = 1.2
        plt.plot(CirAngle, CirPrior, color = 'rebeccapurple', linewidth = 5)
        plt.fill_between(CirAngle, CirPrior, color='mediumpurple')
        plt.title('Prior Over VTs'), plt.xlabel('Virtue Theshold Angles'), plt.ylabel('Probability Density')
        plt.show()
    CirPrior = normround(CirPrior, 5) if norm else CirPrior
    return CirPrior

# Prior = prior(mean = 90, stdev = 50, show = True)
# # print(Prior), 
# exit()

def likelihood(choice, reject, loaf = True, show = False, stdev = 0.5, steps = 1, Deg360 = 360, Angle = True):
    if loaf:
        cangle, vangle = np.arange(0, Deg360, steps), np.arange(0, Deg360, steps)
        cangle, vangle = np.meshgrid(cangle, vangle)
        likeprob = sp.stats.norm.cdf(np.cos(cangle*np.pi/180) * np.cos(vangle*np.pi/180) - np.sin(cangle*np.pi/180) * np.sin(vangle*np.pi/180), 0, stdev)
        if show:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            plt.suptitle('Likelihood: p(Choice Given Virtue Threshold)', fontdict={'fontname':'Calibri'}),
            ax.set_xlabel('Choice Angle 0 - 360'), ax.set_ylabel('VT Angle 0 - 360'), ax.set_zlabel('Probability of Choice')
            ax.set_xlim(0, Deg360), ax.set_ylim(0, Deg360), ax.set_zlim(0, 1)
            ax.plot_surface(cangle, vangle, likeprob, cmap=cm.viridis, linewidth=0, antialiased=False), plt.show()
    else:
        x, y = choice[0] - reject[0], choice[1] - reject[1]
        ctheta = slopedegs(x, y, False)
        vangle = np.arange(0, Deg360, steps)
        likeprob = sp.stats.norm.cdf(np.cos(ctheta) * np.cos(vangle*np.pi/180) - np.sin(ctheta) * np.sin(vangle*np.pi/180), 0, stdev)
        if show:
            plt.rcParams['axes.linewidth'] = 1.2
            plt.plot(vangle, likeprob, color = 'rebeccapurple', linewidth = 5)
            plt.fill_between(vangle, likeprob, color='mediumpurple')
            plt.title('Likelihood: p(Choice Given Virtue Threshold)'), plt.xlabel('Virtue Theshold Angles'), plt.ylabel('Probability Density'), plt.show()
    return likeprob

# Like = likelihood([1, 1], [9, 9], loaf = True, show = True)
# print(Like)
# exit()

def posterior(priorwt, likewt, choice, reject, loaf = False, show = False, update = prior(), Title = 'Posterior Over VTs'):
    Prior, Deg360 = update, 360
    if loaf == False:
        Like = likelihood(choice = choice, reject = reject, loaf = False, show = False)
        xax, zax = np.arange(0, Deg360, 1), []
        for i, v in enumerate(xax):
            palpha, pbeta = (1 - Prior[i]) * priorwt, Prior[i] * priorwt
            lalpha, lbeta = (1 - Like[i]) * likewt, Like[i] * likewt
            zax.append(round((pbeta + lbeta) / (priorwt + likewt), 5))
        #     zax.append((pbeta + lbeta) / (priorwt + likewt))
        # zax = normround(zax,5)
        if show:
            plt.rcParams['axes.linewidth'] = 1.2
            plt.plot(xax, zax, color = 'rebeccapurple', linewidth = 5)
            plt.fill_between(xax, zax, color='mediumpurple')
            plt.title('Posterior Over VTs'), plt.xlabel('Virtue Theshold Angles'), plt.ylabel('Probability Density')
            plt.show()
    else:
        Like = likelihood(choice = choice, reject = reject, loaf = True, show = False)
        xax, yax = np.arange(0, Deg360, 1), np.arange(0, Deg360, 1)
        zax = np.zeros(shape=(Deg360, Deg360), dtype=object)
        for i, v in enumerate(Prior):
            for j, w in enumerate(yax):
                palpha, pbeta = (1 - v) * priorwt, v * priorwt
                lalpha, lbeta = (1 - Like[i][j]) * likewt, Like[i][j] * likewt
                zax[i][j] = round((pbeta + lbeta) / (priorwt + likewt), 5)
        if show:
            fig = go.Figure(data=[go.Surface(x=xax, y=yax, z=zax, opacity=0.9, colorscale='Viridis')])
            fig.update_layout(title=Title, scene = dict(
                        xaxis = dict(nticks=10, range=[0,Deg360]),
                        yaxis = dict(nticks=10, range=[0,Deg360]),
                        zaxis = dict(nticks=11, range=[0,1]),
                        xaxis_title='Choice Angle 0 - 360',
                        yaxis_title='VT Angle 0 - 360',
                        zaxis_title='Probability of Choice'),
                        font=dict(family="Calibri", color="Black", size=14))
            fig.show()
    return zax

# Post = posterior(12, 12, [1, 9], [9, 1], loaf = True, show = True, update = prior(60, 50))
# # print(Post)
# exit()

def prediction(priorwt, likewt, choice, reject, nextpay, update, bets = [9, 1, 5], showbeta = False, showpost = False, loaf = False):
    """The updates are coming from postnew, which comes before seeing the choice.  I need to fix this!"""
    A, B = [nextpay[0], nextpay[1]], [nextpay[2], nextpay[3]]
    postold = posterior(priorwt, likewt, choice, reject, False, False, update)
    postnew = posterior(priorwt+1, likewt, A, B, loaf, showpost, postold)
    beta, totalwt = round(np.mean(postnew), 7), priorwt + likewt
    best, worst, sure, alpha = bets[0], bets[1], bets[2], 1 - beta
    cutoff = (sure - worst) / (best - worst)
    subprob = sp.stats.beta.cdf(cutoff, alpha * totalwt, beta * totalwt)
    predcut = sp.stats.beta.cdf(.5, alpha * totalwt, beta * totalwt)
    randpred, randbet = randexcept(predcut), randexcept(subprob)
    predyes = [nextpay[0], nextpay[1]] if predcut > randpred else [nextpay[2], nextpay[3]]
    prednot = [nextpay[2], nextpay[3]] if predcut > randpred else [nextpay[0], nextpay[1]]
    bet = "Bet" if subprob > randbet else "Stay"
    if showbeta:
        X, PMF = np.arange(0, 1, 1/360), []
        for i, v in enumerate(X):
            pofp = sp.stats.beta.pdf(v, alpha * totalwt, beta * totalwt) / 360
            PMF.append(pofp)
        ix, iy = np.linspace(0, cutoff), []
        for i, v in enumerate(ix):
            pofp = sp.stats.beta.pdf(v, alpha * totalwt, beta * totalwt) / 360
            iy.append(pofp)
        plt.rcParams['axes.linewidth'] = 1.2
        fig, ax = plt.subplots()
        ax.plot(X, PMF, color = 'rebeccapurple', linewidth = 5)
        ax.fill_between(X, PMF, color='mediumpurple', edgecolor='rebeccapurple')
        verts = [(0, 0), *zip(ix, iy), (cutoff, 0)]
        poly = Polygon(verts, facecolor='indigo', edgecolor='rebeccapurple')
        ax.add_patch(poly), plt.show()
    return predyes, prednot, bet, postnew

# pred = prediction(2, 2, [3, 3], [6, 1], [8, 4, 9, 1, 'MG01'], prior(260, 30), [9, 1, 7], True, True)
# print(f'The agent prediction {pred[0]}, instead of {pred[1]}, and they chose to {pred[2]}')
# exit()

def normroundmat(matrix, rnd=20):
    row, col, Min = 0, 0, matrix.min()
    while row < numrounds:
        while col < nagents:
            matrix[row][col] = matrix[row][col] - Min
            col = col + 1
        row, col = row + 1, 0
    row, col, Max = 0, 0, matrix.max()
    while row < numrounds:
        while col < nagents:
            matrix[row][col] = matrix[row][col] / Max
            col = col + 1
        row, col = row + 1, 0
    row, col = 0, 0
    while row < numrounds:
        while col < nagents:
            matrix[row][col] = round(matrix[row][col], rnd)
            col = col + 1
        row, col = row + 1, 0   
    return matrix

def trunc(values, decs=0):
    """This truncates values by cutting off excessive decimal places."""
    return np.trunc(values*10**decs)/(10**decs)

def coords(rows, cols):
    """This generates a matrix of xy-coordinates for any number of rows and columns."""
    row, col, coordmat = 0, 0, np.zeros(shape=(rows, cols), dtype=object)
    while row < rows:
        while col < cols:
            coordmat[row][col] = (row, col)
            col = col + 1
        col = 0
        row = row + 1
    return coordmat

def factor_int(n):
    """This provides the factors nearest to the square root of n."""
    val1 = math.ceil(math.sqrt(n))
    val2 = int(n/val1)
    while val2 * val1 != float(n):
        val1 -= 1
        val2 = int(n/val1)
    return n, val1, val2

time1 = timeit.default_timer()

def intig(equation,lim1=0,lim2=1,slices=1000,rnd=5):
    """This function intigrates one-dimensional polynomials.  It is flexible enough to take a list of coefficients A, B, C, and D
    or an equation written as a string with the variable x.  If you feed it the list [A,B,C,D], then it will integrate via calculus.
    If you feed it an equation, like '9*x**5 - sqrt(25)', then it will integrate via Riemann sums.  Note: It only computes over a 
    range from 0 to 1 and divides the integral by the definite integral from 0 to 1 so all outputs are normalized."""
    if isinstance(equation,list):    
        A, B, C, D = equation[0], equation[1], equation[2], equation[3]
        Apoly = A*lim2**4/4 - A*lim1**4/4
        Bpoly = B*lim2**3/3 - B*lim1**3/3
        Cpoly = C*lim2**2/2 - C*lim1**2/2
        Dpoly = D*lim2**1/1 - D*lim1**1/1
        t = A*1**4/4 + B*1**3/3 + C*1**2/2 + D
        area = round((Apoly + Bpoly + Cpoly + Dpoly)/t,rnd)
    if isinstance(equation,str): 
        segs, area, t = np.arange(0,1,1/slices), [], []
        for x in segs:
            t.append(eval(equation)*(1/slices))
        t = round(sum(t),rnd+5)
        for x in segs:
            if x >= lim1 and x < lim2:
                area.append((eval(equation)*(1/slices)))
        area = round(sum(area)/t,rnd)
    return area

def hillcover(equation, nagents, standard=True, rand=True, aps=2, maxloop=200):
    """This is distgen's backup plan.  It generates a grid over the density function (hill), calculates the number
    of cells that cover the hill and repeats until the number of covering cells reaches the desired number of agents
    divided by the agents per square (aps).  Next, it fills the cells with parameter values and groups these values 
    by column.  This takes the same inputs as distgen, except aps and maxloop, which prevents infinite loops."""
    count, nsq, csq, A, B, C = 0, 1, 0, equation[0], equation[1], equation[2]
    if standard:
        """If the equation is in standard form..."""
        A, B, C = equation[0], equation[1], equation[2]
    else:
        """If the equation is in vertex form..."""
        A, H, K = equation[0], equation[1], equation[2]
        B, C = -2*A*H, A*(H**2) + K
    while csq < nagents/aps and nsq < maxloop:
        row, col, sqlen, comat = 0, 0, 1/nsq, coords(nsq, nsq)
        sqmat = np.zeros(shape=(nsq, nsq), dtype=object)
        """Comat---coordinate matrix is a matrix of xy coordinates.  Sqmat---square matrix is a matrix of 0s and 1s.  
        1 if that cell covers the hill.  Sqlen---square length is the side length of each square, which is determined
        by 1 divided by the nsq---number of squares."""
        if rand == False:
            """If rand == False, the algorithm must fill the bins with values right on the bin edges.  The number of 
            bin edges is the number of bins + 1.  This the algorithm must create an extra row/column in the matrices."""
            comat = coords(nsq+1, nsq+1)
            sqmat = np.zeros(shape=(nsq+1, nsq+1), dtype=object)
        for x in comat:
            for y in x:
                """This calculates if the square covers the hill.  If the row number is the number of stories of a building,
                then the height of that room is the number of rows x the height of each room.  The room covers the hill if 
                the room height is greater than the y output of the polynomeal for the same x input---street address."""
                row = (len(comat)-y[0]-1)*sqlen
                col = (y[1]+1)*sqlen
                h = A*col**2+B*col+C
                if h > row:
                    sqmat[y[0], y[1]] = 1
        nsq = nsq + 1
        csq = sum(sum(sqmat))
    bins, params = list(enumerate(sqmat.sum(axis=0), 0)), []
    for x in bins:
        while count < x[1]*aps:
            if rand:
                """It fills the bins with random values between the two edges of each bin."""
                params.append(
                    round(x[0]*sqlen+random.random()*((x[0]+1)*sqlen-x[0]*sqlen), len(str(abs(int(round(nagents, 0)))))+2))
            else:
                """It fills the bins with deterministic values at the edges of the bins."""
                params.append(
                    round((x[0])*sqlen, len(str(abs(int(round(nagents, 0)))))+2))
            count = count + 1
        count = 0
    params = sorted(params)
    while len(params) > nagents:
        params.pop(random.randrange(len(params)))
    random.shuffle(params)
    return params

def distgen(equation, nagents, standard=True, rand=True, nbins=nagents):
    """This function generates the parameter values for n artificial agents based a density function that you specify.
    This equation is a 2nd-order polynomeal with a domain from 0 to 1.  This algorithm creates a discrete set of values
    that conform to the shape of this continious equation.  A histogram or density estimation plot should look like the 
    equation you specified.  Enter a list of coefficients [A, B, C], the number of agents, if the equation is in standard
    or vertex form, if the outputs are noisy, and the prefered numbner of bins."""
    count, ticks, check, delimiters, params = 0, 0, 0, [], []
    if standard:
        """If the equation is in standard form..."""
        A, B, C = equation[0], equation[1], equation[2]
    else:
        """If the equation is in vertex form..."""
        A, H, K = equation[0], equation[1], equation[2]
        B, C = -2*A*H, A*(H**2) + K
    t = (A*1**3)/3 + (B*1**2)/2 + C
    while count < nbins:
        """By integrating a 2nd-order polynomeal, the algorithm raises it to a 3rd-order 
        polynomeal.  It uses the cubic equation to factor this.  Thank you Justin Nordheim
        for all your friendship and help working out the math for this algorithm!"""
        a, b, c, d = 2*A*nbins, 3*B*nbins, 6*C*nbins, -6*t*count
        p, r = -b/(3*a), c/(3*a)
        q = p**3+((b*c-3*a*d)/(6*a**2))
        x = (q+(q**2+(r-p**2)**3)**(1/2))**(1/3) + \
            (q-(q**2+(r-p**2)**3)**(1/2))**(1/3)+p
        np.asarray(delimiters.append(
            round(x.real, len(str(abs(int(round(nbins, 0)))))+2)))
        if delimiters[count] < delimiters[count-1] or delimiters[count] < 0 or delimiters[count] > 1 or t <= 0:
            """These are all the conditions for which this algorithm has failed, and thus must
            revert to backup plan---the hillcover method."""
            count, delimiters = nbins + 1, []
        count = count + 1
    if delimiters == []:
        """If the equation does not have real roots, this above calculus method will fail.  
        Thus the algorithm tries the hillcover method, which is less efficient but more robust."""
        params = hillcover(equation, nagents, standard, rand)
        #print("Hillcover Method")
    else:
        listdelim = list(enumerate(delimiters, 0))
        listdelim.append((nbins, 1))
        """Once the delimiters have been created, the algorithm fills the bins with values."""
        for x in listdelim:
            try:
                subsect = (listdelim[x[0]+1][1] - listdelim[x[0]][1]) 
            except: IndexError
            while ticks < nagents/nbins:
                if rand:
                    """It fills the bins with random values between the two edges of each bin."""
                    params.append(round(listdelim[x[0]-1][1]+random.random()*(
                        listdelim[x[0]][1]-listdelim[x[0]-1][1]), len(str(abs(int(round(nagents, 0)))))+2))
                else:
                    """It fills the bins with deterministic values at the edges of the bins."""
                    params.append(round(check, len(str(abs(int(round(nagents, 0)))))+2))
                ticks, check = ticks + 1, x[1] + subsect * (ticks / (nagents/nbins))
            ticks = 0
        params = sorted(params)
        if rand:
            while len(params) > nagents:
                params.pop(random.randrange(len(params)))
        else:
            params.pop(0)
            while len(params) > nagents:
                params.pop()
        #print("Integration Method")
        random.shuffle(params)
    return params

time2 = timeit.default_timer()

"""Params is a list of parameter values waiting to be assigned to agents.  This vector is sent to
the histogram function to visualize the distribution of agent parameter values.  This histogram is
often smooth for large numbers, but jagged for small numbers.  This is a problem with the display, 
not the underlying distribution.  The bin widths must be carefully calibrated to contain the right 
proportion of values.  Distgen[1] and hillcover[1] return the number of bin edges that it used to
create the parameters so that the histogram function can calibrate the appropriate bin width."""
Params = distgen(equation,nagents,standard,rand)

if showhist:
    bins, plt.rcParams['axes.linewidth'] = 50, 1.2
    wbins = [round(x * 1/bins,3) for x in list(range(0,int(bins + 1),1))]

    """Histfig is the histogram displaying the distribution of agent parameters."""
    histfig = plt.hist(x=Params, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
    n, bins, patches= histfig[0], histfig[1], histfig[2]
    plt.grid(axis='y', alpha=0.75), plt.xlabel('Agent Parameter Value'), plt.ylabel('Number of Agents')
    plt.title('Agent Parameter Distribution', fontdict={'fontname':'Calibri'}), plt.text(23, 45, r'$\mu=15, b=3$')
    fracs = n / n.max()

    """This normalizes the data between 0 and 1 for the full range of the colormap.  
    Then it loops through to set the color of each bar according to its height."""
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    """This displays the histogram."""
    maxfreq = n.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10), plt.show()

Power = distgen(powerequ, nagents, standard, rand)

if circular:
    Self, Other, Envy, Guilt, Lucidity, Honesty = [], [], [], [], [], []
    Forgiving, Stochastic, Prior, WtRatio, WtTotal = [], [], [], [], []
    VTtheta = normcircle(vtthetaequ[0], vtthetaequ[1])
    EGtheta = normcircle(egthetaequ[0], egthetaequ[1])
    for i in VTtheta:
        Self.append(round((math.cos(i) + 1) / 2, 5))
        Other.append(round((math.sin(i) + 1) / 2, 5))
        Lucidity.append(round(random.random(), 5))
        Honesty.append(round(random.random(), 5))
        Forgiving.append(round(random.random(), 5))
        Stochastic.append(round(random.random(), 5))
        Prior.append(round(random.random(), 5))
        WtRatio.append(round(random.random(), 5))
        WtTotal.append(round(random.random(), 5))
    for i in EGtheta:
        Envy.append(round((math.cos(i) + 1) / 2, 5))
        Guilt.append(round((math.sin(i) + 1) / 2, 5))
else:
    Envy, Guilt, Lucidity, Honesty, VTtheta, EGtheta = [], [], [], [], [], []
    Forgiving, Stochastic, Prior, WtRatio, WtTotal = [], [], [], [], []
    Self = distgen(selfequ, nagents, standard, rand)
    Other = distgen(otherequ, nagents, standard, rand)
    for i, v in enumerate(Self):
        # VTtheta.append(round(2 * (math.tan(Other[i]/v)**-1) - 1, 5))
        # EGtheta.append(round(2 * (math.tan(Other[i]/v)**-1) - 1, 5))
        x = (2 * v) - 1
        y = (2 * Other[i] - 1)
        VT = round(slopedegs(x, y, True) / 360, 5)
        VTtheta.append(VT)
        Lucidity.append(round(random.random(), 5))
        Honesty.append(round(random.random(), 5))
        Envy.append(round(random.random(), 5))
        Guilt.append(round(random.random(), 5))
        Forgiving.append(round(random.random(), 5))
        Stochastic.append(round(random.random(), 5))
        Prior.append(round(random.random(), 5))
        WtRatio.append(round(random.random(), 5))
        WtTotal.append(round(random.random(), 5))
    for i, v in enumerate(Envy):
        x = (2 * v) - 1
        y = (2 * Guilt[i] - 1)
        EG = round(slopedegs(x, y, True) / 360, 5)
        EGtheta.append(EG)

time3 = timeit.default_timer()

def Agents(params1,params2=0,params3=0,params4=0,params5=0,params6=0,params7=0,params8=0,params9=0,params10=0,params11=0,params12=0,params13=0,params14=0):
    Agents = []
    for x in range(len(params1)):
        Agents.append({'Power' : params1[x], 'Self' : params2[x], 'Other' : params3[x], 'VTtheta' : params4[x], 'Envy' : params5[x], 
            'Guilt' : params6[x], 'EGtheta' : params7[x], 'Lucidity' : params8[x], 'Honesty' : params9[x], 'Forgiving' : params10[x], 
                'Stochastic' : params11[x], 'Prior' : params12[x], 'WtRatio' : params13[x], 'WtTotal' : params14[x]})
    Agents = list(enumerate(Agents,0))
    return Agents

Agents = Agents(Power, Self, Other, VTtheta, Envy, Guilt, EGtheta, Lucidity, Honesty, Forgiving, Stochastic, Prior, WtRatio, WtTotal)

time4 = timeit.default_timer() #,print(Agents), exit()

def randmatch(nagents=nagents):
    """This produces a list of randomly matched pairs of agents"""
    if nagents % 2 > 0:
        """If there is an odd number of agents, remove the leftover person from the list."""
        numlist, matchlist = list(range(0, nagents - 1)), []
    else:
        numlist, matchlist = list(range(0, nagents)), []
    random.shuffle(numlist)
    """This generates a list of random pairs by sampling a sequential list without replacement."""
    while len(numlist) > 0:
        first = numlist.pop(0)
        secon = numlist.pop()
        pair = [first, secon]
        matchlist.append(pair)
    return matchlist

time5 = timeit.default_timer() #,print(randmatch(nagents+2)), exit()

def chooselist(matchlist=randmatch()):
    """This takes a list of agents and sorts them into chooser and predictor based on their power disparity."""
    chooselist = []
    for i in matchlist:
        first, secon = i[0], i[1]
        power1, power2 = Agents[first][1]['Power'], Agents[secon][1]['Power']
        rando, powerdiff = randexcept(.5), power1 / (power1 + power2)
        if rando > powerdiff:
            first, secon = secon, first
        chooselist.append([first, secon])
    return chooselist

time6 = timeit.default_timer()

def betpayoffs(best = 9, worst = 1):
    # sure = random.choice(list(range(worst,best+1)))
    sure = random.choice(list(range(worst+1,best)))
    return [best, worst, sure]

# print(betpayoffs())
# exit()

def givepaystruct(chooselist=chooselist(), paylist = sellist(nagents), randomize = RandPay):
    """This assigns payoff structures to the list of paired agents."""
    steps, payindx, newpays, hereyougo = 0, 0, [], []
    if randomize:
        paylist = MGPayoffs9
        random.shuffle(paylist)
        while steps < len(chooselist):
            As, Ao, Bs, Bo, Lab = paylist[payindx][1], paylist[payindx][2], \
                paylist[payindx][3], paylist[payindx][4], paylist[payindx][0]
            newpays.append([As, Ao, Bs, Bo, Lab])
            steps = steps + 1
            payindx = payindx + 1 if payindx < len(paylist) - 1 else 0
        steps, paylist = 0, newpays
    while steps < len(chooselist):
        bets = betpayoffs()
        pair, pay = chooselist[steps], paylist[steps]
        hereyougo.append([pair, pay, bets])
        steps = steps + 1
    return hereyougo

time7 = timeit.default_timer() 
# givepays = givepaystruct()
# print(givepays)
# exit()

def switcher(As, Ao, Bs, Bo, pswitch):
    rando, rndx = random.random(), int(np.random.choice(5,1))
    switch = [[As, Ao, Bo, Bs], [As, Bo, Bs, Ao], [As, Bs, Ao, Bo], 
        [Bo, Ao, Bs, As], [Bs, Ao, As, Bo], [Ao, As, Bs, Bo]]
    if pswitch > rando:
        switchlist = switch[rndx]
    else:
        switchlist = [As, Ao, Bs, Bo]
    return switchlist

time8 = timeit.default_timer() #,print(switcher(1,2,3,4,.9)), exit()


# def agentfunctsimp(payoffs, chooser, predictor, chooparams, predparams, worldparams, bets = [9, 1, 5], now=0, history=None):
#     Paylabel, Aself, Aothr, Bself, Bothr = payoffs[4], payoffs[0], payoffs[1], payoffs[2], payoffs[3]
#     pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
#     s, o = 2 * chooparams['Self'] - 1, 2 * predparams['Other'] - 1
#     g, e = 2 * chooparams['Guilt'] - 1, 2 * predparams['Envy'] - 1
#     s, o, g, e = s * sweight, o * oweight, g * gweight, e * eweight
#     if now > 0:
#         cplist, pclist = {}, {}
#         for i in list(range(0,now)):
#             """This is the history of prior interactions between the current chooser and predictor."""
#             if chooser == history[i][chooser]['Chooser'][1] and predictor == history[i][chooser]['Predictor'][1]:
#                 present = 'Round ' + str(history[i][chooser]['Round'])
#                 cplist[present] = {'Chooser': chooser, 'Predictor': predictor, \
#                     'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject'], \
#                         'Posterior' : history[i][chooser]['Posterior']}
#             if chooser == history[i][chooser]['Predictor'][1] and predictor == history[i][chooser]['Chooser'][1]:
#                 present = 'Round ' + str(history[i][chooser]['Round'])
#                 pclist[present] = {'Chooser': predictor, 'Predictor': chooser, \
#                     'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
#         if cplist != {}:
#             """Agents adjust their VTs to mirror the past action, but vary in how flexible their VT is."""
#             choosum = []
#             for i, v in enumerate(cplist.values()):
#                 x, y = v['Choice'][0] - v['Reject'][0], v['Choice'][1] - v['Reject'][1]
#                 choosum.append(slopedegs(x, y, True))
#             meets, choosum = len(choosum), sum(choosum)
#             pS, pO = math.cos(angletheta(choosum/meets,True)), math.sin(angletheta(choosum/meets,True))
#             pA = pS * (Aself - Bself) + pO * (Aothr - Bothr) 
#             pB = pS * (Bself - Aself) + pO * (Bothr - Aothr) 
#             if pA == pB:
#                 pA, pB = randexcept(0.5), randexcept(0.5)
#             predy = [Aself, Aothr] if pA > pB else [Bself, Bothr]
#             predn = [Bself, Bothr] if pA > pB else [Aself, Aothr]
#             bet = [Aself, Aothr] if random.random() > .5 else [Bself, Bothr]
#         else:
#             predy = [Aself, Aothr] if random.random() > .5 else [Bself, Bothr]
#             predn = [Bself, Bothr] if random.random() > .5 else [Aself, Aothr]
#             bet = [Aself, Aothr] if random.random() > .5 else [Bself, Bothr]
#     else:
#         predy = [Aself, Aothr] if random.random() > .5 else [Bself, Bothr]
#         predn = [Bself, Bothr] if random.random() > .5 else [Aself, Aothr]
#         bet = [Aself, Aothr] if random.random() > .5 else [Bself, Bothr]
#     A = s * (Aself - Bself) + o * (Aothr - Bothr) - g * (Aself - Aothr) - e * (Aothr - Aself)
#     B = s * (Bself - Aself) + o * (Bothr - Aothr) - g * (Bself - Bothr) - e * (Bothr - Bself)
#     if A == B:
#         A, B = randexcept(0.5), randexcept(0.5)
#     choice = [Aself, Aothr] if A > B else [Bself, Bothr]
#     reject = [Bself, Bothr] if A > B else [Aself, Aothr]
#     poster = list(range(1,360))
#     return choice, reject, predy, predn, bet, poster


# def agentfunct(payoffs, chooser, predictor, chooparams, predparams, worldparams, bets = [9, 1, 5], now=0, history=None):
#     Paylabel, Aself, Aothr, Bself, Bothr = payoffs[4], payoffs[0], payoffs[1], payoffs[2], payoffs[3]
#     pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
#     s, o = 2 * chooparams['Self'] - 1, 2 * predparams['Other'] - 1
#     g, e = 2 * chooparams['Guilt'] - 1, 2 * predparams['Envy'] - 1
#     s, o, g, e = s * sweight, o * oweight, g * gweight, e * eweight
#     if now > 0:
#         cplist, pclist = {}, {}
#         for i in list(range(0,now)):
#             """This is the history of prior interactions between the current chooser and predictor."""
#             if chooser == history[i][chooser]['Chooser'][1] and predictor == history[i][chooser]['Predictor'][1]:
#                 present = 'Round ' + str(history[i][chooser]['Round'])
#                 cplist[present] = {'Chooser': chooser, 'Predictor': predictor, \
#                     'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject'], \
#                         'Posterior' : history[i][chooser]['Posterior']}
#             if chooser == history[i][chooser]['Predictor'][1] and predictor == history[i][chooser]['Chooser'][1]:
#                 present = 'Round ' + str(history[i][chooser]['Round'])
#                 pclist[present] = {'Chooser': predictor, 'Predictor': chooser, \
#                     'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
#         if cplist != {}:
#             """Agents adjust their VTs to mirror the past action, but vary in how flexible their VT is."""
#             choosum = []
#             for i, v in enumerate(cplist.values()):
#                 x, y = v['Choice'][0] - v['Reject'][0], v['Choice'][1] - v['Reject'][1]
#                 choosum.append(slopedegs(x, y, True))
#             meets, choosum = len(choosum), sum(choosum)
#             xweight, yweight = math.cos(angletheta(choosum/meets,True)), math.sin(angletheta(choosum/meets,True))
#             s, o = (s * vweight + xweight * meets) / (vweight + meets), (o * vweight + yweight * meets) / (vweight + meets)
#             cpolist = list(cplist.items())
#             oldpost, oldchoice, oldreject = cpolist[-1][1]['Posterior'], cpolist[-1][1]['Choice'], cpolist[-1][1]['Reject']
#         else:
#             oldpost, oldchoice, oldreject = prior(int(predparams['Prior'] * 360)), [2, 2], [1, 1]    
#     else:
#         oldpost, oldchoice, oldreject = prior(int(predparams['Prior'] * 360)), [2, 2], [1, 1]
#     A = s * (Aself - Bself) + o * (Aothr - Bothr) - g * (Aself - Aothr) - e * (Aothr - Aself)
#     B = s * (Bself - Aself) + o * (Bothr - Aothr) - g * (Bself - Bothr) - e * (Bothr - Bself)
#     if A == B:
#         A, B = randexcept(0.5), randexcept(0.5)
#     choice = [Aself, Aothr] if A > B else [Bself, Bothr]
#     reject = [Bself, Bothr] if A > B else [Aself, Aothr]
#     # priorwt = predparams['WtRatio'] * predparams['WtTotal'] + age
#     # likewt = (1 - predparams['WtRatio']) * predparams['WtTotal'] if now > 0 else 0
#     priorwt, likewt = predparams['WtRatio'] * predparams['WtTotal'] + now + age, 50
#     predicts = prediction(priorwt, likewt, oldchoice, oldreject, payoffs, oldpost, bets)
#     predy, predn, bet = predicts[0], predicts[1], predicts[2]
#     poster = posterior(priorwt, likewt, oldchoice, oldreject, False, False, oldpost)
#     Bprior = int(predparams['Prior'] * 360)
#     poster = posterior(priorwt, likewt, oldchoice, oldreject, False, False, oldpost, \
#         f'Agent {predictor}s Model of Agent {chooser}.  Prior Angle: {Bprior}') \
#         if now == numrounds - 1 and showposts and nagents < 22 else poster
#     return choice, reject, predy, predn, bet, poster

time9 = timeit.default_timer()

# def stimresp(numrounds = numrounds, nagents = nagents, Agents = Agents, givepaystruct = givepaystruct(), 
#     worldparams = WorldParameters, allinfo = False, now = 0, history = None):
#     history = np.zeros(shape=(numrounds, nagents), dtype=object) if now == 0 else history
#     pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
#     for i in givepaystruct:
#         Chooser, Predictor = i[0][0], i[0][1]
#         best, worst, sure = i[2][0], i[2][1], i[2][2]
#         Paylabel, Aself, Aothr, Bself, Bothr = i[1][4], i[1][0], i[1][1], i[1][2], i[1][3]
#         switchlist = switcher(Aself, Aothr, Bself, Bothr, pswitch)
#         chooparams, predparams = Agents[i[0][0]][1], Agents[i[0][1]][1]
#         if now > 0:
#             choomet = history[now-1][Chooser]['Chooser'][1] if history[now-1][Chooser]['Chooser'][1] != Chooser else history[now-1][Chooser]['Predictor'][1]
#             predmet = history[now-1][Predictor]['Chooser'][1] if history[now-1][Predictor]['Chooser'][1] != Predictor else history[now-1][Predictor]['Predictor'][1]
#             # print("Round: ", now, " Choo: ", Chooser, " Earlier vs ", choomet)
#             # print("Round: ", now, " Pred: ", Predictor, " Earlier vs ", predmet)
#         # Notice that I changed agentfunct to angentfunctsimp to test out simpler prediciton methods
#         choice, reject, predy, predn, bet, Posterior = agentfunctsimp(i[1], Chooser, Predictor, chooparams, predparams, worldparams, bets = [best, worst, sure], now=now, history=history)
#         pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
#         pred_accuracy = 1 if predy == choice else 0
#         if bet == "Bet":
#             pts_bet = best if predy == choice else worst
#         else:
#             pts_bet = sure
#         actions = {"Round" : now, "Chooser" : ['Agent' + str(Chooser), Chooser], "Predictor" : ['Agent' + str(Predictor), Predictor], \
#             "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, "Reject" : reject, \
#             "Prediction" : predy, "Not-Pred" : predn, "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor, "Bet-Pay" : i[2], \
#             "Bet" : bet, "Betting-Points" : pts_bet, "Prediction-Accuracy" : pred_accuracy, "Posterior" : Posterior}
#         if allinfo:
#             actions["Chooser-Params"], actions["Predictor-Params"], actions["World-Params"] = chooparams, predparams, worldparams
#         history[now][Chooser], history[now][Predictor] = actions, actions
#         # print("Round: ", now, " Payoffs: ", i[1], " Chooser: ", i[0][0], " Predictor: ", i[0][1], " Pts-Choo: ", pts_chooser, " Pts-Pred: ", pts_predictor)
#         # print(f'R{now}: {i[1]}, Chooser: {i[0][0]} earned {pts_chooser} and Predictor: {i[0][1]} earned {pts_predictor}, who chose to {bet}, thus earning {pts_bet}')
#     return history

time10 = timeit.default_timer() 
# stimresp = stimresp() 
# print(stimresp)
# exit()

def agentfunctog(payoffs, chooser, predictor, chooparams, predparams, worldparams, now=0, history=None):
    Paylabel, Aself, Aothr, Bself, Bothr = payoffs[4], payoffs[0], payoffs[1], payoffs[2], payoffs[3]
    pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
    s, o = 2 * chooparams['Self'] - 1, 2 * predparams['Other'] - 1
    g, e = 2 * chooparams['Guilt'] - 1, 2 * predparams['Envy'] - 1
    s, o, g, e = s * sweight, o * oweight, g * gweight, e * eweight
    if now > 0:
        cplist, pclist = {}, {}
        for i in list(range(0,now)):
            """This is the history of prior interactions between the current chooser and predictor."""
            if chooser == history[i][chooser]['Chooser'][1] and predictor == history[i][chooser]['Predictor'][1]:
                present = 'Round ' + str(history[i][chooser]['Round'])
                cplist[present] = {'Chooser': chooser, 'Predictor': predictor, \
                    'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
            if chooser == history[i][chooser]['Predictor'][1] and predictor == history[i][chooser]['Chooser'][1]:
                present = 'Round ' + str(history[i][chooser]['Round'])
                pclist[present] = {'Chooser': predictor, 'Predictor': chooser, \
                    'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
        if cplist != {}:
            """Agents adjust their VTs to mirror the past action, but vary in how flexible their VT is."""
            choosum = []
            for i, v in enumerate(cplist.values()):
                x, y = v['Choice'][0] - v['Reject'][0], v['Choice'][1] - v['Reject'][1]
                choosum.append(slopedegs(x, y, True))
            meets, choosum = len(choosum), sum(choosum)
            xweight, yweight = math.cos(angletheta(choosum/meets,True)), math.sin(angletheta(choosum/meets,True))
            s, o = (s * vweight + xweight * meets) / (vweight + meets), (o * vweight + yweight * meets) / (vweight + meets)
    A = s * (Aself - Bself) + o * (Aothr - Bothr) - g * (Aself - Aothr) - e * (Aothr - Aself)
    B = s * (Bself - Aself) + o * (Bothr - Aothr) - g * (Bself - Bothr) - e * (Bothr - Bself)
    if A == B:
        A, B = randexcept(0.5), randexcept(0.5)
    choice = [Aself, Aothr] if A > B else [Bself, Bothr]
    reject = [Bself, Bothr] if A > B else [Aself, Aothr]
    return choice, reject

def stimrespog(numrounds = numrounds, nagents = nagents, Agents = Agents, givepaystruct = givepaystruct(), 
    worldparams = WorldParameters, allinfo = False, now = 0, history = None):
    history = np.zeros(shape=(numrounds, nagents), dtype=object) if now == 0 else history
    pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
    for i in givepaystruct:
        Chooser, Predictor = i[0][0], i[0][1]
        Paylabel, Aself, Aothr, Bself, Bothr = i[1][4], i[1][0], i[1][1], i[1][2], i[1][3]
        switchlist = switcher(Aself, Aothr, Bself, Bothr, pswitch)
        chooparams, predparams = Agents[i[0][0]][1], Agents[i[0][1]][1]
        if now > 0:
            choomet = history[now-1][Chooser]['Chooser'][1] if history[now-1][Chooser]['Chooser'][1] != Chooser else history[now-1][Chooser]['Predictor'][1]
            predmet = history[now-1][Predictor]['Chooser'][1] if history[now-1][Predictor]['Chooser'][1] != Predictor else history[now-1][Predictor]['Predictor'][1]
            # print("Round: ", now, " Choo: ", Chooser, " Earlier vs ", choomet)
            # print("Round: ", now, " Pred: ", Predictor, " Earlier vs ", predmet)
        choice, reject = agentfunctog(i[1], Chooser, Predictor, chooparams, predparams, worldparams, now=now, history=history)
        pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
        actions = {"Round" : now, "Chooser" : ['Agent' + str(Chooser), Chooser], "Predictor" : ['Agent' + str(Predictor), Predictor], \
            "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, "Reject" : reject, \
                "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
        if allinfo:
            actions["Chooser-Params"], actions["Predictor-Params"], actions["World-Params"] = chooparams, predparams, worldparams
        history[now][Chooser], history[now][Predictor] = actions, actions
        # print("Round: ", now, " Payoffs: ", i[1], " Chooser: ", i[0][0], " Predictor: ", i[0][1], " Pts-Choo: ", pts_chooser, " Pts-Pred: ", pts_predictor)
    return history

def multirnd(numrounds = numrounds, nagents = nagents, Agents = Agents, worldparams = WorldParameters, allinfo = False):
    now, record = 0, np.zeros(shape=(numrounds, nagents), dtype=object)
    pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
    while now < numrounds:
        record = stimrespog(givepaystruct = givepaystruct(chooselist = chooselist(matchlist = randmatch()), 
            paylist = sellist(nagents = nagents)), now = now, history = record)
        now = now + 1
    return record

time11 = timeit.default_timer() 
record = multirnd()
# print("Space")
# print("Space")
# print(record)
# exit()

def score(record = multirnd(), norm = False):
    running = np.zeros(shape=(numrounds, nagents), dtype=object)
    for i, v in enumerate(record):
        for j, w in enumerate(v):
            if j == w['Chooser'][1]:
                running[i][j] = w['Points-Chooser'] 
            else:
                running[i][j] = w['Points-Predictor'] 
    rsum = running.sum(axis=0)
    rsum = normround(rsum,4) if norm else rsum
    return running, rsum

def fairscore(record = multirnd(), norm = False):
    """This provides a matrix of chooser points only and a matrix of predictor points only.
    The ratio between chooser and predictor points is a way to measure the cooperation rate."""
    choopts = np.zeros(shape=(numrounds, nagents), dtype=object)
    predpts = np.zeros(shape=(numrounds, nagents), dtype=object)
    for i, v in enumerate(record):
        for j, w in enumerate(v):
            if j == w['Chooser'][1]:
                choopts[i][j], predpts[i][j] = w['Points-Chooser'], 'N'
            else:
                choopts[i][j], predpts[i][j] = 'N', w['Points-Predictor'], 
    return choopts, predpts

def predaccur(record = multirnd(), norm = False):
    """This returns the prediction accuracy and betting points for all agents across all rounds."""
    predac = np.zeros(shape=(numrounds, nagents), dtype=object)
    betpts = np.zeros(shape=(numrounds, nagents), dtype=object)
    for i, v in enumerate(record):
        for j, w in enumerate(v):
            if j == w['Chooser'][1]:
                predac[i][j], betpts[i][j] = 'N', 'N'
            else:
                predac[i][j] = w['Prediction-Accuracy'] 
                betpts[i][j] = w['Betting-Points']
    return predac, betpts

def runavg(matrix, final = False):
    """This takes a matrix and returns the running averages and running totals of each column.  
    It ignores strings, which is crucial for calculating prediction accuracy, which fills cells
    with 'N' for the chooser and numbers for the predictor.  If final, it returns the last row."""
    numnum = np.zeros(shape=(len(matrix), len(matrix[0])), dtype=object)
    runavs = np.zeros(shape=(len(matrix), len(matrix[0])), dtype=object)
    count, runsum = 0, np.zeros(shape=(len(matrix), len(matrix[0])), dtype=object)
    while count < len(matrix):
        for i, v in enumerate(matrix[count]):
            if count == 0:
                runsum[count][i] = v if isinstance(v, (int, float)) else runsum[count][i]
                numnum[count][i] = 1 if isinstance(v, (int, float)) else 0
                runavs[count][i] = round(runsum[count][i] / 1, 3)
            else:
                runsum[count][i] = runsum[count - 1][i] + v if isinstance(v, (int, float)) else runsum[count - 1][i]
                numnum[count][i] = numnum[count - 1][i] + 1 if isinstance(v, (int, float)) else numnum[count - 1][i]
                runavs[count][i] = round(runsum[count][i] / numnum[count][i], 3) if numnum[count][i] != 0 else round(runsum[count][i] / 1, 3)
        count = count + 1
    runsum = runsum[-1] if final else runsum
    runavs = runavs[-1] if final else runavs
    return runavs, runsum

def predbettime(average = True, individuals = False, show = True, best = 9, worst = 1):
    predra, predrs = runavg(predaccur()[0], final = individuals)
    betra, betrs = runavg(predaccur()[1], final = individuals)  
    rndtime, accvtime, betvtime = list(range(0,numrounds)), [], []
    for i, j in zip(predra, betra):
        accvtime.append(round(sum(i) / len(i), 3))
        betvtime.append(round((sum(j) / len(j) - worst) / (best - worst), 3))
    if show:
        plt.rcParams['axes.linewidth'] = 1.2
        x1, x2 = rndtime, rndtime
        y1, y2 = accvtime, betvtime
        plt.plot(x1, y1, color = 'rebeccapurple', linewidth = 5, label = 'Prediction')
        plt.plot(x2, y2, color = 'crimson', linewidth = 5, label = 'Betting')
        plt.title('Average Prediction Accuracy Across Rounds'), plt.xlabel('n Rounds'), plt.ylabel('Prediction Accuracy')
        plt.axis([0, numrounds - 1, 0, 1])
        plt.legend(), plt.show()
    return accvtime, betvtime


# predgraph = predbettime()


time12 = timeit.default_timer() #,runmat, runsum = score(), print(runmat), print(runsum), exit()

def xyz(xaxis = 'Self', yaxis = 'Other', zaxis = 'Final Points'):
    runmat, runsum = score()
    count, x, y = 0, [], []
    while count < nagents:
        xpoint = Agents[count][1][xaxis]
        ypoint = Agents[count][1][yaxis]
        x.append(xpoint)
        y.append(ypoint)
        count = count + 1
    if zaxis == 'Points':
        z = runmat
    elif zaxis == 'Final Points':
        z = runsum
    else:
        z = runsum
    return x, y, z

x, y, z = xyz(xaxis = Axes['X Axis'], yaxis = Axes['Y Axis'], zaxis = Axes['Z Axis'])

if givetimes:
    print('Time 0 -- 1: ', time1 - time0, (time1 - time0) / (time12 - time0))
    print('Time 1 -- 2: ', time2 - time1, (time2 - time1) / (time12 - time0))
    print('Time 2 -- 3: ', time3 - time2, (time3 - time2) / (time12 - time0))
    print('Time 3 -- 4: ', time4 - time3, (time4 - time3) / (time12 - time0))
    print('Time 4 -- 5: ', time5 - time4, (time5 - time4) / (time12 - time0))
    print('Time 5 -- 6: ', time6 - time5, (time6 - time5) / (time12 - time0))
    print('Time 6 -- 7: ', time7 - time6, (time7 - time6) / (time12 - time0))
    print('Time 7 -- 8: ', time8 - time7, (time8 - time7) / (time12 - time0))
    print('Time 8 -- 9: ', time9 - time8, (time9 - time8) / (time12 - time0))
    print('Time 9 -- 10: ', time10 - time9, (time10 - time9) / (time12 - time0))
    print('Time 10 -- 11: ', time11 - time10, (time11 - time10) / (time12 - time0))
    print('Time 11 -- 12: ', time12 - time11, (time12 - time11) / (time12 - time0))
    print('Time Total: ', time12 - time0)

    print('n Agents: ', nagents, ' n Rounds: ', numrounds, ' Runtime: ', round(time12 - time0, 5))
    exit()

if givetime:
    print('n Agents: ', nagents, ' n Rounds: ', numrounds, ' Runtime: ', round(time12 - time0, 5))

if showgalaxy:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = normround(z,5) if z.ndim == 1 else normroundmat(z,5)
    colors = [x * 100 for x in colors]
    ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
    ax.set_xlabel(Axes['X Label']), ax.set_ylabel(Axes['Y Label']), ax.set_zlabel(Axes['Z Label'])
    if normaxes or chronicle:
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
    if chronicle:
        ax.clear()
        z = xyz()[2][0] 
        colors = [x * 100 for x in z]
        plt.subplots_adjust(bottom=0.25)
        ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
        evo = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightsalmon')
        evolution = Slider(ax = evo, label = "Evolution", valmin = 0, valmax = numrounds - 1, \
            valinit = 0, valstep = 1, orientation= 'horizontal', color = 'coral')
        ax.set_xlabel(Axes['X Label']), ax.set_ylabel(Axes['Y Label']), ax.set_zlabel(Axes['Z Label'])
        ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)

        def update(val):
            ax.clear()
            z = xyz()[2][val] 
            colors = [x * 100 for x in z]
            ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
            ax.set_xlabel(Axes['X Label']), ax.set_ylabel(Axes['Y Label']), ax.set_zlabel(Axes['Z Label'])
            ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1), fig.canvas.draw()

        evolution.on_changed(update)
    plt.show()

exit()













































Choices = [[[3, 3], [6, 1], [8, 4, 9, 1, 'MG01']], 
[[8, 4], [9, 1], [1, 2, 9, 8, 'MG02']], 
[[1, 2], [9, 8], [6, 9, 4, 1, 'MG03']], 
[[4, 1], [6, 9], [5, 5, 1, 1, 'MG04']], 
[[1, 1], [5, 5], [9, 1, 1, 9, 'MG05']], 
[[1, 9], [9, 1], [1, 9, 9, 1, 'MG06']], 
[[1, 9], [9, 1], [3, 6, 9, 5, 'MG07']], 
[[3, 6], [9, 5], [1, 5, 9, 5, 'MG08']], 
[[1, 5], [9, 5], [8, 2, 2, 8, 'MG09']], 
[[2, 8], [8, 2], [7, 3, 5, 1, 'MG10']], 
[[5, 1], [7, 3], [4, 2, 8, 2, 'MG11']], 
[[8, 4], [9, 1], [1, 2, 9, 8, 'MG12']], 
[[1, 2], [9, 8], [6, 9, 4, 1, 'MG13']], 
[[4, 1], [6, 9], [5, 5, 1, 1, 'MG14']], 
[[1, 1], [5, 5], [9, 1, 1, 9, 'MG15']], 
[[1, 9], [9, 1], [1, 9, 9, 1, 'MG16']], 
[[1, 9], [9, 1], [3, 6, 9, 5, 'MG17']], 
[[3, 6], [9, 5], [1, 5, 9, 5, 'MG18']], 
[[1, 5], [9, 5], [8, 2, 2, 8, 'MG19']], 
[[2, 8], [8, 2], [7, 3, 5, 1, 'MG20']], 
[[5, 1], [7, 3], [4, 2, 8, 2, 'MG21']], 
[[8, 4], [9, 1], [1, 2, 9, 8, 'MG22']], 
[[1, 2], [9, 8], [6, 9, 4, 1, 'MG23']], 
[[4, 1], [6, 9], [5, 5, 1, 1, 'MG24']], 
[[1, 1], [5, 5], [9, 1, 1, 9, 'MG25']], 
[[1, 9], [9, 1], [1, 9, 9, 1, 'MG26']], 
[[1, 9], [9, 1], [3, 6, 9, 5, 'MG27']], 
[[3, 6], [9, 5], [1, 5, 9, 5, 'MG28']], 
[[1, 5], [9, 5], [8, 2, 2, 8, 'MG29']], 
[[2, 8], [8, 2], [7, 3, 5, 1, 'MG30']], 
[[5, 1], [7, 3], [4, 2, 8, 2, 'MG31']], 
[[8, 4], [9, 1], [1, 2, 9, 8, 'MG32']], 
[[1, 2], [9, 8], [6, 9, 4, 1, 'MG33']], 
[[4, 1], [6, 9], [5, 5, 1, 1, 'MG34']], 
[[1, 1], [5, 5], [9, 1, 1, 9, 'MG35']], 
[[1, 9], [9, 1], [1, 9, 9, 1, 'MG36']], 
[[1, 9], [9, 1], [3, 6, 9, 5, 'MG37']], 
[[3, 6], [9, 5], [1, 5, 9, 5, 'MG38']], 
[[1, 5], [9, 5], [8, 2, 2, 8, 'MG39']], 
[[2, 8], [8, 2], [7, 3, 5, 1, 'MG40']], 
[[5, 1], [7, 3], [4, 2, 8, 2, 'MG41']], 
[[4, 2], [8, 2], [1, 4, 7, 3, 'MG42']]]

priorwt, likewt = 2, 2
for i, v in enumerate(Choices):
    choice, reject, nextpay = v[0], v[1], v[2]
    if i == 0:
        pred = prediction(priorwt+i, likewt, choice, reject, nextpay, prior(60, 50), [9, 1, 5], False)
    elif i > 0 and i < len(Choices) - 1:
        pred = prediction(priorwt+i, likewt, choice, reject, nextpay, pred[3], [9, 1, 5], False)
    else:
        pred = prediction(priorwt+i, likewt, choice, reject, nextpay, pred[3], [9, 1, 5], showbeta=False, showpost=False, loaf=False)
        # pred = prediction(priorwt+i, likewt, choice, reject, nextpay, pred[3], [9, 1, 5], showbeta=True, showpost=True, loaf=True)
    # print(pred[0], pred[1], pred[2])

# exit()


# def actionlist(givepaystruct = givepaystruct(), Agents = Agents, attachparams = False, worldparams = WorldParameters, rnd = 0, history=None):
#     actionlist, pswitch, pcatch = [], worldparams['p(switch)'], worldparams['p(catch)']
#     for i in givepaystruct:
#         Paylabel, Aself, Aothr, Bself, Bothr = i[1][4], i[1][0], i[1][1], i[1][2], i[1][3]
#         switchlist = switcher(Aself, Aothr, Bself, Bothr, pswitch)
#         if circular:
#             choice = virtuechoice(Agents[i[0][0]-1][1]['VTtheta'], [Aself, Aothr, Bself, Bothr], Angle = False)[0]
#             reject = virtuechoice(Agents[i[0][0]-1][1]['VTtheta'], [Aself, Aothr, Bself, Bothr], Angle = False)[1]
#         else:
#             s, o = round(2 * Agents[i[0][0]-1][1]['Self'] - 1, 5), round(2 * Agents[i[0][0]-1][1]['Other'] - 1, 5)
#             A = round(s * (Aself - Bself) + o * (Aothr - Bothr), 5)
#             B = round(s * (Bself - Aself) + o * (Bothr - Aothr), 5)
#             if A == B:
#                 A, B = randexcept(0.5), randexcept(0.5)
#             choice = [Aself, Aothr] if A > B else [Bself, Bothr]
#             reject = [Bself, Bothr] if A > B else [Aself, Aothr]
#         pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
#         if attachparams:
#             actions = {"Round" : rnd, "Chooser" : ['Agent' + str(i[0][0]), i[0][0]], "Predictor" : ['Agent' + str(i[0][1]), i[0][1]], \
#                 "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, "Points-Chooser" : pts_chooser, \
#                     "Points-Predictor" : pts_predictor, "Chooser-Params" : Agents[i[0][0]-1][1], "Predictor-Params" : Agents[i[0][1]-1][1]}    
#         else:
#             actions = {"Round" : rnd, "Chooser" : ['Agent' + str(i[0][0]), i[0][0]], "Predictor" : ['Agent' + str(i[0][1]), i[0][1]], \
#                     "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, \
#                         "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
#         actionlist.append(actions) 
#     return actionlist

# def multiround2(numrounds = numrounds, nagents = nagents):
#     """This iterates the game over n rounds.  It generates new lists of matches and payoffs each round."""
#     now, allrounds = 0, []
#     while now < numrounds:
#         if circular:
#             allrounds.append(actionlist(givepaystruct=givepaystruct(chooselist=chooselist(\
#                 matchlist=randmatch()), paylist=sellist360), rnd=now))    
#         else:
#             allrounds.append(actionlist(givepaystruct=givepaystruct(chooselist=chooselist(\
#                 matchlist=randmatch()), paylist=sellist(nagents=nagents)), rnd=now, history=[]))
#         now = now + 1
#     return allrounds

def actionlist(givepaystruct = givepaystruct(), Agents = Agents, attachparams = False, worldparams = WorldParameters):
    actionlist, pswitch, pcatch = [], worldparams['p(switch)'], worldparams['p(catch)']
    for i in givepaystruct:
        Paylabel, Aself, Aothr, Bself, Bothr = i[1][4], i[1][0], i[1][1], i[1][2], i[1][3]
        switchlist = switcher(Aself, Aothr, Bself, Bothr, pswitch)
        if circular:
            choice = virtuechoice(Agents[i[0][0]-1][1]['VTtheta'], [Aself, Aothr, Bself, Bothr], Angle = False)[0]
            reject = virtuechoice(Agents[i[0][0]-1][1]['VTtheta'], [Aself, Aothr, Bself, Bothr], Angle = False)[1]
        else:
            s, o = round(2 * Agents[i[0][0]-1][1]['Self'] - 1, 5), round(2 * Agents[i[0][0]-1][1]['Other'] - 1, 5)
            A = round(s * (Aself - Bself) + o * (Aothr - Bothr), 5)
            B = round(s * (Bself - Aself) + o * (Bothr - Aothr), 5)
            if A == B:
                A, B = randexcept(0.5), randexcept(0.5)
            choice = [Aself, Aothr] if A > B else [Bself, Bothr]
            reject = [Bself, Bothr] if A > B else [Aself, Aothr]
        pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
        if attachparams:
            actions = {"Chooser" : ['Agent' + str(i[0][0]), i[0][0]], "Predictor" : ['Agent' + str(i[0][1]), i[0][1]], \
                "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, "Points-Chooser" : pts_chooser, \
                    "Points-Predictor" : pts_predictor, "Chooser-Params" : Agents[i[0][0]-1][1], "Predictor-Params" : Agents[i[0][1]-1][1]}    
        else:
            actions = {"Chooser" : ['Agent' + str(i[0][0]), i[0][0]], "Predictor" : ['Agent' + str(i[0][1]), i[0][1]], \
                    "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, \
                        "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
        actionlist.append(actions)  
    return actionlist


def multiround(numrounds = numrounds, nagents = nagents):
    """This iterates the game over n rounds.  It generates new lists of matches and payoffs each round."""
    now, allrounds = 0, []
    while now < numrounds:
        if circular:
            allrounds.append(actionlist(givepaystruct=givepaystruct(chooselist=chooselist(\
                matchlist=randmatch()), paylist=sellist360)))    
        else:
            allrounds.append(actionlist(givepaystruct=givepaystruct(chooselist=chooselist(\
                matchlist=randmatch()), paylist=sellist(nagents=nagents))))
        now = now + 1
    return allrounds


time9 = timeit.default_timer()

def scorekeeper(gamerecord = multiround(numrounds), nagents = nagents, numrounds = numrounds):
    recordict = {'Round' : [], 'Agent' : [], 'Points' : []}
    for roundnum, roundrecord in enumerate(gamerecord):
        for gamerecord in roundrecord:
            recordict['Round'].append(roundnum), recordict['Agent'].append(gamerecord['Chooser'][1]), recordict['Points'].append(gamerecord['Points-Chooser'])
            recordict['Round'].append(roundnum), recordict['Agent'].append(gamerecord['Predictor'][1]), recordict['Points'].append(gamerecord['Points-Predictor'])
    recordframe = pd.DataFrame(recordict)
    recordframe = recordframe.sort_values(by = ['Round', 'Agent'])
    recordframe = recordframe.pivot(index = 'Round', columns = 'Agent', values = 'Points')
    recordframe, recordarray = recordframe.values, []
    for step, scores in enumerate(recordframe):
        recordarray.append([])
        for score in scores:
            recordarray[step].append(int(score))
    finalpts = [0] * len(recordarray[0])
    for steps, vals in enumerate(recordarray):
        for step, val in enumerate(vals): 
            finalpts[step] = finalpts[step] + val
    return recordarray, finalpts

time10 = timeit.default_timer()

def cummat(matrix):
    row, col, rows, cols = 0, 0, len(matrix), len(matrix[0])
    cumtotals = np.zeros(shape=(rows, cols), dtype=object)
    runaverag = np.zeros(shape=(rows, cols), dtype=object)
    while row < rows:
        while col < cols:
            if row == 0:
                cumtotals[row][col] = matrix[row][col]
                runaverag[row][col] = matrix[row][col]
            else:
                cumtotals[row][col] = cumtotals[row-1][col] + matrix[row][col]
                runaverag[row][col] = cumtotals[row][col] / (row + 1)
            col = col + 1
        row, col = row + 1, 0
    runaverag = normroundmat(runaverag, 2)
    return cumtotals, runaverag

time11 = timeit.default_timer()

def xyz(xaxis = 'Self', yaxis = 'Other', zaxis = 'Final Points'):
    recordarray = scorekeeper()
    count, x, y = 0, [], []
    while count < nagents:
        xpoint = Agents[count][1][xaxis]
        ypoint = Agents[count][1][yaxis]
        x.append(xpoint)
        y.append(ypoint)
        count = count + 1
    if zaxis == 'Points':
        z = cummat(scorekeeper(gamerecord = multiround(numrounds))[0])[1]
    elif zaxis == 'Final Points':
        z = normround(recordarray[1],5)
    else:
        z = normround(recordarray[1],5)
    return x, y, z

xyz = xyz(zaxis='Final Points')
x, y, z = xyz[0], xyz[1], xyz[2] 

time12 = timeit.default_timer()

if givetimes:
    print('Time 0 -- 1: ', time1 - time0, (time1 - time0) / (time12 - time0))
    print('Time 1 -- 2: ', time2 - time1, (time2 - time1) / (time12 - time0))
    print('Time 2 -- 3: ', time3 - time2, (time3 - time2) / (time12 - time0))
    print('Time 3 -- 4: ', time4 - time3, (time4 - time3) / (time12 - time0))
    print('Time 4 -- 5: ', time5 - time4, (time5 - time4) / (time12 - time0))
    print('Time 5 -- 6: ', time6 - time5, (time6 - time5) / (time12 - time0))
    print('Time 6 -- 7: ', time7 - time6, (time7 - time6) / (time12 - time0))
    print('Time 7 -- 8: ', time8 - time7, (time8 - time7) / (time12 - time0))
    print('Time 8 -- 9: ', time9 - time8, (time9 - time8) / (time12 - time0))
    print('Time 9 -- 10: ', time10 - time9, (time10 - time9) / (time12 - time0))
    print('Time 10 -- 11: ', time11 - time10, (time11 - time10) / (time12 - time0))
    print('Time 11 -- 12: ', time12 - time11, (time12 - time11) / (time12 - time0))
    print('Time Total: ', time12 - time0)

    print('n Agents: ', nagents, ' n Rounds: ', numrounds, ' Runtime: ', round(time12 - time0, 5))
    exit()

if givetime:
    print('n Agents: ', nagents, ' n Rounds: ', numrounds, ' Runtime: ', round(time12 - time0, 5))

if showsp:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = normround(z,5)
    colors = [x * 100 for x in colors]
    ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
    ax.set_xlabel('Self Regard'), ax.set_ylabel('Other Regard'), ax.set_zlabel('Total Points')
    if normaxes or chronicle:
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)
    if chronicle:
        ax.clear()
        z = xyz()[2][0] 
        colors = [x * 100 for x in z]
        plt.subplots_adjust(bottom=0.25)
        ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
        evo = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightsalmon')
        evolution = Slider(ax = evo, label = "Evolution", valmin = 0, valmax = numrounds - 1, \
            valinit = 0, valstep = 1, orientation= 'horizontal', color = 'coral')
        ax.set_xlabel('Self Regard'), ax.set_ylabel('Other Regard'), ax.set_zlabel('Total Points')
        ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)

        def update(val):
            ax.clear()
            z = xyz()[2][val] 
            colors = [x * 100 for x in z]
            ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
            ax.set_xlabel('Self Regard'), ax.set_ylabel('Other Regard'), ax.set_zlabel('Total Points')
            ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1), fig.canvas.draw()

        evolution.on_changed(update)
    plt.show()







exit()

def actionlist(givepaystruct = givepaystruct(), Agents = Agents, attachparams = False, worldparams = WorldParameters):
    actionlist, pswitch, pcatch = [], worldparams['p(switch)'], worldparams['p(catch)']
    for i in givepaystruct:
        Paylabel, Aself, Aothr, Bself, Bothr = i[1][4], i[1][0], i[1][1], i[1][2], i[1][3]
        switchlist = switcher(Aself, Aothr, Bself, Bothr, pswitch)
        if circular:
            choice = virtuechoice(Agents[i[0][0]-1][1]['VTtheta'], [Aself, Aothr, Bself, Bothr], Angle = False)[0]
            reject = virtuechoice(Agents[i[0][0]-1][1]['VTtheta'], [Aself, Aothr, Bself, Bothr], Angle = False)[1]
        else:
            s, o = round(2 * Agents[i[0][0]-1][1]['Self'] - 1, 5), round(2 * Agents[i[0][0]-1][1]['Other'] - 1, 5)
            A = round(s * (Aself - Bself) + o * (Aothr - Bothr), 5)
            B = round(s * (Bself - Aself) + o * (Bothr - Aothr), 5)
            if A == B:
                A, B = randexcept(0.5), randexcept(0.5)
            choice = [Aself, Aothr] if A > B else [Bself, Bothr]
            reject = [Bself, Bothr] if A > B else [Aself, Aothr]
        pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
        if attachparams:
            actions = {"Chooser" : ['Agent' + str(i[0][0]), i[0][0]], "Predictor" : ['Agent' + str(i[0][1]), i[0][1]], \
                "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, "Points-Chooser" : pts_chooser, \
                    "Points-Predictor" : pts_predictor, "Chooser-Params" : Agents[i[0][0]-1][1], "Predictor-Params" : Agents[i[0][1]-1][1]}    
        else:
            actions = {"Chooser" : ['Agent' + str(i[0][0]), i[0][0]], "Predictor" : ['Agent' + str(i[0][1]), i[0][1]], \
                    "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, \
                        "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
        actionlist.append(actions) 
        print(actionlist)   
    return actionlist




def multiround(numrounds = numrounds, nagents = nagents):
    """This iterates the game over n rounds.  It generates new lists of matches and payoffs each round."""
    now, allrounds = 0, []
    while now < numrounds:
        if circular:
            allrounds.append(actionlist(givepaystruct=givepaystruct(chooselist=chooselist(\
                matchlist=randmatch()), paylist=sellist360)))    
        else:
            allrounds.append(actionlist(givepaystruct=givepaystruct(chooselist=chooselist(\
                matchlist=randmatch()), paylist=sellist(nagents=nagents))))
        now = now + 1
    return allrounds




def virtueplot(mean = 0, stdev = 1.5, angle = 225):
    fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.suptitle('Virtue Threshold', fontdict={'fontname':'Calibri'}),
    ax.set_xlabel('Utility Self'), ax.set_ylabel('Utility Other'), ax.set_zlabel('Probability of Choice')
    X = np.arange(-9, 9, .1)
    Y = np.arange(-9, 9, .1)
    X, Y = np.meshgrid(X, Y)
    quotient = angle // 360
    if quotient != 0:
        angle = angle - 360*quotient
    theta = (angle * math.pi) / 180
    Z = sp.stats.norm.cdf(X * math.cos(theta) - Y * math.sin(theta), mean, stdev)
    ax.set_xlim(-10, 10), ax.set_ylim(-10, 10)
    ax.plot_surface(X, Y, Z, cmap=cm.inferno, linewidth=0, antialiased=False), plt.show()    





def likelihoodplot(stdev = 0.5, steps = 1):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.suptitle('Likelihood: p(Choice Given Virtue Threshold)', fontdict={'fontname':'Calibri'}),
    ax.set_xlabel('Choice Angle 0 - 360'), ax.set_ylabel('VT Angle 0 - 360'), ax.set_zlabel('Probability of Choice')
    Cangle, Vangle = np.arange(0, 360, steps), np.arange(0, 360, steps)
    Cangle, Vangle = np.meshgrid(Cangle, Vangle)
    Prob = sp.stats.norm.cdf(np.cos(Cangle*np.pi/180) * np.cos(Vangle*np.pi/180) - np.sin(Cangle*np.pi/180) * np.sin(Vangle*np.pi/180), 0, stdev)
    ax.set_xlim(0, 360), ax.set_ylim(0, 360), ax.set_zlim(0, 1)
    ax.plot_surface(Cangle, Vangle, Prob, cmap=cm.viridis, linewidth=0, antialiased=False), plt.show()    

# Liplot = likelihoodplot()
# exit()



def prediction(priorwt, likewt, choice, reject, nextpay, \
    showpost = False, showbeta = False, pmean = pmean, pstdev = pstdev, lmean = 0, lstdev = 0.5, \
        update = prior(mean = pmean, stdev = pstdev, show = False)):
    Post = posterior(priorwt, likewt, choice, reject, loaf = True, show = showpost, \
        pmean = pmean, pstdev = pstdev, lmean = lmean, lstdev = lstdev, update = update)
    x2, y2 = nextpay[0] - nextpay[2], nextpay[1] - nextpay[3]
    x1, y1 = choice[0] - reject[0], choice[1] - reject[1]
    if x1 == 0 or y1 == 0:
        ctheta, cangle = 0, 0
    else:
        ctheta = np.tan(x1/y1)**-1
        cangle = ctheta*180/np.pi
        quotient = cangle // 360
        if quotient != 0:
            cangle = cangle - 360*quotient
        ctheta = (cangle * np.pi) / 180

    if x2 == 0 or y2 == 0:
        ntheta, nangle = 0, 0
    else:
        ntheta = np.tan(x2/y2)**-1
        nangle = ntheta*180/np.pi
        if x2 < 0:
            nangle = nangle + 180
        nquotient = nangle // 360
        if nquotient != 0:
            nangle = nangle - 360*nquotient
        ntheta = (nangle * np.pi) / 180

    cutoff = nangle / 360 
    update = Post[int(round(cangle,0))]
    beta = np.mean(Post[int(round(cangle,0))])
    alpha, tweight = 1 - beta, priorwt + likewt
    subprob = sp.stats.beta.cdf(cutoff, alpha * tweight, beta * tweight)
    rando = randexcept(subprob)
    predyes = [nextpay[0], nextpay[1]] if subprob > rando else [nextpay[2], nextpay[3]]
    prednot = [nextpay[2], nextpay[3]] if subprob > rando else [nextpay[0], nextpay[1]]

    if showbeta:
        X, PMF = np.arange(0, 1, 1/360), []
        for i, v in enumerate(X):
            pofp = sp.stats.beta.pdf(v, alpha * tweight, beta * tweight) / 360
            PMF.append(pofp)
        ix, iy = np.linspace(0, cutoff), []
        for i, v in enumerate(ix):
            pofp = sp.stats.beta.pdf(v, alpha * tweight, beta * tweight) / 360
            iy.append(pofp)
        plt.rcParams['axes.linewidth'] = 1.2
        fig, ax = plt.subplots()
        ax.plot(X, PMF, color = 'rebeccapurple', linewidth = 5)
        ax.fill_between(X, PMF, color='mediumpurple', edgecolor='rebeccapurple')
        verts = [(0, 0), *zip(ix, iy), (cutoff, 0)]
        poly = Polygon(verts, facecolor='indigo', edgecolor='rebeccapurple')
        ax.add_patch(poly), plt.show()
    return predyes, prednot, update, Post

pred = prediction(8, 8, [9, 9], [1, 1], [1, 9, 9, 1, 'MG91'], showpost = False, showbeta = True, pmean = 45, pstdev = 50)
# print(pred[0], pred[1])
# print(pred[2])
exit()



r, priorwt, likewt, pmean, pstdev, lmean, lstdev = 0, 2, 2, 100, 50, 0, 0.5
paylist1 = sellist(nagents*2)
while r < numrounds:
    vchoose = virtuechoice(45, [paylist1[r][0], paylist1[r][1], paylist1[r][2], paylist1[r][3]])
    if r <= 0:
        pred = prediction(priorwt=priorwt + r, likewt=0, choice=vchoose[0], reject=vchoose[1], nextpay=paylist1[r], showpost = False, showbeta = False, pmean = pmean, pstdev = pstdev, lmean = 0, lstdev = 0.5)
        update = pred[2]
    elif r > 0 and r < numrounds - 1:
        pred = prediction(priorwt=priorwt + r, likewt=likewt, choice=vchoose[0], reject=vchoose[1], nextpay=paylist1[r], showpost = False, showbeta = False, pmean = pmean, pstdev = pstdev, lmean = 0, lstdev = 0.5, update = update)
        update = pred[2]
    else:
        pred = prediction(priorwt=priorwt + r, likewt=likewt, choice=vchoose[0], reject=vchoose[1], nextpay=paylist1[r], showpost = True, showbeta = False, pmean = pmean, pstdev = pstdev, lmean = 0, lstdev = 0.5, update = update)
        update = pred[2]
    print(vchoose[0], vchoose[1], pred[0], pred[1])
    r = r + 1


exit()


def stimresp(numrounds = numrounds, nagents = nagents, Agents = Agents, givepaystruct=givepaystruct(chooselist=chooselist(matchlist=randmatch()), paylist=sellist360), worldparams = WorldParameters, allinfo = False):
    now, meetings1, meetings2, record = 0, [], [], np.zeros(shape=(numrounds, nagents), dtype=object)
    pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
    while now < numrounds:
        for i in givepaystruct:
            Chooser, Predictor = i[0][0], i[0][1]
            meetings1.append([Chooser, Predictor])
            Paylabel, Aself, Aothr, Bself, Bothr = i[1][4], i[1][0], i[1][1], i[1][2], i[1][3]
            switchlist = switcher(Aself, Aothr, Bself, Bothr, pswitch)
            chooparams, predparams = Agents[i[0][0]][1], Agents[i[0][1]][1]
            choice, reject = agentfunct(i[1], chooparams, predparams, worldparams)
            pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
            actions = {"Round" : now, "Chooser" : ['Agent' + str(Chooser), Chooser], "Predictor" : ['Agent' + str(Predictor), Predictor], \
                "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Switch" : switchlist, "Choice" : choice, "Points-Chooser" : \
                    pts_chooser, "Points-Predictor" : pts_predictor}
            if allinfo:
                actions["Chooser-Params"], actions["Predictor-Params"], actions["World-Params"] = chooparams, predparams, worldparams
            record[now][Chooser], record[now][Predictor] = actions, actions
            print("Round: ", now, " Chooser: ", i[0][0], " Predictor: ", i[0][1], " Pts-Choo: ", pts_chooser, " Pts-Pred: ", pts_predictor)
        meetings2.append(meetings1)
        now = now + 1
    # print(meetings2)
    return record


stimresp = stimresp()
# print(stimresp)
# print(Agents)


def Agents(params1,params2=0,params3=0,params4=0,params5=0,params6=0,params7=0,params8=0,params9=0,params10=0,params11=0,params12=0,params13=0,params14=0):
    x, Agents = 0, []
    while x < nagents:
        Agents.append({'Power' : params1[x], 'Self' : params2[x], 'Other' : params3[x], 'VTtheta' : params4[x], 'Envy' : params5[x], 
            'Guilt' : params6[x], 'EGtheta' : params7[x], 'Lucidity' : params8[x], 'Honesty' : params9[x], 'Forgiving' : params10[x], 
                'Stochastic' : params11[x], 'Prior' : params12[x], 'WtRatio' : params13[x], 'WtTotal' : params14[x]})
        x = x + 1
    Agents = list(enumerate(Agents,0))
    return Agents



# Pays = [[8, 4, 9, 1, 'MG01'], 
# [1, 2, 9, 8, 'MG02'], 
# [6, 9, 4, 1, 'MG03'], 
# [5, 5, 1, 1, 'MG04'], 
# [9, 1, 1, 9, 'MG05'], 
# [1, 9, 9, 1, 'MG06'], 
# [3, 6, 9, 5, 'MG07'], 
# [1, 5, 9, 5, 'MG08'], 
# [8, 2, 2, 8, 'MG09'], 
# [7, 3, 5, 1, 'MG10'], 
# [4, 2, 8, 2, 'MG11'], 
# [1, 2, 9, 8, 'MG12'], 
# [6, 9, 4, 1, 'MG13'], 
# [5, 5, 1, 1, 'MG14'], 
# [9, 1, 1, 9, 'MG15'], 
# [1, 9, 9, 1, 'MG16'], 
# [3, 6, 9, 5, 'MG17'], 
# [1, 5, 9, 5, 'MG18'], 
# [8, 2, 2, 8, 'MG19'], 
# [7, 3, 5, 1, 'MG20']]

# def virtuechoice(VTangle, stdev, Aself, Aothr, Bself, Bothr, Angle = True):
#     S, O = Aself - Bself, Aothr - Bothr
#     if Angle: 
#         quotient = VTangle // 360
#         if quotient != 0:
#             VTangle = VTangle - 360*quotient
#         theta = (VTangle * math.pi) / 180
#     else:
#         theta = VTangle
#     probA = sp.stats.norm.cdf(S * math.cos(theta) - O * math.sin(theta), 0, stdev)  
#     probpolicy = randexcept(probA)  
#     choice = [Aself, Aothr] if probA > probpolicy else [Bself, Bothr]
#     reject = [Bself, Bothr] if probA > probpolicy else [Aself, Aothr]
#     return choice, reject


# fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
# ax0, ax1, ax2, ax3 = axes.flatten()
# fig.suptitle('Payoff Distribution', fontdict={'fontname':'Calibri'}), plt.text(23, 45, r'$\mu=15, b=3$')
# ax0.hist(x=CoordAlign, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
# ax2.hist(x=CoordMagnit, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
# ax2.set_xlabel('Payoff Dimension'), ax1.set_ylabel('Number of Payoff Structures')
# ax1.hist2d(CoordAlign, CoordMagnit, bins=wbins)
# ax3.scatter(CoordAlign, CoordMagnit, color = 'blueviolet', edgecolors='indigo')
# ax3.set_xlabel('Interest Aligned --- Conflicting'), ax1.set_ylabel('Low Stakes --- High Stakes')
# plt.grid(color = 'black', linestyle = '--', linewidth = 1)
# n, bins, patches2 = ax1.hist(x=CoordMagnit, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
# n, bins, patches1 = ax0.hist(x=CoordAlign, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
# fracs = n / MaxCoord

# """This normalizes the data between 0 and 1 for the full range of the colormap.  
# Then it loops through to set the color of each bar according to its height."""
# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches1):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)

# norm = colors.Normalize(fracs.min(), fracs.max())
# for thisfrac, thispatch in zip(fracs, patches2):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)

# """This displays the histogram."""
# maxfreq = n.max()
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10), #plt.show()



fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 8))
ax1, ax2, ax3, ax4 = axes.flatten()
fig.suptitle('Payoff Distribution', fontdict={'fontname':'Calibri'}), plt.text(23, 45, r'$\mu=15, b=3$')

plt.subplot(2,2,1)
ax1 = plt.hist(x=CoordAlign, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
n, bins, patches = plt.hist(x=CoordAlign, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
plt.ylim(0, bigbin + 54)
fracs = n / MaxCoord
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.subplot(2,2,2)
ax2 = plt.hist(x=CoordMagnit, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
n, bins, patches = plt.hist(x=CoordMagnit, bins=wbins, color='forestgreen', alpha=0.7, rwidth=1)
plt.ylim(0, bigbin + 54)
fracs = n / MaxCoord
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.subplot(2,2,3)
plt.hist2d(CoordAlign, CoordMagnit, bins=wbins)
plt.subplot(2,2,4)
plt.scatter(CoordAlign, CoordMagnit, color = 'blueviolet', edgecolors='indigo')
plt.grid(color = 'black', linestyle = '--', linewidth = 1)


plt.show()

def normroundarr(array,rnd=20):
    """This normalizes the values in any array between zero 
    and one.  There is an optional rounding arguement."""
    Min = min(array)
    for i in range(len(array)):
        array[i] = array[i] - Min
    Max = max(array)
    for i in range(len(array)):
        array[i] = array[i] / Max
    for i in range(len(array)):
        array[i] = round(array[i],rnd)
    return array


Asubx, Bsubx, Csubx, Dsubx = 0, 2, 1, 1
Asuby, Bsuby, Csuby, Dsuby = 0, 2, -1, 1
Asubz, Bsubz, Csubz, Dsubz = 0, 2, -1, 1

CoeffMatrix = np.matrix([[Asubx, Bsubx, Csubx, Dsubx], [
                        Asuby, Bsuby, Csuby, Dsuby], [Asubz, Bsubz, Csubz, Dsubz]])

def jPDF(x, y=None, z=None, equation=None):
    if y is None and z is None:
        jPDF = (Asubx*x**3 + Bsubx*x**2 + Csubx*x + Dsubx)
    elif y is None and z is not None:
        xPDF = (Asubx*x**3 + Bsubx*x**2 + Csubx*x + Dsubx)
        zPDF = (Asubz*z**3 + Bsubz*z**2 + Csubz*z + Dsubz)
        jPDF = xPDF * zPDF
    elif y is not None and z is None:
        xPDF = (Asubx*x**3 + Bsubx*x**2 + Csubx*x + Dsubx)
        yPDF = (Asuby*y**3 + Bsuby*y**2 + Csuby*y + Dsuby)
        jPDF = xPDF * yPDF
    else:
        xPDF = (Asubx*x**3 + Bsubx*x**2 + Csubx*x + Dsubx)
        yPDF = (Asuby*y**3 + Bsuby*y**2 + Csuby*y + Dsuby)
        zPDF = (Asubz*z**3 + Bsubz*z**2 + Csubz*z + Dsubz)
        jPDF = xPDF * yPDF * zPDF
    if equation is not None:
        try:
            jPDF = eval(equation)
        except:
            pass
    return jPDF



# x = np.outer(np.linspace(-2, 2, 30), np.ones(30))
# y = x.copy().T
# z = np.cos(x ** 2 + y ** 2)

# fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
  
#fig.show()



Deg360, priorwt, likewt = 6, 3, 10
Prior = normround(prior(mean = 225, stdev = 50, show = False))
Like = likelihood(choice = [1, 1], reject = [9, 9], loaf = True, show = False)
xax, yax = np.zeros(shape=(Deg360, Deg360), dtype=object), np.zeros(shape=(Deg360, Deg360), dtype=object)
# X, Y = np.arange(0, Deg360, 1), np.arange(0, Deg360, 1)
# X, Y = np.meshgrid(X, Y)

X = np.arange(0, Deg360, 1)
Y = X.copy().T

for i, v in enumerate(xax):
    for j, w in enumerate(v):
        xax[i][j] = round(Prior[j] * priorwt, 5)
        yax[i][j] = round(Like[i][j] * likewt, 5)
zax = (xax + yax) / (priorwt + likewt)

#print(zax)
# for i in zax:
#     for j in i:
#         j = float(j)
# if np.any(np.isnan(zax)):
# zax = pd.Series(zax)
# zax.fillna(method='ffill')
# zax = [x for x in zax if str(x) != 'nan']
# if np.any(np.isnan(zax)):
#     print("Found")
# else:
#     print("Not Found")


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# plt.suptitle('Posterior Over VTs', fontdict={'fontname':'Calibri'}),
# ax.set_xlabel('Choice Angle 0 - 360'), ax.set_ylabel('VT Angle 0 - 360'), ax.set_zlabel('Probability Density')
# ax.set_xlim(0, Deg360), ax.set_ylim(0, Deg360), ax.set_zlim(0, 1)
# ax.plot_surface(X, Y, zax, cmap=cm.viridis, linewidth=0, antialiased=False), plt.show()


exit()


# fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})
# plt.suptitle('Virtue Threshold', fontdict={'fontname':'Calibri'}),
# ax.set_xlabel('Utility Self'), ax.set_ylabel('Utility Other'), ax.set_zlabel('Probability of Choice')
# X = np.arange(0, 1, .1)
# Y = np.arange(0, 1, .1)
# xx = np.arange(1, 2, .1)
# yy = np.arange(1, 2, .1)
# X, Y = np.meshgrid(X, Y)
# xx, yy = np.meshgrid(xx, yy)
# Z = 2*xx**2 + 3*yy**2
# print(Z)
# ax.set_xlim(0, 1), ax.set_ylim(0, 1)
# ax.plot_surface(X, Y, Z, cmap=cm.inferno, linewidth=0, antialiased=False), plt.show()    






#print(scorekeeper()[1])

    # def startmatrices(Agents=Agents, nagents=nagents):
    #     round0 = np.zeros(shape=(nagents+1, nagents+1), dtype=object)
    #     Adjacency = np.zeros(shape=(nagents+1, nagents+1), dtype=object)
    #     roundn = np.zeros(shape=(nagents+1, nagents+1), dtype=object)
    #     clicks = 0
    #     while clicks < nagents:
    #         round0[0][clicks + 1] = Agents[clicks]
    #         round0[clicks + 1][0] = Agents[clicks]
    #         Adjacency[0][clicks + 1] = clicks + 1
    #         Adjacency[clicks + 1][0] = clicks + 1
    #         clicks = clicks + 1
    #     return round0, Adjacency, roundn

    # round0 = startmatrices(Agents)[0]
    # roundn = startmatrices(Agents)[2]
    # paylist = sellist(nagents*2 + 2)

    # def matchrand(nagents=nagents, duplicates=False):
    #     """Part 1: This produces a matrix with row and column headers labeled by sequential integers."""
    #     adj, clicks = np.zeros(shape=(nagents+1, nagents+1), dtype=object), 0
    #     while clicks < nagents:
    #         adj[0][clicks + 1] = clicks + 1
    #         adj[clicks + 1][0] = clicks + 1
    #         clicks = clicks + 1
    #     if duplicates:
    #         """If agents can play twice each round..."""
    #         if nagents % 2 > 0:
    #             numlist, matchlist = list(range(0, nagents + 1)), []
    #             """If there is an odd number of agents, then pair the leftover person with zero."""
    #         else:
    #             numlist, matchlist = list(range(1, nagents + 1)), []
    #         random.shuffle(numlist)
    #         """Part 2: This generates a random sequence of integers where no item equals its index."""
    #         count, maxloop = 0, nagents * 10
    #         while count < len(numlist) * 2:
    #             for x in numlist:
    #                 index = numlist.index(x)
    #                 if index + 1 == x:
    #                     pos1, pos2, count = x - 1, random.choice(numlist) - 1, 0
    #                     numlist[pos1], numlist[pos2] = numlist[pos2], numlist[pos1]
    #                 else:
    #                     count = count + 1
    #         """Part 3: This creates an adjacency matrix with '1's representing a meeting between players."""
    #         row, col = 1, 1
    #         while row < nagents + 1:
    #             while col < nagents + 1:
    #                 if col == numlist[row-1]:
    #                     adj[row][col] = "1"
    #                 col = col + 1
    #             row, col = row + 1, 1
    #     else:
    #         """If agents can play once each round..."""
    #         if nagents % 2 > 0:
    #             numlist, matchlist = list(range(1, nagents)), []
    #             """If there is an odd number of agents, remove the leftover person from the list."""
    #         else:
    #             numlist, matchlist = list(range(1, nagents + 1)), []
    #         random.shuffle(numlist)
    #         """Part 2: This generates a list of random pairs by sampling a sequential list without replacement."""
    #         while len(numlist) > 0:
    #             first = numlist.pop(0)
    #             secon = numlist.pop()
    #             pair = [first, secon]
    #             matchlist.append(pair)
    #         """Part 3: This converts the list of pairs into a one-sided adjacency matrix."""
    #         for x in matchlist:
    #             if x[0] > x[1]:
    #                 adj[x[0]][x[1]] = "1"
    #             else:
    #                 adj[x[1]][x[0]] = "1"
    #     return adj


    # def actionS(matches, freshmatrix = startmatrices(Agents)[2]):
    #     """This iterates the function below for all matches in the matrix."""
    #     adjmat = matches
    #     def action(col_player, row_player, payoffs):
    #         """This function takes the players and payoffs and returns their choices, predictions, bets, and points."""
    #         rando, pratio = randexcept(0.5), col_player[1]['Power'] / (col_player[1]['Power'] + row_player[1]['Power'])
    #         if pratio > rando:
    #             chooser, predictor = col_player, row_player
    #             #print("Col Choice: ", col_player[1]['Power'], row_player[1]['Power'], round(pratio,5), " > ", round(rando,5))
    #         else:
    #             chooser, predictor = row_player, col_player
    #             #print("Row Choice: ", col_player[1]['Power'], row_player[1]['Power'], round(pratio,5), " < ", round(rando,5))
    #         Paylabel, Aself, Aothr, Bself, Bothr = paylist[col][4], \
    #             paylist[col][0], paylist[col][1], paylist[col][2], paylist[col][3]
    #         s, o = 2 * (chooser[1]['Self']) - 1, 2 * (chooser[1]['Other']) - 1
    #         A = s * (Aself - Bself) + o * (Aothr - Bothr)
    #         B = s * (Bself - Aself) + o * (Bothr - Aothr)
    #         if A == B:
    #             A, B = randexcept(0.5), randexcept(0.5)
    #         choice = [Aself, Aothr] if A > B else [Bself, Bothr]
    #         reject = [Bself, Bothr] if A > B else [Aself, Aothr]
    #         if chooser == col_player:
    #             pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
    #             actions = {"Chooser" : ['Agent' + str(col), col], "Predictor" : ['Agent' + str(row), row], \
    #                 "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Choice" : choice, \
    #                     "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
    #             #print(actions)
    #         else:
    #             pts_chooser, pts_predictor = choice[1] - reject[1], choice[0] - reject[0]
    #             actions = {"Chooser" : ['Agent' + str(row), row], "Predictor" : ['Agent' + str(col), col], \
    #                 "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Choice" : choice, \
    #                     "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
    #             #print(actions)
    #         return actions
    #     """Here begins the iteration of the above funciton."""
    #     row, col = 0, 0 
    #     roundn = freshmatrix
    #     while row < nagents + 1:
    #         while col < nagents + 1:
    #             if adjmat[row][col] == "1":
    #                 roundn[row][col] = action(round0[0][col], round0[row][0], paylist[col])
    #             col = col + 1
    #         col, row = 0, row + 1
    #     return roundn


    # def nrounds(nrounds=numrounds,nagents=nagents):
    #     """This iterates the game over n rounds.  It provides actionS() with a new adjacency matrix each time."""
    #     nround, allrounds = 0, []
    #     while nround < nrounds:
    #         Adj, fresh = matchrand(duplicates=True), startmatrices(Agents)[2]
    #         allrounds.append(actionS(Adj, fresh))
    #         nround = nround + 1
    #     return allrounds

    # rawrecord = nrounds(numrounds)

    # def results(nagents = nagents, numrounds = numrounds, rawrecord = rawrecord, normalize = True, show = False):
    #     ptsmat, row, col = np.zeros(shape=(numrounds+1, nagents+1), dtype=object), 0, 0
    #     while row < numrounds + 1:
    #         ptsmat[row][0] = row
    #         row = row + 1
    #     while col < nagents + 1:
    #         ptsmat[0][col] = col
    #         col = col + 1
    #     row, col, rounds = 1, 1, 0
    #     while rounds < numrounds:
    #         while row < nagents + 1:
    #             while col < nagents + 1:
    #                 item = rawrecord[rounds][row][col]
    #                 if item != 0:
    #                     chooser = int(item['Chooser'][1])
    #                     #print("Round: ", rounds, chooser, row, item)
    #                     if chooser == row:
    #                         ptsmat[rounds+1][row] = item['Points-Chooser']
    #                         #print("Round: ", rounds, chooser, row, item)
    #                         #print("Yes")
    #                     else:
    #                         ptsmat[rounds+1][row] = item['Points-Predictor']
    #                         #print("Round: ", rounds, chooser, row, item)
    #                         #print("No")
    #                     #ptsmat[rounds+1][row] = item['Points-Chooser']
    #                     if show:
    #                         print("Row: ", row, " Col: ", col," Round: ", rounds+1, " Payoffs: ", item['Payoffs'], " Choice: ", \
    #                             item['Choice'], " Chooser Agent: ", item['Chooser'][1], " Predictor Agent: ", item['Predictor'][1], \
    #                                 " C-Pts: ", item['Points-Chooser'], " P-Pts: ", item['Points-Predictor'])
    #                 col = col + 1
    #             col, row = 1, row + 1
    #         col, row, rounds = 1, 1, rounds + 1
    #     if normalize:
    #         finalpts = normround(sum(np.delete(np.delete(ptsmat,0,0),0,1)), 5)
    #         results = normroundmat(np.delete(np.delete(ptsmat,0,0),0,1), 5)
    #     finalpts = sum(np.delete(np.delete(ptsmat,0,0),0,1))
    #     results = np.delete(np.delete(ptsmat,0,0),0,1)
    #     return ptsmat, results, finalpts

    # #print(results()[0])

# recordframe = scorekeeper(gamerecord = multiround(numrounds))
# finalpoints = sum(recordframe)


# def cummat(matrix):
#     row, col, rows, cols = 0, 0, len(matrix), len(matrix[0])
#     cumtotals = np.zeros(shape=(rows, cols), dtype=object)
#     runaverag = np.zeros(shape=(rows, cols), dtype=object)
#     while row < rows:
#         while col < cols:
#             if row == 0:
#                 cumtotals[row][col] = matrix[row][col]
#                 runaverag[row][col] = matrix[row][col]
#             else:
#                 cumtotals[row][col] = cumtotals[row-1][col] + matrix[row][col]
#                 runaverag[row][col] = cumtotals[row][col] / (row + 1)
#             col = col + 1
#         row, col = row + 1, 0
#     runaverag = normroundmat(runaverag, 2)
#     return cumtotals, runaverag


# print(cummat(scorekeeper(gamerecord = multiround(numrounds)))[1])
# print(sum(scorekeeper(gamerecord = multiround(numrounds))))

# def xyz(xaxis = 'Self', yaxis = 'Other', zaxis = 'Points'):
#     count, x, y = 0, [], []
#     while count < nagents:
#         xpoint = Agents[count][1][xaxis]
#         ypoint = Agents[count][1][yaxis]
#         x.append(xpoint)
#         y.append(ypoint)
#         count = count + 1
#     if zaxis == 'Points':
#         #z = cummat(results()[1])[1]
#         #z = cummat(scorekeeper()[0])[1]
#         z = cummat(scorekeeper(gamerecord = multiround(numrounds)))[1]
#     elif zaxis == 'Final Points':
#         #z = results()[2]
#         #z = scorekeeper()[1]
#         z = sum(scorekeeper(gamerecord = multiround(numrounds)))
#     else:
#         #z = results()[2]
#         #z = scorekeeper()[1]
#         z = sum(scorekeeper(gamerecord = multiround(numrounds)))
#     return x, y, z

# x, y, z = xyz()[0], xyz()[1], xyz(zaxis='Final Points')[2]

# if showsp:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     colors = normround(z,5)
#     colors = [x * 100 for x in colors]
#     ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
#     ax.set_xlabel('Self Regard'), ax.set_ylabel('Other Regard'), ax.set_zlabel('Total Points')
#     if normaxes or chronicle:
#         ax.set_xlim3d(0, 1)
#         ax.set_ylim3d(0, 1)
#         ax.set_zlim3d(0, 1)
#     if chronicle:
#         ax.clear()
#         z = xyz()[2][0] 
#         colors = [x * 100 for x in z]
#         plt.subplots_adjust(bottom=0.25)
#         ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
#         evo = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightsalmon')
#         evolution = Slider(ax = evo, label = "Evolution", valmin = 0, valmax = numrounds - 1, \
#             valinit = 0, valstep = 1, orientation= 'horizontal', color = 'coral')
#         ax.set_xlabel('Self Regard'), ax.set_ylabel('Other Regard'), ax.set_zlabel('Total Points')
#         ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1)

#         def update(val):
#             ax.clear()
#             z = xyz()[2][val] 
#             colors = [x * 100 for x in z]
#             ax.scatter(x, y, z, c=colors, marker='o', s=56, cmap='plasma')
#             ax.set_xlabel('Self Regard'), ax.set_ylabel('Other Regard'), ax.set_zlabel('Total Points')
#             ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(0, 1), fig.canvas.draw()

#         evolution.on_changed(update)
#     plt.show()


# exit()

# def scorekeeper(gamerecord = multiround(numrounds), nagents = nagents, numrounds = numrounds):
#     """This is the least efficient function.  Work on it!"""
#     recordmatrix = []
#     for roundnum, roundrecord in enumerate(gamerecord):
#         for gamerecord in roundrecord:
#             recordmatrix.append([roundnum, gamerecord['Chooser'][1], gamerecord['Predictor'][1], gamerecord['Points-Chooser'], gamerecord['Points-Predictor']])
#     pointsmatrix = np.zeros(shape=(numrounds, nagents), dtype=object)
#     rndstep, agntstep = 0, 0
#     while rndstep < numrounds:
#         while agntstep < nagents:
#             for r in recordmatrix:
#                 if r[0] == rndstep and r[1] == agntstep+1:
#                     pointsmatrix[rndstep][agntstep] = r[3]
#                 if r[0] == rndstep and r[2] == agntstep+1:
#                     pointsmatrix[rndstep][agntstep] = r[4]
#             agntstep = agntstep + 1
#         agntstep, rndstep = 0, rndstep + 1
#     finalpoints = sum(pointsmatrix)
#     return pointsmatrix, finalpoints

# score = scorekeeper()[0]
#print(scorekeeper()[0])

# exit()

# ptsmat, row, col = np.zeros(shape=(numrounds+1, nagents+1), dtype=object), 0, 0
# while row < numrounds + 1:
#     ptsmat[row][0] = row
#     row = row + 1
# while col < nagents + 1:
#     ptsmat[0][col] = col
#     col = col + 1
# row, col = 1, 1
# while row < numrounds + 1:
#     while col < nagents + 1:
#         for i in rawrecord[row-1]:
#             for j in i:
#                 if j != 0:
#                     #print(row, col, j)
#                     if ptsmat[0][col] == j['Chooser'][1]:
#                         ptsmat[row][col] = j['Points-Chooser']
#                         print("Round: ", row, " Chooser: ", ptsmat[0][col], j)
#                     elif ptsmat[0][col] == j['Predictor'][1]:
#                         ptsmat[row][col] = j['Points-Predictor']
#                         print("Round: ", row, " Predictor: ", ptsmat[0][col], j)

#         col = col + 1
#     col, row = 1, row + 1

# print(ptsmat)
# exit()
# def results(nagents = nagents, numrounds = numrounds, rawrecord = rawrecord, normalize = True, show = False):
#     ptsmat, row, col = np.zeros(shape=(numrounds+1, nagents+1), dtype=object), 0, 0
#     while row < numrounds + 1:
#         ptsmat[row][0] = row
#         row = row + 1
#     while col < nagents + 1:
#         ptsmat[0][col] = col
#         col = col + 1
#     row, col = 1, 1
#     while row < numrounds + 1:
#         while col < nagents + 1:
#             for i in rawrecord[row-1]:
#                 for j in i:
#                     if j != 0:
#                         #print(row, col, j)
#                         if ptsmat[0][col] == j['Chooser'][1]:
#                             ptsmat[row][col] = j['Points-Chooser']
#                             #print("Round: ", row, " Chooser: ", ptsmat[0][col], j)
#                         elif ptsmat[0][col] == j['Predictor'][1]:
#                             ptsmat[row][col] = j['Points-Predictor']
#                             #print("Round: ", row, " Predictor: ", ptsmat[0][col], j)
#             col = col + 1
#         col, row = 1, row + 1
#     if normalize:
#         finalpts = normround(sum(np.delete(np.delete(ptsmat,0,0),0,1)), 5)
#         results = normroundmat(np.delete(np.delete(ptsmat,0,0),0,1), 5)
#     finalpts = sum(np.delete(np.delete(ptsmat,0,0),0,1))
#     results = np.delete(np.delete(ptsmat,0,0),0,1)
#     #print(ptsmat)
#     return ptsmat, results, finalpts

# print(results()[0])

# exit()


# def matchrand(nagents=nagents):
#     """Part 1: This generates a random sequence of integers where no item equals its index."""
#     count, maxloop, numlist = 0, nagents * 10, list(range(1, nagents + 1))
#     random.shuffle(numlist)
#     while count < len(numlist) * 2:
#         for x in numlist:
#             index = numlist.index(x)
#             if index + 1 == x:
#                 pos1, pos2, count = x - 1, random.choice(numlist) - 1, 0
#                 numlist[pos1], numlist[pos2] = numlist[pos2], numlist[pos1]
#             else:
#                 count = count + 1
#     """Part 2: This produces a matrix with row and column headers labeled by sequential integers."""
#     adj, clicks = np.zeros(shape=(nagents+1, nagents+1), dtype=object), 0
#     while clicks < nagents:
#         adj[0][clicks + 1] = clicks + 1
#         adj[clicks + 1][0] = clicks + 1
#         clicks = clicks + 1
#     """Part 3: This creates an adjacency matrix with '1's representing a meeting between players."""
#     row, col = 1, 1
#     while row < nagents + 1:
#         while col < nagents + 1:
#             if col == numlist[row-1]:
#                 adj[row][col] = "1"
#             col = col + 1
#         row, col = row + 1, 1
#     return adj

# #print(matchrand())


def multiplyList(myList, multiple) :
    for x in myList:
        print(x, multiple)
        x = x * multiple 
    return myList 

def ptsbyround(rnd, ptsmatrix = ptsmat):
    pmat = np.delete(np.delete(ptsmatrix,0,0),0,1)
    pts = pmat[rnd]
    return pts


# m = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(m)
# print(m.sum(axis=0))

# a = np.array([4,6,2,8,6,3,8,0])
# print(list(range(1,len(a)+1)))
# b = list(enumerate(a,1))
# print(b[2][0])

# v = random.randrange(0,2)
# w = random.random()
# print(w)

# r = [5,9,3,2,6,8,0,1]
# print(sorted(r))

# print(len(round0))
# print(round0[0][1])
# mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(mat[1][2])
# print(Agents[1])


def rand1(list1):
    r = random.choice(list1)
    return r
print(rand1([1,2,3,4,5,6]))

# Rlist = list(range(1,10))
# print(Rlist)
# print(random.sample(Rlist, 9))


# def matchrand(adjacency_matrix):
#     """Provide this with a matrix of all zeros, except labled rows and columns.
#     It will produce a random adjacency matrix."""
#     rmeet = random.sample(range(1,len(adjacency_matrix)), len(adjacency_matrix)-1)
#     mcoords = coords(len(adjacency_matrix),len(adjacency_matrix))
#     for x in mcoords:
#         for y in x:
#             if y[0] > 0 and y[1] > 0:
#                 adjacency_matrix[y[0]][rmeet[y[0]-1]] = "1"
#     return adjacency_matrix

# Adjacency = matchrand(Adjacency)

# def action(col_player=round0[0][col], row_player=round0[row][0], payoffs=paylist[col]):
# def action(col_player, row_player, payoffs):
#     """This function takes the players and payoffs and returns their choices, predictions, bets, and points."""
#     rando, pratio = randexcept(0.5), col_player[1]['Power'] / (col_player[1]['Power'] + row_player[1]['Power'])
#     if pratio > rando:
#         chooser, predictor = col_player, row_player
#     else:
#         chooser, predictor = row_player, col_player
#     Paylabel, Aself, Aothr, Bself, Bothr = paylist[col][4], \
#         paylist[col][0], paylist[col][1], paylist[col][2], paylist[col][3]
#     s, o = chooser[1]['Self'], chooser[1]['Other']
#     A = s * (Aself - Bself) + o * (Aothr - Bothr)
#     B = s * (Bself - Aself) + o * (Bothr - Aothr)
#     if A == B:
#         A, B = randexcept(0.5), randexcept(0.5)
#     choice = [Aself, Aothr] if A > B else [Bself, Bothr]
#     reject = [Bself, Bothr] if A > B else [Aself, Aothr]
#     pts_chooser, pts_predictor = choice[0] - reject[0], choice[1] - reject[1]
#     actions = {"Chooser" : ['Agent' + str(col), col], "Predictor" : ['Agent' + str(row), row], \
#         "Payoff ID" : Paylabel, "Payoffs" : [Aself, Aothr, Bself, Bothr], "Choice" : choice, \
#             "Points-Chooser" : pts_chooser, "Points-Predictor" : pts_predictor}
#     return actions

# row, col = 0, 0 
# while row < len(Adjacency):
#     while col < len(Adjacency):
#         if Adjacency[row][col] == "1":
#             roundn[row][col] = action(round0[0][col], round0[row][0], paylist[col])
#             #print(row,col)
#         col = col + 1
#     col, row = 0, row + 1

# print(roundn)

# def actionS(size=nagents+1):
#     row, col = 0, 0 
#     while row < size:
#         while col < size:
#             if Adjacency[row][col] == "1":
#                 roundn[row][col] = action(round0[0][col], round0[row][0], paylist[col])
#             col = col + 1
#         col, row = 0, row + 1    
#     return roundn

def jaynord(equation, nbins):
    count, delimiters = 0, []
    A, B, C = equation[0], equation[1], equation[2]
    t = (A*1**3)/3 + (B*1**2)/2 + C
    while count < nbins:
        section = (count*t)/nbins
        a, b, c, d = 2*A*nbins, 3*B*nbins, 6*C*nbins, -6*t*count
        p, r = -b/(3*a), c/(3*a)
        q = p**3+((b*c-3*a*d)/(6*a**2))
        x = (q + (q**2 + (r - p**2)**3)**(1/2))**(1/3) + \
            (q - (q**2 + (r - p**2)**3)**(1/2))**(1/3) + p
        np.asarray(delimiters.append(
            round(x.real, len(str(abs(int(round(nbins, 0)))))+2)))
        print("Count: ", count, " t: ", round(t, 4), " Section: ", round(
            section, 4), " Real x: ", round(x.real, 4), " x: ", x)
        count = count + 1
    return delimiters

# print(jaynord([1,1,0],30))



def inter(equation, c1=0, c2=1, slices=1000):
    if isinstance(equation, list):
        A, B, C, D = equation[0], equation[1], equation[2], equation[3]
        Apoly = A*c2**4/4 - A*c1**4/4
        Bpoly = B*c2**3/3 - B*c1**3/3
        Cpoly = C*c2**2/2 - C*c1**2/2
        Dpoly = D*c2**1/1 - D*c1**1/1
        t = A*1**4/4 + B*1**3/3 + C*1**2/2 + D
        area = round((Apoly + Bpoly + Cpoly + Dpoly)/t, 5)
    if isinstance(equation, str):
        segs, area, t = np.arange(0, 1, 1/slices), [], []
        for x in segs:
            t.append(eval(equation)*(1/slices))
        t = round(sum(t), 9)
        for x in segs:
            if x >= c1 and x < c2:
                area.append((eval(equation)*(1/slices)))
        area = round(sum(area)/t, 5)
    return area

# print("R: ",inter("x**2+5",0.3,0.7,5000))
# print("P: ",inter([0,1,0,5],0.3,0.7,5000))

# print("R: ",inter("x**3+8",0.2,0.8,5000))
# print("P: ",inter([1,0,0,8],0.2,0.8,5000))

# print("R: ",inter("x**3 + x**2 + x**1 + 1",0.2,0.8,5000))
# print("P: ",inter([1,1,1,1],0.2,0.8,5000))

# print("R: ",inter("2*x**3 + 2*x**2 + 2*x**1 + 2",0.3,0.7,5000))
# print("P: ",inter([2,2,2,2],0.3,0.7,5000))


exit()
a1, a2, b1, b2 = 5, 1, 3, 5
Magnit = 'math.sqrt(abs(a1 - b1) + abs(a2 - b2))'
#Magnit = 'abs(a1 - b1) + abs(a2 - b2)'

print(eval(Magnit))


def equationroots(a, b, c):

    # calculating discriminant using formula
    dis = b * b - 4 * a * c
    sqrt_val = math.sqrt(abs(dis))

    # checking condition for discriminant
    if dis > 0:
        print(" real and different roots ")
        print((-b + sqrt_val)/(2 * a))
        print((-b - sqrt_val)/(2 * a))

    elif dis == 0:
        print(" real and same roots")
        print(-b / (2 * a))

    # when discriminant is less than 0
    else:
        print("Complex Roots")
        print(- b / (2 * a), " + i", sqrt_val)
        print(- b / (2 * a), " - i", sqrt_val)


print(equationroots(4, 4, 4))


Operator = ['x', 'y', 'z']
#Relations = [' + ',' - ', ' * ', ' / ', ' > ', ' < ', ' = ', ' <= ', ' >= ']
Relations = [' > ', ' < ', ' <= ', ' >= ']
x, y, z = 2, 4, 8

tickx, ticky, tickz = 0, 0, 0
Numx, Numy, Numz = 3, 3, 3
Bins = numpy.zeros(shape=(Numx, Numy, Numz), dtype=object)
while tickx < len(Operator):
    while ticky < len(Relations):
        while tickz < len(Operator):
            if eval(Operator[tickx]+Relations[ticky]+Operator[tickz]):
                print(Operator[tickx]+Relations[ticky] +
                      Operator[tickz]+" is true.")
            else:
                print(Operator[tickx]+Relations[ticky] +
                      Operator[tickz]+" is false.")
            #print(Operator[tickx]+Relations[ticky]+Operator[tickz], eval(Operator[tickx]+Relations[ticky]+Operator[tickz]))
            tickz = tickz + 1
        tickz = 0
        ticky = ticky + 1
    ticky = 0
    tickx = tickx + 1
# print(Bins)

exit()
tickx, ticky, tickz = 0, 0, 0
Numx, Numy, Numz = 3, 3, 3
Bins = numpy.zeros(shape=(Numx, Numy, Numz), dtype=object)
while tickx < Numx:
    while ticky < Numy:
        while tickz < Numz:
            print("Tickx: ", tickx, " Ticky: ", ticky, " Tickz: ", tickz)
            tickz = tickz + 1
        tickz = 0
        ticky = ticky + 1
    ticky = 0
    tickx = tickx + 1
# print(Bins)

# MMatrix = [[0, 9, 9, 1, 1],
# [0, 9, 1, 1, 9],
# [0, 3, 3, 1, 1],
# [0, 3, 1, 1, 3],
# [0, 5, 5, 1, 1],
# [0, 5, 1, 1, 5]]


# MMAlign = dimensionalizer(MMatrix,Align)
# MMagnit = dimensionalizer(MMatrix,Magnit)
# MMAlign = normround(MMAlign,5)
# MMagnit = normround(MMagnit,5)

# print(MMAlign)
# print(MMagnit)

print(mindist([3, 3, 4, 7, 9, 9.5], 5))
print(mindist([3, 3, 4, 7, 9, 9.5]))

print("Distances xyz: ", Distx, Disty)
print("Numbins xyz: ", NumBinsx, NumBinsy)

# arr = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
# print(arr)
# newarr = arr.reshape(3,4)
# print(newarr)
# newmat = newarr.reshape(-1)
# print(newmat)


# print(cumnorm(RawProbBins.reshape(-1)).reshape(NumBinsx,NumBinsy))


rando = random.random()
print("Rando: ", rando)
binsel = (next(x[0] for x in enumerate(
    cumnorm(RawProbBins.reshape(-1))) if x[1] > rando))
print("Binsel: ", binsel)
print("Row: ", binsel // NumBinsx)
print("Col: ", binsel % NumBinsx)

# binsel = (next(x[0] for x in enumerate(cumnorm(RawProbBins.reshape(-1))) if x[1] > random.random()))
# rowsel, colsel = binsel // NumBinsx, binsel % NumBinsx
# if BinSizes[rowsel][colsel] == 0:
#     binsel = newbin(BinSizes,rowsel,colsel)
#     rowsel, colsel = binsel[0], binsel[1]
#     print('Selection: ', random.choice(list(AllBins[rowsel][colsel])))
# else:
#     print('Selection: ', random.choice(list(AllBins[rowsel][colsel])))

A1, A2, A3, A4 = len(subset(0.0, 0.125, 0.00, 0.125)), len(subset(0.125, 0.25, 0.00, 0.125)), len(
    subset(0.25, 0.375, 0.00, 0.125)), len(subset(0.375, 0.5, 0.00, 0.125))
A5, A6, A7, A8 = len(subset(0.5, 0.625, 0.00, 0.125)), len(subset(0.625, 0.75, 0.00, 0.125)), len(
    subset(0.75, 0.875, 0.00, 0.125)), len(subset(0.875, 1.0, 0.00, 0.125))
B1, B2, B3, B4 = len(subset(0.0, 0.125, 0.125, 0.25)), len(subset(0.125, 0.25, 0.125, 0.25)), len(
    subset(0.25, 0.375, 0.125, 0.25)), len(subset(0.375, 0.5, 0.125, 0.25))
B5, B6, B7, B8 = len(subset(0.5, 0.625, 0.125, 0.25)), len(subset(0.625, 0.75, 0.125, 0.25)), len(
    subset(0.75, 0.875, 0.125, 0.25)), len(subset(0.875, 1.0, 0.125, 0.25))
C1, C2, C3, C4 = len(subset(0.0, 0.125, 0.25, 0.375)), len(subset(0.125, 0.25, 0.25, 0.375)), len(
    subset(0.25, 0.375, 0.25, 0.375)), len(subset(0.375, 0.5, 0.25, 0.375))
C5, C6, C7, C8 = len(subset(0.5, 0.625, 0.25, 0.375)), len(subset(0.625, 0.75, 0.25, 0.375)), len(
    subset(0.75, 0.875, 0.25, 0.375)), len(subset(0.875, 1.0, 0.25, 0.375))
D1, D2, D3, D4 = len(subset(0.0, 0.125, 0.375, 0.50)), len(subset(0.125, 0.25, 0.375, 0.50)), len(
    subset(0.25, 0.375, 0.375, 0.50)), len(subset(0.375, 0.5, 0.375, 0.50))
D5, D6, D7, D8 = len(subset(0.5, 0.625, 0.375, 0.50)), len(subset(0.625, 0.75, 0.375, 0.50)), len(
    subset(0.75, 0.875, 0.375, 0.50)), len(subset(0.875, 1.0, 0.375, 0.50))
E1, E2, E3, E4 = len(subset(0.0, 0.125, 0.50, 0.625)), len(subset(0.125, 0.25, 0.50, 0.625)), len(
    subset(0.25, 0.375, 0.50, 0.625)), len(subset(0.375, 0.5, 0.50, 0.625))
E5, E6, E7, E8 = len(subset(0.5, 0.625, 0.50, 0.625)), len(subset(0.625, 0.75, 0.50, 0.625)), len(
    subset(0.75, 0.875, 0.50, 0.625)), len(subset(0.875, 1.0, 0.50, 0.625))
F1, F2, F3, F4 = len(subset(0.0, 0.125, 0.625, 0.75)), len(subset(0.125, 0.25, 0.625, 0.75)), len(
    subset(0.25, 0.375, 0.625, 0.75)), len(subset(0.375, 0.5, 0.625, 0.75))
F5, F6, F7, F8 = len(subset(0.5, 0.625, 0.625, 0.75)), len(subset(0.625, 0.75, 0.625, 0.75)), len(
    subset(0.75, 0.875, 0.625, 0.75)), len(subset(0.875, 1.0, 0.625, 0.75))
G1, G2, G3, G4 = len(subset(0.0, 0.125, 0.75, 0.875)), len(subset(0.125, 0.25, 0.75, 0.875)), len(
    subset(0.25, 0.375, 0.75, 0.875)), len(subset(0.375, 0.5, 0.75, 0.875))
G5, G6, G7, G8 = len(subset(0.5, 0.625, 0.75, 0.875)), len(subset(0.625, 0.75, 0.75, 0.875)), len(
    subset(0.75, 0.875, 0.75, 0.875)), len(subset(0.875, 1.0, 0.75, 0.875))
H1, H2, H3, H4 = len(subset(0.0, 0.125, 0.875, 1.00)), len(subset(0.125, 0.25, 0.875, 1.00)), len(
    subset(0.25, 0.375, 0.875, 1.00)), len(subset(0.375, 0.5, 0.875, 1.00))
H5, H6, H7, H8 = len(subset(0.5, 0.625, 0.875, 1.00)), len(subset(0.625, 0.75, 0.875, 1.00)), len(
    subset(0.75, 0.875, 0.875, 1.00)), len(subset(0.875, 1.0, 0.875, 1.00))
LengMat = [[A1, A2, A3, A4, A5, A6, A7, A8], [B1, B2, B3, B4, B5, B6, B7, B8], [C1, C2, C3, C4, C5, C6, C7, C8], [D1, D2, D3, D4, D5, D6, D7, D8],
           [E1, E2, E3, E4, E5, E6, E7, E8], [F1, F2, F3, F4, F5, F6, F7, F8], [G1, G2, G3, G4, G5, G6, G7, G8], [H1, H2, H3, H4, H5, H6, H7, H8]]
# print(np.matrix(LengMat))

# print(PayoffDict2)
# print(PayoffDict['MG5500'])

# These are equations to plot payoff structures along dimensions.  These strings are inputs to the dimensionalizer function.
# Align = 'abs(a1 - b1 + b2 - a2)/ \
#    (abs(a1 - b1 + b2 - a2) + abs(a1 - b1 + a2 - b2))'



def agentfunct(payoffs, chooser, predictor, chooparams, predparams, worldparams, now=0, history=None):
    Paylabel, Aself, Aothr, Bself, Bothr = payoffs[4], payoffs[0], payoffs[1], payoffs[2], payoffs[3]
    pswitch, pcatch = worldparams['p(switch)'], worldparams['p(catch)']
    s, o = 2 * chooparams['Self'] - 1, 2 * predparams['Other'] - 1
    g, e = 2 * chooparams['Guilt'] - 1, 2 * predparams['Envy'] - 1
    s, o, g, e = s * sweight, o * oweight, g * gweight, e * eweight
    if now > 0:
        pairhist, cclist, cplist, pclist, pplist = {}, {}, {}, {}, {}
        for i in list(range(0,now)):
            # print("space")
            # print(chooser, history[i][chooser])
            # print(predictor, history[i][predictor])
            """Pair history stores the history of prior interactions between the current chooser and predictor."""
            # if (chooser == history[i][chooser]['Chooser'][1] or chooser == history[i][chooser]['Predictor'][1]) and \
            #     (predictor == history[i][predictor]['Chooser'][1] or predictor == history[i][predictor]['Predictor'][1]):
            #     present = 'Round ' + str(history[i][chooser]['Round'])
            #     pairhist[present] = {'Chooser': history[i][chooser]['Chooser'][1], 'Predictor': history[i][chooser]['Predictor'][1], \
            #         'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
            present = 'Round ' + str(history[i][chooser]['Round'])
            if chooser == history[i][chooser]['Chooser'][1] and predictor == history[i][predictor]['Predictor'][1]:
                
                cclist[present] = {'Chooser': chooser, 'Predictor': predictor, \
                    'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}

            elif chooser == history[i][chooser]['Predictor'][1] and predictor == history[i][predictor]['Chooser'][1]:

                cplist[present] = {'Chooser': predictor, 'Predictor': chooser, \
                    'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}

            # elif predictor == history[i][predictor]['Chooser'][1] and chooser == history[i][chooser]['Predictor'][1]:

            #     pclist[present] = {'Chooser': history[i][chooser]['Chooser'][1], 'Predictor': history[i][chooser]['Predictor'][1], \
            #         'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}

            # elif predictor == history[i][predictor]['Predictor'][1] and chooser == history[i][chooser]['Chooser'][1]:

            #     pplist[present] = {'Chooser': history[i][chooser]['Chooser'][1], 'Predictor': history[i][chooser]['Chooser'][1], \
            #         'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
            print(chooser, predictor)
            print(cplist)
                # if chooser == history[i][chooser]['Predictor'][1]:
                #     cplist = [present] = {'Chooser': history[i][chooser]['Chooser'][1], 'Predictor': history[i][chooser]['Predictor'][1], \
                #         'Choice': history[i][chooser]['Choice'], 'Reject': history[i][chooser]['Reject']}
        # if pairhist != {}:
        #     choosum = []
        #     # print(chooser, predictor, pairhist)
        #     for i, v in enumerate(pairhist.values()):
        #         if v['Predictor'] == chooser:
        #             x, y = v['Choice'][0] - v['Reject'][0], v['Choice'][1] - v['Reject'][1]
        #             choosum.append(slopedegs(x, y, True))
        #         # print("I'm reacting!")
        #     meets, choosum = len(choosum), sum(choosum)
        #     if meets != 0:
        #         xweight, yweight = math.cos(angletheta(choosum/meets,True)), math.sin(angletheta(choosum/meets,True))
        #         s, o = (s * vweight + xweight * meets) / (vweight + meets), (o * vweight + yweight * meets) / (vweight + meets)
    A = s * (Aself - Bself) + o * (Aothr - Bothr) - g * (Aself - Aothr) - e * (Aothr - Aself)
    B = s * (Bself - Aself) + o * (Bothr - Aothr) - g * (Bself - Bothr) - e * (Bothr - Bself)
    if A == B:
        A, B = randexcept(0.5), randexcept(0.5)
    choice = [Aself, Aothr] if A > B else [Bself, Bothr]
    reject = [Bself, Bothr] if A > B else [Aself, Aothr]
    return choice, reject

