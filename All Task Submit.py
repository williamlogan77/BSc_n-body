#Import all of our module that we use
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Initialise our time arrays, with different step sizes
SecondsInYear = 24 * 365 * 60 * 60
t_hours = np.arange(0, SecondsInYear + 3600, 3600)
t_minutes = np.arange(0, SecondsInYear + 3600, 60)
t_days = np.arange(0, SecondsInYear + 3600, (60 * 60 * 24))

#Define the inital conditions for the planets and the constants that we use
G =  6.67408e-11
Jupiter_state = np.array([0, 0, 0, 0])
Jupiter_mass = 1.89819e27
Io_state = np.array([-421700000, 0, 0, -17334])
Io_mass = 8.9319e22
Europa_state = np.array([-670900000, 0, 0, -13740])
Europa_mass = 4.7998e22
Ganymede_state = np.array([-1070400000, 0, 0, -10880])
Ganymede_mass = 1.4819e23
Callisto_state = np.array([-1882700000, 0 ,0, -8204])
Callisto_mass = 1.0759e23
Moons = ["Jupiter", "Io", "Europa", "Ganymede", "Callisto"]

#Combine all of the inital conditions into two arrays, one for position and velocity, one for mass
moon_data = np.array([Io_state, Europa_state, Ganymede_state, Callisto_state])
MoonMass = np.array([Io_mass, Europa_mass, Ganymede_mass, Callisto_mass])

'''
The below function calculates the distance between two bodies and returns the distance 
between them, both the x and y component, and the mass of the body considered.
Returning these here makes using the function later on a lot easier to use as it is all returned in one place.
'''
def derivative_body(body1, body2, b1mass ):
    Jupiter_posx = body1[0] #Assign the x position
    Jupiter_posy = body1[1] #Assign the y position

    Moon_posx = body2[0] #Assign the x position
    Moon_posy = body2[1] #Assign the y position

    R_x = (Jupiter_posx - Moon_posx) #Compare the x-position
    R_y = (Jupiter_posy - Moon_posy) #Compare the y-position
    R = np.sqrt( (R_x ** 2) + (R_y ** 2) ) #Find the absolute distance

    return (R_x, R_y, R, b1mass) #Return our variables used

'''
The below function is the function that we pass through to the solve_ivp function.
In this solution, I have used the same function throughout all of the questions so that it handles each case as they come.
So this solution can handle each moon individually as well as considering the moons together (see the Moons_system function).
Initially, we pass through the parameters that the solver need, t and Moon, where Moon is the changing velocity and acceleration.
We also pass through a few keyword variables, these take the same value for these variables and can change if specified.
Doing this calculation with a function means we can pass as many moons through as we need, so it is modular regardless of amount of moons
'''
def Derrivative(t, Moon, Moon_states = moon_data, Consider_Moons = False, Planet_state = Jupiter_state, M = Jupiter_mass, Moon_mass = MoonMass, G = G):
    R_x, R_y, R, M = derivative_body(Planet_state, Moon, M) 
    #Call the aforementioned function to calculate the inital distance between Jupiter and the moon in question, setting the mass to be the mass of Jupiter
    a_x = G * (M / (R ** 2)) * (R_x / R) #Calculate the acceleration in x due to jupiter on the moon 
    a_y = G * (M / (R ** 2)) * (R_y / R) #Calculate the acceleration in y due to jupiter on the moon
    if Consider_Moons == True: #However if we have to consider the acceleration due to the other moons pass through this phrase
        for i in range(len(Moon_states)//4): #Loop through the amount of moons. Since we pass through a 1d array of all of the moons, we have to consider which part is related to which moon.
            j = len(Moon_states) // 4 #We set the index of the start of each set of variables for one moon
            k = j + 4 #We set the index of the end of each set of variables for one moon
            current_moon = Moon_states[j:k] #Here we take out the variables for the moon we consider and put it in a variable
            R_x, R_y, R, M = derivative_body(current_moon, Moon, Moon_mass) #Calculate the relevant distance between all of the moons and the moon we considered
            a_x += ((G * (M / (R ** 2)) * (R_x / R))) #Calculate the corresponding x acceleration and add them onto the already calculated accleration
            a_y += ((G * (M / (R ** 2)) * (R_y / R))) #Calculate the corresponding y acceleration and add them onto the already calculated accleration
            #We loop over this and add the accleration relative to each of the bodies considered to a variable until we have considered all bodies
    Moon_dv_xdt = a_x #Swap the variable names to be relevant to the problem 
    Moon_dv_ydt = a_y #Swap the variable names to be relevant to the problem
    Moon_velx = Moon[2] #Set the velocity to be the inital condition, this is changed when we pass it through the function
    Moon_vely = Moon[3] #Set the velocity to be the inital condition, this is changed when we pass it through the function
    return [Moon_velx, Moon_vely, Moon_dv_xdt, Moon_dv_ydt] #Return the two positions and velocities

'''
We call the below function for when there is more than one gravitational influence that we need to consider 
It is essentially just a wrapper function that takes the data and modifies it so that we can consider the change in position of each moon simultaneously
We only need to pass through the data that the derrivative function takes, as we modify the data and call the derrivative function again
'''
def Moons_System(t, ivp_states, Planet_state = Jupiter_state, M = Jupiter_mass, Moon_mass = MoonMass, G = G):
    system_state=[] #Initalise a list to add the data of each moon to, to then return it for each moon
    for i in range(len(ivp_states)//4): #Similar to earlier, loop through the data and find each of the set of data for each moon
        mystart=i*4 #Create a variable that changes every time we go through the loop to find the start of each set of data for each moon
        myend=mystart+4 #Create a variable that changes every time we go through the loop to find the end of each set of data for each moon
        mymoon=ivp_states[mystart:myend] #Set the moon we are looking to the relevant set of data
        all_moons=ivp_states #Store the data for every moon in this variable
        other_moons = np.delete(all_moons, [range(mystart,myend)]) #Remove the elements that correspond to the moon that we are doing calculations for
        system_state.append(Derrivative(t, mymoon, Moon_states=other_moons,Consider_Moons =True, Moon_mass=Moon_mass[i])) 
        #Call the derrivative function for the moon in question and store the corresponding value in the list.
    flat_system_state = [] #Initalise a list to move data into a 1d array so that we can use the solve_ivp function for each moon simulataneously
    for state in system_state:#Loop through the set of arrays
        for item in state: #Loop through each index of each array
            flat_system_state.append(item) #Append each item to the list. This basically converts a 2d list into a 1d list of readable data for solve_ivp
    return flat_system_state #Return the list of each state of each moon

'''
The below function simply calls the solve_ivp function with all the conditions set, so that we don't have to type it out every time.
We also have a few keyword arguments, in the case that we need to change the time scale, or if we want to consider different bodies effect.
'''
def solution(moon_state, time = t_hours, consider_moons = False):
    if consider_moons == True: #If we consider extra bodies run this loop
        sols = solve_ivp(lambda t, Moon: Moons_System(t, Moon) , (0, time[-1]), (moon_state), t_eval = time, rtol = 1e-10, atol = 1e-12) 
        #Call the solve_ivp function using the Moons_System function, as it allows us to consider multible bodies, using the lambda function, which lets 
        #us ignore all other arguments other than the ones that the solver wants

    else: #In every other case (we don't consider extra bodies) run this loop
        sols = solve_ivp(lambda t, Moon: Derrivative(t, Moon, Consider_Moons = consider_moons) , (0, time[-1]), (moon_state), t_eval = time, rtol = 1e-10, atol = 1e-12)
        #Call the solve_ivp function using the Derrivative function, as we only need to consider the acceleration due to Jupiter
    return sols #Return our relevant solution

moon_data = np.reshape(moon_data, (16))#Here I reshape the previously 2d array to a state that the function can use

#Now all the leg work is done, all we need to do is call the functions and manipulate the output to show what we need

Io_solution = solution(Io_state) #Call the solution function, and put the solution into a variable to use later on so that we don't have to keep calling the solver
##############################################################
#Task 1c

fig, ax = plt.subplots() #Initalise the plot

ax.title.set_text("The Orbit of Io") #Set the title of the plot
ax.plot(Io_solution.y[1] * 1e-6, Io_solution.y[0] * 1e-6) #Plot the x coordinate against the y coordinate in 10^3km
ax.set_ylabel("y-position (10e3 Km)") #Set the y label
ax.set_xlabel("x-position (10e3 Km)") #Set the x label
ax.plot(0, 0, "o") #Plot jupiter as a dot in the middle

plt.show() #Give us a GUI for the plot
################################################################
#Task 1d

fig, ax1 = plt.subplots() #Initialise the plot

ax1.plot(Io_solution.y[0] * 1e-6, (Io_solution.y[2] * 1e-6), "-", color = "blue") #Plot the x coordinate against the x velocity on the left axis

ax1.title.set_text("x-y Velocity against x-Position") #Set the plot title
ax1.set_xlabel("x-position (10e3 Km)") #Set the x label
ax1.set_ylabel("Evolution of y velocity") #Set the left y axis label
ax1.spines['left'].set_color('blue') #Change the colour of the left axis ticks to blue
ax1.tick_params(axis='y', colors = "blue") #Change the colour of the left axis to blue
ax1.yaxis.label.set_color('blue') #Change the colour of the left axis text to blue

ax1.semilogy() #Change the y scale to semilog

ax2 = ax1.twinx() #Copy the left y axis but onto the right handside

ax2.plot(Io_solution.y[0] * 1e-6 , (Io_solution.y[3] * 1e-6), ".", color = "red") #Plot the x coordinate against the y velocity on the right axis

ax2.set_ylabel("Evolution of x velocity") #Set the right axis label
ax2.spines['right'].set_color('red') #Change the colour of the right axis ticks to red
ax2.tick_params(axis='y', colors = "red") #Change the colour of the right axis to red
ax2.yaxis.label.set_color('red') #Change the colour of the right axis label to red

ax2.semilogy() #Change the y scale of the right axis to semilog

plt.show() #Give us a GUI for the plot

###########################################################
#Task2b
Europa_solution = solution(Europa_state) #Store the solution for Europa in a variable so we dont have to keep calling the function
Ganymede_solution = solution(Ganymede_state) #Store the solution for Ganymede in a variable so we dont have to keep calling the function
Callisto_solution = solution(Callisto_state) #Store the solution for Callisto in a variable so we dont have to keep calling the function

fig, ax = plt.subplots() #Initialise the plot

ax.title.set_text("Plot of Jupiters first four Moons orbits") #Set the title of the plot

ax.plot(0, 0, "o", color = "brown") #Plot jupiter as a dot in the center
ax.plot( (Io_solution.y[1] * 1e-6), (Io_solution.y[0] * 1e-6), color = "red" ) #Plot the Io x coordinate against the Io y coordinate
ax.plot( (Europa_solution.y[1] * 1e-6), (Europa_solution.y[0] * 1e-6), color = "green" ) #Plot the Europa x coordinate against the Europa y coordinate
ax.plot( (Ganymede_solution.y[1] * 1e-6), (Ganymede_solution.y[0] * 1e-6), color = "blue" ) #Plot the Ganymede x coordinate against the Ganymede y coordinate
ax.plot( (Callisto_solution.y[1] * 1e-6), (Callisto_solution.y[0] * 1e-6), color = "orange" ) #Plot the Callisto x coordinate against the Callisto y coordinate

ax.set_ylabel("y position (10e3 Km)") #Set the y label
ax.set_xlabel("x position (10e3 Km)") #Set the x label

ax.legend(Moons, loc="upper right") #Show a legend with the appropriate labels

plt.show() #Give us a GUI for the plot
#################################################################
#Task 2b

fig, ax1 = plt.subplots() #Initalise the plot

ax1.plot(Io_solution.y[0] * 1e-6, (Io_solution.y[2] * 1e-6), "-", color = "red" ) #Plot the Io y velocity against its x position
ax1.plot(Europa_solution.y[0] * 1e-6, (Europa_solution.y[2] * 1e-6), "-", color = "green") #Plot the Europa y velocity against its x position
ax1.plot(Ganymede_solution.y[0] * 1e-6, (Ganymede_solution.y[2] * 1e-6), "-", color = "blue") #Plot the Ganymede y velocity against its x position
ax1.plot(Callisto_solution.y[0] * 1e-6, (Callisto_solution.y[2] * 1e-6), "-", color = "orange") #Plot the Callisto y velocity against its x position

ax1.title.set_text("x-y Velocity against x-Position") #Set the title of the plot
ax1.set_xlabel("x-position (10e3 Km)") #Set the x label of the plot
ax1.set_ylabel("Evolution of y velocity (straight lines)") #Set the y label of the plot

ax1.semilogy() #Change the y scale to a semilog scale

ax2 = ax1.twinx() #Copy the left y axis to the right hand side

ax2.plot(Io_solution.y[0] * 1e-6 , (Io_solution.y[3] * 1e-6), ".", color = "red") #Plot the Io x velocity against its x position
ax2.plot(Europa_solution.y[0] * 1e-6, (Europa_solution.y[3] * 1e-6), ".", color = "green") #Plot the Europa x velocity against its x position
ax2.plot(Ganymede_solution.y[0] * 1e-6, (Ganymede_solution.y[3] * 1e-6), ".", color = "blue") #Plot the Ganymede x velocity against its x position
ax2.plot(Callisto_solution.y[0] * 1e-6, (Callisto_solution.y[3] * 1e-6), ".", color = "orange") #Plot the Callisto x velocity against its x position

ax2.set_ylabel("Evolution of x velocity (dots)") #Set the right hand y axis label

ax2.semilogy() #Change the y scale to a semilog scale 

ax1.legend(Moons[1:], loc="upper left") #Set a legend with the corresponding values

plt.show() #Give us a GUI for the plot

#########################################################
'''
Task 2c
Making the same simulation with the RK2 solver would cause the simulation to be much more inaccurate.
The direct comparison between RK4 and RK2 is that there are more points at which the curves are evaluated in, thus giving us a higher 
accuracy over a longer time period.
The major challenge by implementing an RK2 solver is that we would have to create the function ourselves. If we compare this to the premade
solve_ivp function from scipy, there has been a lot more testing and use cases with the scipy function, thus deeming it more reliable. Our
"homemade" RK2 function would most likely require lots of testing to be used in all cases.
To be able to reuse our own RK2 function, we must also make sure that we can catch any errors, so that we’re not debugging our code and the function at the same time
To implement the RK2 function, like in class, we can use Euler's method, and expand it from there by implementing each step, similar to the RK4 function just without as many steps
'''
#######################################################
#Task 2d

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (10, 10)) # Initalise the plot, with 3 plots and a figure size

ax1.plot((t_minutes / (SecondsInYear)), solution(Io_state, t_minutes).y[1] * 1e-6) #Plot the time with steps of minutes against the y coordinate

ax1.set_ylabel("y position (10e3 Km)") #Set the y label
ax1.set_xlabel("Normallised time (Evaluated every minute)") #Set the x label

ax2.plot((t_hours / (SecondsInYear)), (solution(Io_state, t_hours).y[1] * 1e-6)) #Plot the time with steps of hours against the y coordinate

ax2.set_ylabel("y position (10e3 Km)") #Set the y label
ax2.set_xlabel("Normallised time (Evaluated every hour)") #Set the x label

ax3.plot((t_days / SecondsInYear), solution(Io_state, t_days).y[1] * 1e-6) #Plot the time with steps of days against the y coordinate

ax3.set_ylabel("y position (10e3 Km)") #Set the y label
ax3.set_xlabel("Normallised time (Evaluated every day)") #Set the x label

plt.show() #Give us a GUI for the plot

######################################################
#Task 3
#Here we define a function just so we don't have to write the same code out over and over
#It calculates the absolute distance between two bodies
def Radius(Solution):
    Radii = np.sqrt( (Solution.y[0][u] ** 2) + (Solution.y[1][u] ** 2))
    return(Radii)  


f = open("moon_orbits.txt", "w") #Open our file for writing

f.write("A file containing time coordinate and orbital Radius of Jupiter's Moons" + "\n") #Write a header, and proceed to the next line
names = ["Time (years)", "Io", "Europa", "Ganymede", "Callisto"] #Set the names of the headings to be written
#Define the spaces so that we can use these so that the format of our file is appropriate
space1 = "         " 
space2 = "        "
space3 = "                  "
space4 = "              "
space5 = "            "

flot = ".5e" #Set the format that we want to use

f.write(names[0] + space2 + names[1] + space3 + names[2] + space4 + names[3] + space5 + names[4]) #Write the titles of each column to the names of the moons and the time coordinate

f.write("\n") #Write a new line

for u in range(len(t_hours)): #Start a loop for each time coordinate, so we can write each correpsonding coordinate
    f.write(str(format((t_hours[u] / SecondsInYear) , flot)) + str(space1) \
        + str(format(Radius(Io_solution), flot)) + str(space1) \
        + str(format(Radius(Europa_solution), flot)) + str(space1) \
        + str(format(Radius(Ganymede_solution), flot)) + str(space1) \
        + str(format(Radius(Callisto_solution), flot)))
    #Here we take a value for each coordinate and format it to the float format that we wanted and write an appropriate space between the next value and its previous
    #When we loop through this we take the corresponding value for each coordinate and write it to the file
    f.write("\n") #Go to the next line for the next values to be written

f.close() #Close our file after we are done writing to it

#########################################################
#Task 4
#Because the previous function that figures out the position and velocity of each moon only doesn't consider the change in position from the releavnt moons, 
#We must resolve all of the positions of the bodies all at once. This is what the Solutions variable holds, all of the positions and velocities of each moon in
#respect to each other
Solutions = solution(moon_data, consider_moons = True) #Solve the many bodied problem and store it in Solutions

fig, ax = plt.subplots(figsize = (10, 10))#Initalise the plot

ax.plot(0, 0, "o", color = "brown", label = "Jupiter") #Plot Jupiter
ax.plot( (Io_solution.y[1] * 1e-6), (Io_solution.y[0] * 1e-6), color = "red") #Plot the Io positions before we consider the other bodies
ax.plot( (Europa_solution.y[1] * 1e-6), (Europa_solution.y[0] * 1e-6), color = "green") #Plot the Europa positions before we consider the other bodies
ax.plot( (Ganymede_solution.y[1] * 1e-6), (Ganymede_solution.y[0] * 1e-6), color = "blue") #Plot the Ganymede positions before we consider the other bodies
ax.plot( (Callisto_solution.y[1] * 1e-6), (Callisto_solution.y[0] * 1e-6), color = "orange") #Plot the Callisto positions before we consider the other bodies

ax.plot( (Solutions.y[1] * 1e-6), (Solutions.y[0] * 1e-6), color = "orange", label = "Io") #Plot the Io positions after we consider the other bodies
ax.plot( (Solutions.y[5] * 1e-6), (Solutions.y[4] * 1e-6), color = "blue", label = "Europa") #Plot the Europa positions after we consider the other bodies
ax.plot( (Solutions.y[9] * 1e-6), (Solutions.y[8] * 1e-6), color = "green", label = "Ganymede") #Plot the Ganymede positions after we consider the other bodies
ax.plot( (Solutions.y[13] * 1e-6), (Solutions.y[12] * 1e-6), color = "red", label = "Callisto") #Plot the Callisto positions before we consider the other bodies

ax.legend(loc="upper right") #Show a legend with the appropriate labels

plt.show() #Give us GUI of our plot 
#Here we should see the bodies that have taken into account the other bodies be plotted over the Moons that havent considered each other


fig, axs = plt.subplots(4, 2)# Initalise subplots, in four rows and two columns



#In the first row
axs[0][0].plot(t_hours, Solutions.y[0]) #Plot the x position of Io while considering the other bodies against time
axs[0][0].plot(t_hours, Io_solution.y[0]) #Plot the x position of Io without considering other bodies against time
axs[0][1].plot(t_hours, Solutions.y[1]) #Plot the y position of Io while considering the other bodies against time
axs[0][1].plot(t_hours, Io_solution.y[1]) #Plot the y position of Io without considering other bodies against time

axs[1][0].plot(t_hours, Solutions.y[4]) #Plot the x position of Europa while considering the other bodies against time
axs[1][0].plot(t_hours, Europa_solution.y[0]) #Plot the x position of Europa without considering the other bodies against time
axs[1][1].plot(t_hours, Solutions.y[5]) #Plot the y position of Europa while considering the other bodies against time
axs[1][1].plot(t_hours, Europa_solution.y[1]) #Plot the y position of Europa without considering the other bodies against time 

axs[2][0].plot(t_hours, Solutions.y[8]) #Plot the x position of Ganymede without considering the other bodies against time
axs[2][0].plot(t_hours, Ganymede_solution.y[0]) #Plot the x position of Ganymede without considering the other bodies against time
axs[2][1].plot(t_hours, Solutions.y[9]) #Plot the y position of Ganymede while considering the other bodies
axs[2][1].plot(t_hours, Ganymede_solution.y[1]) #Plot the y position of Ganymede without considering the other bodies

axs[3][0].plot(t_hours, Solutions.y[12]) #Plot the x position of Callisto while considering the other bodies against time
axs[3][0].plot(t_hours, Callisto_solution.y[0]) #Plot the x position of Callisto without considering the other bodies against time
axs[3][1].plot(t_hours, Solutions.y[13]) #Plot the y position of Callisto while considering the other bodies against time
axs[3][1].plot(t_hours, Callisto_solution.y[1]) #Plot the y position of Callisto while considering the other bodies against time

for q in range(len(axs)):
    axs[q][0].set_ylabel("Pos Diff (10e3 Km)") #Here we loop through a label the y axis as the difference in position

plt.show() #Give us a GUI for the plot





