import matplotlib.pyplot as plt
 
# line 1 points
x1 = [10,20,30,40,50]
y1 = [1311,2176,2462,3261,5989]
# plotting the line 1 points 
plt.plot(x1, y1, label = "sequential code")
 
# line 2 points
x2 = [10,20,30,40,50]
y2 = [279,499,769,1019,1249]
# plotting the line 2 points 
plt.plot(x2, y2, label = "parallel code")
 
# naming the x axis
plt.xlabel('x - axis (Epoch count)')
# naming the y axis
plt.ylabel('y - axis (Time taken)')
# giving a title to my graph
plt.title('Comparison of the timings of sequential and parallel codes')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()