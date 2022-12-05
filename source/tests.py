import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from linalg_interp import spline_function
from linalg_interp import gauss_iter_solve

#testing gauss iteration function

#testing jacobi with inital guess
A = np.array([[9,1,4],[2,7,1],[3,2,6]])
b = np.array([11,13,12])
guess = np.array([1,1,1])

sol = gauss_iter_solve(A,b, x0 =guess, alg='jacobi')

print('The solution using jacobi with x0 is',sol)

solution = np.linalg.solve(A,b)
print('The solution using algorithm (1) is', solution)

#testing jacobi with no inital guess 

A = np.array([[15,2,-4],[-1,17,1],[2,5,20]])
b = np.array([6,-2,10])

sol = gauss_iter_solve(A,b, alg='jacobi')

print('The solution using jacobi with no initial guess is',sol)

solution = np.linalg.solve(A,b)
print('The solution using algorithm (2)', solution)

#testing seidel with an inital guess
A = np.array([[9,1,4],[2,7,1],[3,2,6]])
b = np.array([11,13,12])
guess = np.array([1,1,1])

sol = gauss_iter_solve(A,b, x0 = guess)

print('The solution using seidel with x0 is',sol)

solution = np.linalg.solve(A,b)
print('The solution using algorithm (3) is', solution)

# testing seidel with no inital guess
A = np.array([[15,2,-4],[-1,17,1],[2,5,20]])
b = np.array([6,-2,10])

sol = gauss_iter_solve(A,b, alg='jacobi')

print('The solution using seidel with no initial guess is',sol)

solution = np.linalg.solve(A,b)
print('The solution using algorithm (4) is', solution)


# test for rhs matrix such that result is inverse of A

#testing seidel 
#A = ([[11,2,4],[3,9,2],[1,1,13]])
#b = np.eye(len(A))
#x = gauss_iter_solve(A, b)
#print('solution to seidel 2', x)
#solution = np.linalg.solve(A,b)
#print('algorith solution is',solution)
#ident = A @ x
#print('A @ x =', ident)



# spline function tests
# unit tests from linear, quadratic, and cubic function 


xd = np.linspace(-10,10,15)

#linear

yd1 =  2*xd
spline1 = spline_function(xd,yd1, order =1)       
y1 = spline1(xd)


#quadratic

yd2 = 0.5*(xd**2)
spline2 = spline_function(xd, yd2, order =2)
y2 = spline2(xd)

#cubic   
yd3 = 0.05*(xd**3)
spline3 = spline_function(xd, yd3, order =3)
y3 = spline3(xd)




plt.plot(xd,yd1, 'ro')
plt.plot(xd,y1,'m', label = 'linear function')

plt.plot(xd,yd2, 'bs')
plt.plot(xd,y2,'c', label = 'quadratic function')

plt.plot(xd,yd3, 'g^')
plt.plot(xd,y3,'tab:green', label = 'cubic function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spline interpolation for linear, quadratic and cubic functions')
plt.legend()
plt.show()


# unit test to compare results with scipy.interpolate

#testing for exponential 
yd4 = np.exp(xd)
spline4 = spline_function(xd, yd4, order=3)
y4 = spline4(xd)

spline_u = UnivariateSpline(xd, yd4, k=3, s=0)
y_u = spline_u(xd)


plt.plot(xd,yd4, 'ro')
plt.plot(xd,y4,'m', label = 'my spline function')
plt.plot(xd,y_u,'c', label = 'Univariate spline function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of my spline function and Univariate spline function for exponential function')
plt.legend()
plt.show()