# linear-regression

This is an implementation of Linear Regression using only `numpy` library 
and `python` base operations and types. For solving the linear regression
problem, the gradient descent optimizer and the quadratic error function are
applied.

The cost function to be minimized is the **quadratic error function**, whose 
gradient has a known analytical expression. 

The **gradient descent method** searches the parameters that minimizes the 
cost function, going in the opposite of the gradient, given that the 
gradient represents the maximum growth direction.

The challenge here is to implement this initial Machine Learning/Statistics 
algorithm using only python standard library and `numpy`, which performs 
linear algebra operations.

## Project structure

The project has the following structure:

```text
root
|
|--- data
|--- notebooks
|--- src
|    |--- cost_function
|    |--- optimizer
|    |--- preprocess
```

`data` - Example data files

`notebooks` - Applying the code implemented in `src` and 
data from `data` using Jupyter Notebooks.

`src` - Implementations for Linear Regression.

* `cost_function` - Calculating the cost function
and its gradient.
* `optimizer` - Classes representing each optimizer.
* `preprocess` - Useful preprocessing operations.


