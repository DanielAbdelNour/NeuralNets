## Neural Network from Scratch ##
## -- By Daniel Abdel-Nour -- ##

# init and normalise data
x <- mtcars$disp/max(mtcars$disp)
y <- mtcars$mpg/max(mtcars$mpg)
mydata <- as.matrix(data.frame(x, y))

# INITIALISE FUNCTIONS AND VARS
niter <- 1000

sig <- function(x){
  return(1/(1+exp(-x)))
}

sig.prime <- function(x){
  return(sig(x) * (1-sig(x)))
}

loss <- function(yhat){
  return(sum(0.5 * (y - yhat)^2))
}

loss.prime <- function(yhat){
  return(-(y - yhat))
} 

#######################################
#
# setup a [(1),(3),(1)] NN
# 
#              +---+
#         +----+z1 +----+
#         |    +---+    |
#         |             |
#         |             |
# +---+   |    +---+    |    +---+
# | x +--------+z2 +---------+ y |
# +---+   |    +---+    |    +---+
#         |             |
#         |             |
#         |    +---+    |
#         +----+z3 +----+
#              +---+
#########################################

## FORWARD PASS ##

# init weights and layers
set.seed(123)
w1 <- matrix(runif(3), ncol = 3) 
w2 <- matrix(runif(3), ncol = 3) 
# l2 <- matrix(0,nrow = 3, ncol = dim(mydata)[1])

# Each node in the hidden layer will calculate the dot product of inputs and their weights.
# These are not summed in this example because we're only calculating a single input, therefore 
# every node only recieves a single weight.
l2 <- sig(mydata[,1] %*% w1)
# Multiply each column in l2 by the column element in w2
l3 <- sweep(l2, 2, w2, "*")
# take the sum of each row in l3 and apply sigmoid - output from forward pass is done!
l3 <- sig(rowSums(l3)) 




