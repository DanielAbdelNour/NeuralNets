## Neural Network from Scratch ##
## -- By Daniel Abdel-Nour -- ##

# init and normalise data
x <- mtcars$disp/max(mtcars$disp)
y <- mtcars$mpg/max(mtcars$mpg)

mydata <- as.matrix(data.frame(x, y))

# INITIALISE FUNCTIONS AND VARS
niter <- 10000
l.rate <- 0.1

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
  return(y - yhat)
} 

#######################################
#
# setup a [(1),(3),(1)] NN
# 
#   l1          l2             l3
#              +---+
#         +----+z1 +----+
#         |    +---+    |
#         |             |
#         w11           w21    
# +---+   |    +---+    |       +---+
# | x +---w12----+z2 +--w22-----+ y |
# +---+   |    +---+    |       +---+
#         |             |
#         w13           w23
#         |    +---+    |
#         +----+z3 +----+
#              +---+
#########################################

## FORWARD PASS ##

# init weights and layers
set.seed(123)
w1 <- matrix(runif(3,-1,1), ncol = 3) 
w2 <- matrix(runif(3,-1,1), ncol = 3) 

# Each node in the hidden layer will calculate the dot product of inputs and their weights.
# These are not summed in this example because we're only calculating a single input, therefore 
# every node only recieves a single weight.
l2a <- sig(mydata[,1] %*% w1)
# Multiply each column in l2 by the column element in w2
l3s <- sweep(l2a, 2, w2, "*")
# take the sum of each row in l3 and apply sigmoid - output from forward pass is done!
l3s <- rowSums(l3s)
# take the sigmoid activation of the output layer
l3a <- sig(l3s)

## BACK PROPAGATION ##

# repeat n times till convergance 
for(i in c(1:niter)){
  # derivative of error with respect to w2
  l3_error <- loss.prime(l3a) * sig.prime(l3a)
  l2_error <- t(t(w2) %*% l3_error) * sig.prime(l2a)
  
  dw2 <- t(t(l2a) %*% l3_error)
  dw1 <- colSums(t(t(l2_error) %*% mydata[,1]))
  
  w1 <- w1 + (dw1 * l.rate)
  w2 <- w2 + (dw2 * l.rate)
  
  if(i < niter){
    l2a <- sig(mydata[,1] %*% w1)
    l3s <- sweep(l2a, 2, w2, "*")
    l3s <- rowSums(l3s)
    l3a <- sig(l3s)
  }
}


test <- function(x){
  l2a <- sig(x %*% w1)
  # Multiply each column in l2 by the column element in w2
  l3s <- sweep(l2a, 2, w2, "*")
  # take the sum of each row in l3 and apply sigmoid - output from forward pass is done!
  l3s <- rowSums(l3s)
  # take the sigmoid activation of the output layer
  l3a <- sig(l3s)
  
  return(l3a)
}





