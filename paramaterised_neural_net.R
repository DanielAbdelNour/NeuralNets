###    Dan Net V1     ###
# By Daniel Abdel-Nour #
########################

# TEMP VARS #
hidden.layers <- c(100,30,20,60,40)
x <- mtcars$mpg/max(mtcars$mpg)
y <- mtcars$disp/max(mtcars$disp)

#x <- seq(0,10,0.5)
#y <- sin(x) + 1

alpha <- 0.1
reg <- 0.01
max.iter <- 20000
add.bias <- T
# x <- matrix(runif(10,-1,1),nrow = 5, ncol =2)
# # # # # # # 

## TRAIN FUNCTION ##
dan.net <- function(x, y, hidden.layers = c(3,2), cost.function = "sse", alpha = 0.1, max.iter = 100000, add.bias = T, reg = 0.01){
  x <- x/max(x)
  y <- y/max(y)
  x <- as.matrix(x)
  y <- as.matrix(y)
  
  # initialise required functions
  sig <- function(x){
    return(1/(1 + exp(-x)))
  }
  
  sig.prime <- function(x){
    x * (1 - x)
  }
  
  sse.prime <- function(y_hat){
    return(-(y - y_hat))
  }
  
  cross.entropy.prime <- function(y_hat){
    return((y_hat - y)/(y_hat*(1 - y_hat)))
  }
  
  # set var for number of hidden layers
  n.hl <- length(hidden.layers)
  
  # init delta list for the derivatives of each layer
  delta <- list()
  
  # init bias nodes
  bias <- list()
  ones <- matrix(1, nrow = dim(x)[1])
  
  for (b in c(1:(n.hl + 1))){
    # make sure we only add one node to output layer
    if(b == (n.hl + 1))
      bias[[b]] <- matrix(runif(1, -1, 1), nrow = 1, ncol = 1)
    else
      bias[[b]] <- matrix(runif(hidden.layers[b] * 1, -1, 1), nrow = 1, ncol = hidden.layers[b])
  }
  
  if(!add.bias)
    bias <- lapply(bias, function(x){x*0})
  
  # init weights and layers
  l <- list()
  l[[1]] <- x
  wts <- list()
  wt0 <- matrix(runif(dim(x)[2] * hidden.layers[1], -1, 1), nrow = dim(x)[2], ncol = hidden.layers[1])
  wts[[1]] <- wt0
  
  for(p in c(1:(n.hl))){
    if(!is.na(hidden.layers[(p + 1)]))
      wtn <- matrix(runif(hidden.layers[p] * hidden.layers[(p + 1)], -1, 1), nrow = hidden.layers[p], ncol =  hidden.layers[(p + 1)])
    else
      wtn <- matrix(runif(hidden.layers[p] * 1, -1, 1), nrow = hidden.layers[p], ncol = 1)
     
    wts[[(p+1)]] <- wtn
  }
  
  # start training 
  for(i in c(1:max.iter)){
    
    # FORWARD PASS #
    # apply the activiation function to the dot product of layers and corrosponding weights and add bias (via sweep)
    for(q in c(1:(n.hl + 1))){
      if(add.bias)
        l[[(q + 1)]] <- sig((l[[q]] %*% wts[[q]]) + (ones %*% bias[[q]]))
      else
        l[[(q + 1)]] <- sig(l[[q]] %*% wts[[q]])
    }
    
    # BACKPROPAGATION #
    for(r in c((n.hl + 1):1)){
      # compute the derivative of the loss function at the output later else compute hidden layers
      if(r == (n.hl + 1))
        delta[[r]] <- cross.entropy.prime(l[[(r + 1)]]) * sig.prime(l[[(r + 1)]])
      else
        delta[[r]] <- (delta[[(r + 1)]] %*% t(wts[[(r + 1)]])) * sig.prime(l[[(r + 1)]])
    }
    
    # update weights using gradient descent
    for(t in c((n.hl + 1):1)){
      wts[[t]] <- wts[[t]] - (alpha * t(l[[t]]) %*% delta[[t]])
      
      if(add.bias)
        bias[[t]] <- bias[[t]] - (alpha * (t(ones) %*% delta[[t]]))
    }
    
    # output progress showing absolute error at % of completion
    if(i %% 1000 == 0){
      print(paste(mean((y - l[[length(l)]])^2), " : ", (i/max.iter) * 100))
    }
  }
  
  # return a dataframe of the final network result, weights, and biases for predicting new observations
  final.net <- list(net.result = l[[length(l)]],
                    net.wts = wts,
                    net.bias = bias)
  
  return(final.net)
}


## PREDICT FUNCTION ##
dan.predict <- function(x, net.model){
  x <- x/max(x)
  x <- as.matrix(x)
  ones <- matrix(1, nrow = dim(x)[1])
  pr <- list()
  pr[[1]] <- x
  
  sig <- function(x){
    return(1/(1 + exp(-x)))
  }
  
  for(r in c(1:length(net.model$net.wts))){
    pr[[(r + 1)]] <- sig((pr[[r]] %*% net.model$net.wts[[r]]) + (ones %*% bias[[q]]))
  }
  
  return(pr[[length(pr)]])
}










