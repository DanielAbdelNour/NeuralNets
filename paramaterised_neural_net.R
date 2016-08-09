###    Dan Net V1     ###
# By Daniel Abdel-Nour #
########################

# TEMP VARS #
hidden.layers <- c(3,2)
x <- mtcars$mpg/max(mtcars$mpg)
x <- matrix(runif(10,-1,1),nrow = 5, ncol =2)
# # # # # # 

dan.net <- function(x, y, hidden.layers = c(3,2), cost.function = "sse", alpha = 0.1, max.iter = 100000){
    x <- as.matrix(x)
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
    
    # init weights and layers
    l <- list()
    l[[1]] <- x
    wts <- list()
    wt0 <- matrix(runif(dim(x)[2] * hidden.layers[1]), nrow = dim(x)[2], ncol = hidden.layers[1])
    wts[[1]] <- wt0
    
    for(p in c(1:(n.hl))){
        if(!is.na(hidden.layers[(p + 1)]))
            wtn <- matrix(runif(hidden.layers[p] * hidden.layers[(p + 1)]), nrow = hidden.layers[p], ncol =  hidden.layers[(p + 1)])
        else
            wtn <- matrix(runif(hidden.layers[p] * 1), nrow = hidden.layers[p], ncol = 1)
        
        wts[[(p+1)]] <- wtn
    }
    
    # start training 
    for(i in c(1:max.iter)){
      
      # FORWARD PASS #
      # apply the activiation function to the dot product of layers and corrosponding weights
      for(q in c(1:(n.hl + 1))){
        l[[(q + 1)]] <- sig(l[[q]] %*% wts[[q]])
      }
      
      # BACKPROPAGATION #
      
      for(r in c((n.hl + 1):1)){
        # compute the derivative of the loss function at the output later
        if(r == (n.hl + 1))
          delta[[r]] <- cross.entropy.prime(l[[r]]) * sig.prime(l[[r]])
        else
          delta[[r]] <- (delta[[(r + 1)]] %*% t(wts[[r]])) * sig.prime(l[[r]])
      }
      
      l3_delta <- entropy.prime(l3) * sig.prime(l3)
      l2_delta <- (l3_delta %*% t(syn2)) * sig.prime(l2)
      l1_delta <- (l2_delta %*% t(syn1)) * sig.prime(l1)
      
      syn2 <- syn2 - (alpha * t(l2) %*% l3_delta)
      syn1 <- syn1 - (alpha * t(l1) %*% l2_delta)
      syn0 <- syn0 - (alpha * t(x) %*% l1_delta)
      
      if(i %% 1000 == 0){
        print(paste(abs(sum(y - l2)), " : ", (i/1000000) * 100))
      }
    }
}










