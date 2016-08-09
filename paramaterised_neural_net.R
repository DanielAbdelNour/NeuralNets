###    Dan net V1     ###
# By Daniel Abdel-Nour #
########################

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
    
    # parameterise hidden layers
    n.hl <- length(hidden.layers)
    
    # init weights
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
}