x <- mtcars$wt/max(mtcars$wt)
y <- mtcars$mpg/max(mtcars$mpg)

#mydata <- data.frame(x1 = x$x1, x2 = x$x2, y)

x <- matrix(c(0,1,0,1,1,1,0,0), nrow = 4)
y <- c(1,0,0,1)

x <- seq(-10, 10, 0.1)
y <- sin(x)
x <- x/max(x)
y <- y/max(y)

alpha <- 1

#set.seed(1234)
syn0 <- matrix(runif(10), nrow = 1, ncol = 10) # 1 by 4
syn1 <- matrix(runif(10,-1,1), nrow = 10, ncol = 1) # 4 by 1

sig <- function(x){
    return(1/(1 + exp(-x)))
}

sig.prime <- function(x){
    x * (1 - x)
}

loss.prime <- function(y_hat){
    return(-(y - y_hat))
}

for(i in c(1:1000000)){
    l1 <- sig(x %*% syn0)
    l2 <- sig(l1 %*% syn1)
    
    l2_delta <- loss.prime(l2) * sig.prime(l2)
    l1_delta <- (l2_delta %*% t(syn1)) * sig.prime(l1)
    
    syn1 <- syn1 - (alpha * t(l1) %*% l2_delta)
    syn0 <- syn0 - (alpha * t(x) %*% l1_delta)
    
    if(i %% 1000 == 0){
        print(paste(abs(sum(y - l2)), " : ", (i/1000000) * 100))
    }
}


