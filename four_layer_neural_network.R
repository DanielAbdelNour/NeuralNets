x <- mtcars$disp/max(mtcars$disp)
y <- mtcars$cyl/max(mtcars$cyl)

#mydata <- data.frame(x1 = x$x1, x2 = x$x2, y)

# x <- matrix(c(0,1,0,1,1,1,0,0), nrow = 4)
# y <- c(1,0,0,1)
# 
# x <- seq(-10, 10, 0.5)
# y <- sin(x)
# x <- x/max(x)
# y <- y/max(y)

alpha <- 0.7

#set.seed(1234)
syn0 <- matrix(runif(5), nrow = 1, ncol = 5) # 1 by 3
syn1 <- matrix(runif(15, -1, 1), nrow = 5, ncol = 5) # 3 by 3
syn2 <- matrix(runif(5, -1, 1), nrow = 5, ncol = 1) # 3 by 1

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
    l3 <- sig(l2 %*% syn2)
    
    l3_delta <- loss.prime(l3) * sig.prime(l3)
    l2_delta <- (l3_delta %*% t(syn2)) * sig.prime(l2)
    l1_delta <- (l2_delta %*% t(syn1)) * sig.prime(l1)
    
    syn2 <- syn2 - (alpha * t(l2) %*% l3_delta)
    syn1 <- syn1 - (alpha * t(l1) %*% l2_delta)
    syn0 <- syn0 - (alpha * t(x) %*% l1_delta)
    
    if(i %% 1000 == 0){
        print(paste(abs(sum(y - l2)), " : ", (i/1000000) * 100))
    }
}


