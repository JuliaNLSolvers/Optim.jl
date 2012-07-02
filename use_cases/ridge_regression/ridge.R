library('glmnet')
x <- matrix(c(1, 2, 3, 4),
		        nrow = 2,
			      byrow = TRUE)
y <- c(1, 2)
coef(glmnet(x, y, lambda = 1, alpha = 0, standardize = FALSE))

lm(y ~ x)
