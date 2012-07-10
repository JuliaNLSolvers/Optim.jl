library('glmnet')

x <- matrix(c(1, 2, 3, 3, 5, 6),
		        nrow = 3,
			      byrow = TRUE)

y <- c(1, 2, 2)

coef(lm(y ~ x))

coef(glmnet(x, y, lambda = 0.0, alpha = 0.0, standardize = FALSE))
coef(glmnet(x, y, lambda = 1.0, alpha = 0.0, standardize = FALSE))
coef(glmnet(x, y, lambda = 10.0, alpha = 0.0, standardize = FALSE))
coef(glmnet(x, y, lambda = 100.0, alpha = 0.0, standardize = FALSE))
