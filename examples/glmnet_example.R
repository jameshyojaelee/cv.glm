rm(list = ls())
library(glmnet) #This package contains ridge regressions & lasso functions
library(ISLR)
library(ggplot2)
library(gridExtra)


# Load data ---------------------------------------------------------------

data("Hitters")
head(Hitters)
?Hitters


# Get data ready ----------------------------------------------------------

#Many packages and functions in R require data input to meet particular
#specifications. The Ridge Regresssion and LASSO functions in glmnet
#have three important requirements that we must take into account:
#1. All observations with missing data must be removed in advance.
#2. The response variable must be stored in a vector by itself.
#3. The predictors must be formatted as a matrix (not dataframe) 
#   with only numerical inputs.

#Removing NAs
Hitters <- na.omit(Hitters)

#Separating response vector
y <- Hitters$Salary

#Putting all other variables in Hitters data in numerical form
x <- model.matrix(Salary ~ ., data = Hitters)[,-1]

#Notes on the line above:

#1. The "~ ." notation means all other variables in a dataset.
#This is convenient, but use it carefully!

#2. The model.matrix function is very convenient for creating
#the desired predictor matrix, as it automatically also 
#transforms any qualitative variables into dummy variables.

#3. The model.matrix output also includes the intercept as
#a first column of 1s, which we don't want for our current purposes,
#so we use the "[,-1]".


# Fitting ridge and lasso at different values of lambda -------------------

#Check default settings for glmnet function
?glmnet

#Pre-specifying vector of lambdas to assess
lambdas <- 10^seq(from = 6, to = -2, length = 100)

#Fitting ridge reg for all values of lambda:
ridge.mods <- glmnet(x = x, y = y, alpha = 0, lambda = lambdas)
#Note that alpha = 0 denotes the ridge penalty

#Fitting lasso for all values of lambda:
lasso.mods <- glmnet(x = x, y = y, alpha = 1, lambda = lambdas)
#Note that alpha = 1 denotes the lasso penalty

#Baseline OLS model
ols.mod <- lm(Salary ~ ., Hitters)

#Note: you can also fit a ridge or lasso at a single lambda value, e.g.:
#ridge.single.lambda <- glmnet(x = x, y = y, alpha = 0, lambda = 5)


# Comparing coefficients for different values of lambda -------------------

#There are 100 separate fits (at 100 different lambdas),
#so there are 100 separate sets of coefficients (20 coefficients each)
dim(coef(ridge.mods))
dim(coef(lasso.mods))

#Verifying sequence of lambdas
ridge.mods$lambda
lasso.mods$lambda

#OLS Baseline:
coef(ols.mod)

#Now we can extract the coefficients for a specific value of lambda for
#ridge and LASSO. How close to OLS do you think the following will be?
ridge.mods$lambda[100] #We will try the 100th lambda

coef(ridge.mods)[,100]
coef(lasso.mods)[,100]
#Note: coefficients conveniently returned in variables' original scale


#Now try larger value of lambda
ridge.mods$lambda[50]
coef(ols.mod)
coef(ridge.mods)[,50]
coef(lasso.mods)[,50]

#And even larger
ridge.mods$lambda[25]
coef(ols.mod)
coef(ridge.mods)[,25]
coef(lasso.mods)[,25]


# Plotting the coefficient paths ------------------------------------------

plot(ridge.mods, xvar = "lambda")
plot(lasso.mods, xvar = "lambda")


# Training vs. Test MSE ---------------------------------------------------

#Designating training vs. test observations
set.seed(321)
k <- sample(1:nrow(x), nrow(x)/2, replace = FALSE)

#Training the ridge and lasso models at all lambdas (with just training data)
ridge.train <- glmnet(x = x[k,], y = y[k], alpha = 0, lambda = lambdas)
lasso.train <- glmnet(x = x[k,], y = y[k], alpha = 1, lambda = lambdas)

#Prepping empty MSE vectors
train.mse.ridge <- rep(NA,length(lambdas))
train.mse.lasso <- rep(NA,length(lambdas))
test.mse.ridge <- rep(NA,length(lambdas))
test.mse.lasso <- rep(NA,length(lambdas))

#Computing training and test MSE for ridge and lasso at all lambdas:

for (i in 1:length(lambdas)){
  
  #Notes: When using the predict() function with a glmnet object,
  #1. Must specify the prediction data using "newx" argument
  #2. The argument "s" specifies the lambda value
  ridge.pred.train <- predict(ridge.train, newx = x[k,], s = lambdas[i])
  lasso.pred.train <- predict(lasso.train, newx = x[k,], s = lambdas[i])
  ridge.pred.test <- predict(ridge.train, newx = x[-k,], s = lambdas[i])
  lasso.pred.test <- predict(lasso.train, newx = x[-k,], s = lambdas[i])
  
  train.mse.ridge[i] <- mean((y[k] - ridge.pred.train)^2)
  train.mse.lasso[i] <- mean((y[k] - lasso.pred.train)^2)
  test.mse.ridge[i] <- mean((y[-k] - ridge.pred.test)^2)
  test.mse.lasso[i] <- mean((y[-k] - lasso.pred.test)^2)
  
}


# Plotting results --------------------------------------------------------

#Prepping plotting data
gdat <- data.frame(lambdas,
                   train.mse.ridge,train.mse.lasso,
                   test.mse.ridge,test.mse.lasso)
gdat$loglambas <- log(gdat$lambdas)

#Creating plots
ridgeplot <- ggplot(gdat) + 
  geom_path(aes(x = loglambas, y = train.mse.ridge, color = "Train")) +
  geom_path(aes(x = loglambas, y = test.mse.ridge, color = "Test")) +
  theme_bw() + scale_color_manual(name = "MSE", values = c("Blue","Red")) +
  ggtitle("Ridge Regression")

lassoplot <- ggplot(gdat) + 
  geom_path(aes(x = loglambas, y = train.mse.lasso, color = "Train")) +
  geom_path(aes(x = loglambas, y = test.mse.lasso, color = "Test")) +
  theme_bw() + scale_color_manual(name = "MSE", values = c("Blue","Red")) +
  ggtitle("LASSO")

#Display with lambda increasing on x-axis
grid.arrange(ridgeplot,lassoplot,ncol=1)
#Display with model complexity increasing on x-axis (lambda decreases)
grid.arrange(ridgeplot + scale_x_reverse(),
             lassoplot + scale_x_reverse(),ncol=1)


#Optimal lambdas according to this procedure:

lambdas[which.min(test.mse.ridge)]
lambdas[which.min(test.mse.lasso)]

log(lambdas[which.min(test.mse.ridge)])
log(lambdas[which.min(test.mse.lasso)])


# Additional Notes --------------------------------------------------------

#If outcome is binary (i.e. classification rather than regression):
yb <- as.numeric(y > median(y))
ridge.mods.bin <- glmnet(x = x, y = yb, alpha = 0, lambda = lambdas,
                         family = "binomial")
lasso.mods.bin <- glmnet(x = x, y = yb, alpha = 1, lambda = lambdas,
                         family = "binomial")

#Can now issue the same commands and run same processes on fitted objects
#as before.
plot(ridge.mods.bin, xvar = "lambda")
plot(lasso.mods.bin, xvar = "lambda")

#If user does not specify lambda(s), glmnet by default will create its
#own sequence using an algorithm that will generate a good spread
ridge.mods.alt <- glmnet(x = x, y = y, alpha = 0)
ridge.mods.alt$lambda
plot(ridge.mods.alt, xvar = "lambda")
