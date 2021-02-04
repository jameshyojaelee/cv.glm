library(glmnet)
library(ISLR)


# K-Fold Cross-Validation -------------------------------------------------

#Loading data
dat <- read.csv("asylum_data_spain.csv")
head(dat)
summary(dat)
dat <- na.omit(dat)

#Choose number of folds
k <- 10

#Create partitions, one possible method
n <- nrow(dat)
folds <- c( rep(seq(k),n%/%k) , seq(n%%k) )
set.seed(131)
folds <- sample(folds,length(folds))
table(folds)

#Implement partitions, one possible method
dat$yhat <- NA
data.folds <- list()
for (i in 1:k){
  data.folds[[i]] <- dat[folds == i,]
}
str(data.folds)

#Implement CV
for (i in 1:k){
  
  train.dat <- do.call("rbind",data.folds[-i]) # -i means all except for i
  cv.mod <- lm(AsylumHome ~ IdeoScale + Female + Age + 
                 Employed + EISCED + IncomeDecile, 
               data = train.dat)
  data.folds[[i]]$yhat <- predict(cv.mod, newdata = data.folds[[i]])
  rm(train.dat,cv.mod)
  
}

#Create final dataset with CV predictions
CV.dat <- do.call("rbind",data.folds)
head(CV.dat)

#CV estimate of MSE
mean((CV.dat$AsylumHome - CV.dat$yhat)^2)


# LASSO with Cross-Validation ---------------------------------------------

#Hitters data
data("Hitters")
head(Hitters)
Hitters <- na.omit(Hitters)

#Replace salary variable with binary indicator for top-echelon salary
hist(Hitters$Salary)
Hitters$TopSalary <- as.numeric(Hitters$Salary > 
                                  quantile(Hitters$Salary, probs = 0.8))
mean(Hitters$TopSalary)
Hitters <- Hitters[,-which(names(Hitters) == "Salary")]
head(Hitters)

#Separating response vector
y <- Hitters$TopSalary

#Putting all other variables in Hitters data
x <- model.matrix(TopSalary ~ ., data = Hitters)[,-1]


#Recall that we could fit a sequence of classification lasso mods as such:
lasso.mods <- glmnet(x = x, y = y, alpha = 1, family = "binomial")


#To automatically apply cross-validation, we simply use:
k <- 10
set.seed(131)
lasso.cv <- cv.glmnet(x = x, y = y, alpha = 1, family = "binomial", 
                      nfolds = k)

#We can see sequence of lambda values and CV error
lasso.cv$lambda
lasso.cv$cvm

plot(lasso.cv) #Note the CV error used

log(lasso.cv$lambda.min)
lasso.cv$lambda.min

log(lasso.cv$lambda.1se)
lasso.cv$lambda.1se


#Predicting with optimal model: let's try for first 5 rows
opt.lambda <- lasso.cv$lambda.min
predict(lasso.cv, newx = x[1:5,], s = opt.lambda, type = "response")

#What about predicting for one row?
predict(lasso.cv, newx = x[1,], s = opt.lambda, type = "response")

#Must force R to keep the row in matrix form
#so the predict function will properly read it:
predict(lasso.cv, newx = x[1, , drop = FALSE], s = opt.lambda, 
        type = "response")



#To predict with optimal model, you could alternatively
#retrain a simple glmnet object (not cv.glmnet),
#which can be useful if you stored the lambda and you want
#to create the optimal model again later without re-running
#the CV process.
lasso.opt <- glmnet(x = x, y = y, alpha = 1, family = "binomial", 
                      lambda = opt.lambda)
predict(lasso.opt, newx = x[1:5,], s = opt.lambda, type = "response")

#Compare this to the previous output:
predict(lasso.cv, newx = x[1:5,], s = opt.lambda, type = "response")

#Why are the predictions different?




#The discrepancy has to do with differences in the way in which
#the LASSO fitting optimization algorithm is applied in
#glmnet vs. cv.glmnet
#Optimization algorithm convergence precision can be increased
#To address the discrepancy above:

lasso.cv <- cv.glmnet(x = x, y = y, alpha = 1, family = "binomial", 
                      nfolds = k, thresh = 1e-20)
opt.lambda <- lasso.cv$lambda.min
predict(lasso.cv, newx = x[1:5,], s = opt.lambda, type = "response")

lasso.opt <- glmnet(x = x, y = y, alpha = 1, family = "binomial", 
                    lambda = opt.lambda, thresh = 1e-20)
predict(lasso.opt, newx = x[1:5,], s = opt.lambda, type = "response")



# Misc items --------------------------------------------------------------

#Using classification error rate as loss function:
lasso.cv.class.error <- cv.glmnet(x = x, y = y, alpha = 1, family = "binomial", 
                                  nfolds = k, type.measure = "class")
plot(lasso.cv.class.error)

#Compared to using binomial deviance:
plot(lasso.cv)

#Matrix summarization functions
colMeans(x)
colSums(x)
rowMeans(x)
rowSums(x)

#Don't forget about the drop command in matrix brackets
x[1,]
x[1, , drop = FALSE]
str(x[1,])
str(x[1, , drop = FALSE])
