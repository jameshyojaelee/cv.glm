
# Load and inspect data ---------------------------------------------------

dat <- read.csv("asylum_data_spain.csv")
head(dat)
summary(dat)

#AsylumHome: measure of support for increasing or decreasing number of 
#people granted asylum in home country: 
#greatly decrease (-2), decrease (-1), neither increase nor decrease (0), 
#increase (1), greatly increase (2).


# Split into training and test sets ---------------------------------------

n.total <- nrow(dat)
prop.train <- 0.8

set.seed(54321)
k <- sample(1:n.total, size = round(n.total*prop.train), replace = FALSE)
#Note: For randomly splitting data, replace = FALSE!
#(In contrast to bootstrap resampling, where replace = TRUE)

train.dat <- dat[k,]
test.dat <- dat[-k,]


# Train -------------------------------------------------------------------

mod1 <- lm(AsylumHome ~ IdeoScale + Female + Age + 
             Employed + EISCED + IncomeDecile,
           data = train.dat)
summary(mod1)


# Training Error ----------------------------------------------------------

mean((train.dat$AsylumHome - predict(mod1))^2)


# Test Error --------------------------------------------------------------

mean((test.dat$AsylumHome - predict(mod1, newdata = test.dat))^2)


# Comparing Different Models ----------------------------------------------

# First model (above)
#Training MSE:
mean((mod1$model$AsylumHome - predict(mod1))^2)
#Test MSE:
mean((test.dat$AsylumHome - predict(mod1, newdata = test.dat))^2)


# More flexible model

mod2 <- lm(AsylumHome ~ IdeoScale + I(IdeoScale^2) + 
             Female + Age + I(Age^2) +
             Employed + EISCED + factor(IncomeDecile),
           data = train.dat)
#Training MSE:
mean((mod2$model$AsylumHome - predict(mod2))^2)
#Test MSE:
mean((test.dat$AsylumHome - predict(mod2, newdata = test.dat))^2)


# Even more flexible model

mod3 <- lm(AsylumHome ~ Female*factor(IdeoScale) +
             Female*Age + Female*I(Age^2) + Female*I(Age^3) +
             Female*I(Age^4) +
             factor(EISCED) + Employed*factor(IncomeDecile),
           data = train.dat)
#Training MSE:
mean((mod3$model$AsylumHome - predict(mod3))^2)
#Test MSE:
mean((test.dat$AsylumHome - predict(mod3, newdata = test.dat))^2)
