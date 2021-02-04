library(foreign)
library(ggplot2)
library(pROC)


# Load data ---------------------------------------------------------------

#Data from "Ethnicity, Insurgency, and Civil War"
#by James D. Fearon and David D. Laitin
dat <- read.dta("repdata.dta")


# Clean data --------------------------------------------------------------

head(dat)
summary(dat)
table(dat$onset)
dat$onset[dat$onset == 4] <- 1


# Linear probability model ------------------------------------------------

reg1 <- lm(onset ~ warl + gdpenl + lpopl1 + lmtnest +
             ncontig + Oil + nwstate +
             instab + polity2l + ethfrac + relfrac, 
           data = dat)
summary(reg1)

#Note that ~300 observations were dropped due to missing data
#Ever want to know the final dataset used by the regression?
#Find it here:

regdata <- reg1$model
head(regdata)
nrow(regdata)


#Creating data.frame with true y and predicted prob:
rdat <- data.frame(outcome = reg1$model$onset,
                   predprob = predict(reg1))
head(rdat)

min(rdat$predprob)
max(rdat$predprob)

ggplot(rdat, aes(x = predprob)) + geom_histogram(bins=100) + 
  theme_bw()

ggplot(rdat, aes(x = predprob)) + geom_histogram(bins=50) + 
  facet_grid(outcome ~ ., scales = "free") + theme_bw()

ggplot(rdat, aes(x = predprob, y = outcome)) + geom_point() +
  theme_bw()


# Logistic regression -----------------------------------------------------

logit1 <- glm(onset ~ warl + gdpenl + lpopl1 + lmtnest +
              ncontig + Oil + nwstate +
              instab + polity2l + ethfrac + relfrac, 
            family = binomial(link = "logit"),
            data = dat)
summary(logit1)

#Final data used by logit
logitdata <- logit1$model
head(logitdata)
nrow(logitdata)


#Predicting with glm object

#"Link function" predictions
#i.e. in log-odds (logit) units
link.preds <- predict(logit1, type = "link")
link.preds[1:10]
predict(logit1)[1:10] #Type "link" is the default!

#The link predictions are simply x * betahat
x1 <- c(1,as.numeric(logitdata[1,-1])) #first observation (plus intercept)
logit1$coefficients %*% x1

#Reponse predictions
#i.e. in probability
resp.preds <- predict(logit1, type = "response")
resp.preds[1:10]

#The response predictions are the predicted probabilities
#I.e. the link predictions transformed through the
#inverse-logit (logistic) function
1 / (1 + exp(-(logit1$coefficients %*% x1)))


#Creating data.frame with true y and predicted probs:
ldat <- data.frame(outcome = logit1$model$onset,
                   predprob = predict(logit1, type = "response"))

min(ldat$predprob)
max(ldat$predprob)


ggplot(ldat, aes(x = predprob)) + geom_histogram(bins=100) + 
  theme_bw()

ggplot(ldat, aes(x = predprob)) + geom_histogram(bins=50) + 
  facet_grid(outcome ~ ., scales = "free") + theme_bw()

ggplot(ldat, aes(x = predprob, y = outcome)) + geom_point() +
  theme_bw()


# Comparing the two sets of predicted probabilities -----------------------

ggplot() + geom_point(aes(x = rdat$predprob, y = ldat$predprob)) +
  theme_bw() + xlab("LPM Predicted Probability") +
  ylab("Logistic Regression Predicted Probability") +
  geom_abline(slope = 1, intercept = 0, linetype = 2)


# Which is doing a better job producing useful predicted probabilities?

#Based on mean absolute error
mean(abs(rdat$outcome - rdat$predprob))
mean(abs(ldat$outcome - ldat$predprob))

#Based on mean squared error
mean((rdat$outcome - rdat$predprob)^2)
mean((ldat$outcome - ldat$predprob)^2)

#Based on AUC-ROC metric (we will discuss this in a later class)
auc(roc(rdat$outcome,rdat$predprob))
auc(roc(ldat$outcome,ldat$predprob))


# Classification ----------------------------------------------------------

t <- 0.5
ldat$class <- as.numeric(ldat$predprob > t)
mean(ldat$class == ldat$outcome)
table(ldat$class)

mean(ldat$outcome)
mean(1 - ldat$outcome)

#Be careful making classification predictions and assessing classification
#accuracy with rare event prediction!
