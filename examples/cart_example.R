library(tree)


# Load data ---------------------------------------------------------------

dat <- read.csv("asylum_conjoint_data.csv")
head(dat)


# Data split --------------------------------------------------------------

set.seed(123)
n <- nrow(dat)
v <- sample(n,3000,replace = FALSE)
train.dat <- dat[v,]
test.dat <- dat[-v,]


# Regression tree ---------------------------------------------------------

rtree1 <- tree(NAsySeekAccepted ~ ., data = train.dat)
summary(rtree1)
plot(rtree1)
text(rtree1,pretty = 0)
rtree1

rtree2 <- tree(NAsySeekAccepted ~ ., data = train.dat, mindev = 0.002)
summary(rtree2)
plot(rtree2)
rtree2

rtree3 <- tree(NAsySeekAccepted ~ ., data = train.dat, mindev = 0)
summary(rtree3)
plot(rtree3)

rtest1 <- predict(rtree1, newdata = test.dat, type = "vector")
rtest2 <- predict(rtree2, newdata = test.dat, type = "vector")
rtest3 <- predict(rtree3, newdata = test.dat, type = "vector")

mean((test.dat$NAsySeekAccepted - rtest1)^2)
mean((test.dat$NAsySeekAccepted - rtest2)^2)
mean((test.dat$NAsySeekAccepted - rtest3)^2)


# Classification data -----------------------------------------------------

train.dat.class <- train.dat
test.dat.class <- test.dat

train.dat.class$NAsySeekAccepted <- 
  as.factor(train.dat.class$NAsySeekAccepted > 5)
test.dat.class$NAsySeekAccepted <- 
  as.factor(test.dat.class$NAsySeekAccepted > 5)

names(train.dat.class)[1] <- names(test.dat.class)[1] <-
  "MajorityAccepted"

head(train.dat.class)


# Classification trees ----------------------------------------------------

ctree1 <- tree(MajorityAccepted ~ ., data = train.dat.class)
summary(ctree1)
plot(ctree1)
text(ctree1,pretty = 0)

ctree2 <- tree(MajorityAccepted ~ ., data = train.dat.class, mindev = 0.002)
summary(ctree2)
plot(ctree2)

ctree3 <- tree(MajorityAccepted ~ ., data = train.dat.class, mindev = 0)
summary(ctree3)
plot(ctree3)

ctest1.predprobs <- predict(ctree1, newdata = test.dat.class, type = "vector")
head(ctest1.predprobs)

ctest1 <- predict(ctree1, newdata = test.dat.class, type = "class")
ctest2 <- predict(ctree2, newdata = test.dat.class, type = "class")
ctest3 <- predict(ctree3, newdata = test.dat.class, type = "class")

mean(test.dat.class$MajorityAccepted != ctest1)
mean(test.dat.class$MajorityAccepted != ctest2)
mean(test.dat.class$MajorityAccepted != ctest3)
