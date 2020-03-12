
# libraries

# random forest
library(ranger)

# lasso ridge and elastic net
library(glmnet)

# boosting
library(xgboost)


# own functions
source("code/functions/invert_transformations_target.R")
source("code/functions/check_predictions.R")


# load preprocessing
source("code/preprocessing/preprocessing.R")



# read in test data
test <- read.csv("data/test.csv", sep = ",", fileEncoding="UTF-8-BOM")

# prepare test data using the same preprocessing steps on the training data

# 1) MIN PRICE
# train_rec_min_p <- prep(recipe_steps_min_p(data=df), training = df, fresh = TRUE, verbose = TRUE)
# test_data_min_p  <- bake(train_rec_min_p, new_data = test)
# 
# # 2) MAX PRICE
# train_rec_max_p <- prep(recipe_steps_max_p(data=df), training = df, fresh = TRUE, verbose = TRUE)
# test_data_max_p  <- bake(train_rec_max_p, new_data = test)


# invert transformation on the target variable
# compute everything on the training data (train.csv) == df

# 1 MIN PRICE
mean_allData_min_p <- df %>% select(min_price) %>% pull %>% log %>% mean
sd_allData_min_p <- df %>% select(min_price) %>% pull %>% log %>% sd

# MAX PRICE
mean_allData_max_p <- df %>% select(max_price) %>% pull %>% log %>% mean
sd_allData_max_p <- df %>% select(max_price) %>% pull %>% log %>% sd


#------------------------------------------------------------------------------
# # SUBMISSION FOR GLMNET
#------------------------------------------------------------------------------

# if you want to retrain the whole model
#source("code/models/linear_model.R")

# load trained model
glmnet_test_min_p <- readRDS("code/models/save_models/glmnet_min_p.rds")
glmnet_test_max_p <- readRDS("code/models/save_models/glmnet_max_p.rds")


# make predictions

# 1) MIN PRICE

glmnet_pred_test_min_p <- 
  predict(glmnet_test_min_p, new_data = test) %>%
  dplyr::rename(fitted = .pred)


# invert potentially transformation on target
glmnet_pred_test_min_p <- 
  invert_transformations_target(
  pred = glmnet_pred_test_min_p,
  mean = mean_allData_min_p,
  sd = sd_allData_min_p,
  # this is just identity mapping, 
  # suppose you did a log transformation 
  # on the target you can give FUN = exp()
  FUN = exp
)


# 2) Max PRICE
glmnet_pred_test_max_p <- 
  predict(glmnet_test_max_p, new_data = test) %>%
  dplyr::rename(fitted = .pred)

# invert potentially transformation on target
glmnet_pred_test_max_p <- 
  invert_transformations_target(
  pred = glmnet_pred_test_max_p,
  mean = mean_allData_max_p,
  sd = sd_allData_max_p,
  # this is just identity mapping, 
  # suppose you did a log transformation 
  # on the target you can give FUN = exp
  FUN = exp
)
  

# create a tibble and save it
test_sub_glmnet <- 
  tibble(ID = test$id,
  MIN = glmnet_pred_test_min_p$fitted_inverted,
  MAX = glmnet_pred_test_max_p$fitted_inverted)  


# visualize the distributions of the predictions on the test set and compare it 
# with the actual values on the data we have (this is just a check)



# distributions on training data: acutal values

# 1) MIN PRICE
p_test_min_p <- 
  check_predictions(
    df=df,var = "min_price",
    xlab = "Min. price train: actual") + THEME

# 2) MAX PRICE
p_test_max_p <- 
  check_predictions(df=df,var = "max_price",
  xlab = "Max. price train: actual") + THEME

# distribution of the predictions: test data

# 1) MIN PRICE
p_glmnet_pred_test_min_p <- 
  check_predictions(
    df=test_sub_glmnet,var = "MIN", 
    xlab = "Min. price test: fitted") + THEME

# 2) MAX PRICE
p_glmnet_pred_test_max_p <- 
  check_predictions(df=test_sub_glmnet,var = "MAX",
                    xlab = "Max. price test: fitted") + THEME

((p_test_min_p + p_test_max_p)/
    (p_glmnet_pred_test_min_p + p_glmnet_pred_test_max_p))



# predictions where the mimimum is greater than the maximum
test_sub_glmnet %>% mutate(
  diff = MIN - MAX
) %>% filter(diff > 0)


# take the min and max for each row
test_sub_glmnet <- tibble(
  ID = test$id,
  MIN = apply(test_sub_glmnet %>% select(MIN,MAX), 1,min), # take min 
  MAX = apply(test_sub_glmnet %>% select(MIN,MAX), 1,max) # take max 
  ) 


# predictions where the mimimum is greater than the maximum
# there should be no cases
test_sub_glmnet %>% 
  mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# write submission 
test_sub_glmnet %>% 
  write.csv("output/predictions_test/glmnet.csv", row.names=FALSE)


#------------------------------------------------------------------------------
# # SUBMISSION FOR BOOSTING
#------------------------------------------------------------------------------

# if you want to retrain the whole model
#source("code/models/boost_model.R")

# load trained model
boost_test_min_p <- readRDS("code/models/save_models/boost_min_p.rds")
boost_test_max_p <- readRDS("code/models/save_models/boost_max_p.rds")


# make predictions

# 1) MIN PRICE
boost_pred_test_min_p <- 
  predict(boost_test_min_p, new_data  = test) %>%
  dplyr::rename(fitted = .pred)

# invert potentially transformation on target
boost_pred_test_min_p <- 
  invert_transformations_target(
    pred = boost_pred_test_min_p,
    mean = mean_allData_min_p,
    sd = sd_allData_min_p,
    # this is just identity mapping, 
    # suppose you did a log transformation 
    # on the target you can give FUN = exp
    FUN = exp
  )


# 2) Max PRICE
boost_pred_test_max_p <- 
  predict(boost_test_max_p, new_data = test) %>%
  dplyr::rename(fitted = .pred)

# invert potentially transformation on target
boost_pred_test_max_p <- 
  invert_transformations_target(
    pred = boost_pred_test_max_p,
    mean = mean_allData_max_p,
    sd = sd_allData_max_p,
    # this is just identity mapping, 
    # suppose you did a log transformation 
    # on the target you can give FUN = exp()
    FUN = exp
  )


# create a tibble and save it
test_sub_boost <- 
  tibble(ID = test$id,
         MIN = boost_pred_test_min_p$fitted_inverted,
         MAX = boost_pred_test_max_p$fitted_inverted)  


# visualize the distributions of the predictions on the test set and compare it 
# with the actual values on the data we have (this is just a check)



# distributions on training data: acutal values

# 1) MIN PRICE
p_test_min_p <- 
  check_predictions(
    df=df,var = "min_price",
    xlab = "Min. price train: actual") + THEME

# 2) MAX PRICE
p_test_max_p <- 
  check_predictions(df=df,var = "max_price",
    xlab = "Max. price train: actual") + THEME

# distribution of the predictions: test data

# 1) MIN PRICE
p_boost_pred_test_min_p <- 
  check_predictions(
    df=test_sub_boost,var = "MIN", 
    xlab = "Min. price test: fitted") + THEME

# 2) MAX PRICE
p_boost_pred_test_max_p <- 
  check_predictions(df=test_sub_boost,var = "MAX",
    xlab = "Max. price test: fitted") + THEME

((p_test_min_p + p_test_max_p)/
    (p_boost_pred_test_min_p + p_boost_pred_test_max_p))



# predictions where the mimimum is greater than the maximum
test_sub_boost %>% mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# take the min and max for each row
test_sub_boost <- tibble(
  ID = test$id,
  MIN = apply(test_sub_boost %>% select(MIN,MAX), 1,min), # take min 
  MAX = apply(test_sub_boost %>% select(MIN,MAX), 1,max)) # take max 


# predictions where the mimimum is greater than the maximum
# there should be no cases
test_sub_boost %>% 
  mutate(
    diff = MIN - MAX
  ) %>% filter(diff>0)


# write submission 
test_sub_boost %>% 
  write.csv("output/predictions_test/boost.csv", row.names=FALSE)


#------------------------------------------------------------------------------
# # SUBMISSION FOR RANDOM FOREST
#------------------------------------------------------------------------------

# if you want to retrain the whole model
#source("code/models/rf_model.R")

# load trained model
rf_test_min_p <- readRDS("code/models/save_models/rf_min_p.rds")
rf_test_max_p <- readRDS("code/models/save_models/rf_max_p.rds")


# make predictions

# 1) MIN PRICE
rf_pred_test_min_p <- 
  predict(rf_test_min_p, new_data = test) %>%
  dplyr::rename(fitted = .pred)


# invert potentially transformation on target
rf_pred_test_min_p <- 
  invert_transformations_target(
    pred = rf_pred_test_min_p,
    mean = mean_allData_min_p,
    sd = sd_allData_min_p,
    # this is just identity mapping, 
    # suppose you did a log transformation 
    # on the target you can give FUN = exp
    FUN = exp
  )


# 2) Max PRICE
rf_pred_test_max_p <- 
  predict(rf_test_max_p, new_data = test) %>%
  dplyr::rename(fitted = .pred)

# invert potentially transformation on target
rf_pred_test_max_p <- 
  invert_transformations_target(
    pred = rf_pred_test_max_p,
    mean = mean_allData_max_p,
    sd = sd_allData_max_p,
    # this is just identity mapping, 
    # suppose you did a log transformation 
    # on the target you can give FUN = exp()
    FUN = exp
  )


# create a tibble and save it
test_sub_rf <- 
  tibble(ID = test$id,
         MIN = rf_pred_test_min_p$fitted_inverted,
         MAX = rf_pred_test_max_p$fitted_inverted)  


# visualize the distributions of the predictions on the test set and compare it 
# with the actual values on the data we have (this is just a check)



# distributions on training data: acutal values

# 1) MIN PRICE
p_test_min_p <- 
  check_predictions(
    df=df,var = "min_price",
    xlab = "Min. price train: actual"
    ) + THEME

# 2) MAX PRICE
p_test_max_p <- 
  check_predictions(
    df=df,var = "max_price",
    xlab = "Max. price train: actual"
    ) + THEME

# distribution of the predictions: test data

# 1) MIN PRICE
p_rf_pred_test_min_p <- 
  check_predictions(
    df=test_sub_rf, var = "MIN", 
    xlab = "Min. price test: fitted"
    ) + THEME

# 2) MAX PRICE
p_rf_pred_test_max_p <- 
  check_predictions(
    df=test_sub_rf, var = "MAX",
    xlab = "Max. price test: fitted"
    ) + THEME


((p_test_min_p + p_test_max_p)/
    (p_rf_pred_test_min_p + p_rf_pred_test_max_p))



# predictions where the mimimum is greater than the maximum
test_sub_rf %>% mutate(
  diff = MIN - MAX
) %>% filter(diff>0)


# take the min and max for each row
# test_sub_rf <- tibble(
# ID = test$id,
# MIN = apply(test_sub_rf %>% select(MIN,MAX), 1,min), # take min 
# MAX = apply(test_sub_rf %>% select(MIN,MAX), 1,max)) # take max 


# predictions where the mimimum is greater than the maximum
# there should be no cases
test_sub_rf %>% 
  mutate(
    diff = MIN - MAX
  ) %>% filter(diff>0)


# write submission 
test_sub_rf %>% 
  write.csv("output/predictions_test/rf.csv", row.names=FALSE)


# quick comparison
test_sub_glmnet %>% head()
test_sub_boost %>% head()
test_sub_rf %>%head()


