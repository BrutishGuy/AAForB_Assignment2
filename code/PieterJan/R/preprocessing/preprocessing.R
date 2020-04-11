
# clear environment
# rm(list = ls(all.names = TRUE))
# model building libaries
library(recipes)  # to define preprocessing steps
library(rlang)
library(rsample)
# visualize
library(ggplot2)
library(tidyverse)
library(patchwork)
library(plotly) # interactive vis


# load global parameters
source("code/global_params.R")


# seed for reproduceability
set.seed(100)

# import data and rename id
df <- read.csv("data/train.csv",sep = ";", fileEncoding="UTF-8-BOM")
# split in train and validation
# on this traning data I use cross validation to tune the hyper parameters
# the validation data is used as sort of safeguard to prevent ourselfves from overfitting
# 2/3 training, 1/3 validation
split_scheme <- initial_split(df, prop = 2/3)
train_train <- training(split_scheme)
train_validation <- testing(split_scheme)

# store data
write_excel_csv(x = train_train, path = "data/train_train.csv", delim = ';')
write_excel_csv(x = train_validation, path =  "data/train_validation.csv", delim = ';')

# check distributions
# training min price
p_train_min_p <- split_scheme %>%
  training() %>% select(min_price) %>% 
  ggplot(.,aes(min_price)) + geom_histogram(bins = 30) +
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "min. price") + ggtitle("Train") + THEME

# training max price
p_train_max_p <- split_scheme %>%
  training() %>% select(max_price) %>% 
  ggplot(.,aes(max_price)) + 
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "max. price") + ggtitle("Train") + THEME


# validation min price
p_val_min_p <- split_scheme %>%
  testing() %>% select(min_price) %>% 
  ggplot(.,aes(min_price)) + geom_histogram(bins = 30) +
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "min. price") + ggtitle("Validation") + THEME

# validation max price
p_val_max_p <- split_scheme %>%
  testing() %>% select(max_price) %>% 
  ggplot(.,aes(max_price)) + geom_histogram(bins = 30) +
  geom_histogram(bins = 30, color = cbPalette[7], fill="white") +
  labs(x = "max. price") + ggtitle("Validation") + THEME 


# make figure (this uses the libraray patchwork ==> super cool library to arrange figures)
(p_train_min_p + p_train_max_p)/
  (p_val_min_p + p_val_max_p)


# create recips: this is the preprocessing pipeline
# will be applied to every fold in the cross validation and also to the validation
# and test set
# the steps are performed in order so be careful
# also when you do transformation on the target variable (such as standardizing) don't
# foget to backtransform to the orignal space as the predictions are expected in the
# orignal uinits


# create a seperate model for 
#  1) min price
#  2) max price

# preprocessing steps min price

recipe_steps_min_p <- function(data=train){
  
  # remove max_price already here
  data %>%  select(-max_price) %>%
  # specify recipe  
  recipe(min_price ~., data = .) %>%
  # feature we remove 
  step_rm(name, base_name, cpu_details, id) %>%
  # log transformation on target since it's right skewed and normalize
  step_log(all_outcomes()) %>%
  step_normalize(all_outcomes()) %>%
  # take the median for imputation of missing values in the feature space
  step_medianimpute(all_numeric()) %>%
  # normalize for building the knn model
  step_normalize(all_numeric(),-all_outcomes()) %>%
  # for nominal missing values
  step_knnimpute(all_nominal()) %>%
  # to make data more noramlly dist.
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>%
  # for previously unseen factors
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>%
  # one hot encoding
  step_dummy(all_nominal()) %>%
  # normalize data (mean = 0 and sd 1)
  step_normalize(all_predictors()) %>%
  # remove near zero variance features
  step_nzv(all_predictors())
}


# preprocessing steps max price
recipe_steps_max_p <- function(data=train_train){ 
  
  # remove max_price already here
  data %>%  select(-min_price) %>%
  # specify recipe  
  recipe(max_price ~., data = .) %>%
  # features we remove 
  step_rm(name, base_name, cpu_details, id) %>%
  # log transformation on target since it's right skewed
  step_log(all_outcomes()) %>%
  step_normalize(all_outcomes()) %>%
  # take the median for imputation of numeric missing values in the feature space
  step_medianimpute(all_numeric()) %>%
  # normalize for building the knn model
  step_normalize(all_numeric(),-all_outcomes()) %>%
  # for nominal missign values
  step_knnimpute(all_nominal()) %>%
  # to make data more noramlly dist.
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>%
  # for previously unseen factors
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>%
  # one hot encoding
  step_dummy(all_nominal()) %>%
  # normalize data (mean = 0 and sd 1)
  step_normalize(all_predictors()) %>%
  # remove near zero variance features
  step_nzv(all_predictors())
}

# just a quick check
check_juice  <- 
  prep(recipe_steps_min_p(data=train_train), 
      training = train_train, 
      verbose = TRUE
      ) %>% 
  juice() 

colnames(check_juice)
ncol(check_juice)
nrow(check_juice)


# mean zero and std 1
check_juice %>% sapply(., mean)
check_juice %>% sapply(., sd)

