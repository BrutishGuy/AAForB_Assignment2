source("./code/00_libraries.R")

### LOAD AND CLEAN DATA
laptop_data_train_df <- read.csv("./data/train.csv")

## use janitor package to clean the data
laptop_data_train_df <- janitor::clean_names(laptop_data_train_df)


