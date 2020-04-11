# load data
library(naniar)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(plotly)
library(GGally)


# Global parameters ---------------------------------------
# colors
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", 
               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

THEME <- theme_minimal()
LEGEND <- theme(legend.title = element_blank())

# import data and rename id
train <- read.csv("data/train.csv",sep = ";", fileEncoding="UTF-8-BOM")

# summary stats
train %>% anyNA()
train %>% summary()

# missing values
train %>% vis_miss()
train %>% gg_miss_var()


# have a look at the  target variable first
# both are right skewed ==> transformation
p_min_price <- train %>% ggplot(.,aes(min_price)) + 
  geom_histogram(color=cbPalette[4], fill="white", bins = 40) +
  THEME + labs(x = "Minimum price")
 
p_max_price <- train %>% ggplot(.,aes(max_price)) + 
  geom_histogram(color=cbPalette[7], binwidth = 40, fill="white", bins = 40) +
  THEME + labs(x = "Maxiumum price")


# combine min and maxprice
(p_min_price/p_max_price)


# overview plot
dplyr::select_if(train, is.numeric) %>% ggpairs()


# TODO look at factor variables













