source("./code/01_load_data.R")

#### DATA CLEANING PRIOR TO FEATURE ENGINEERING ------

### Alter 'glossy' and 'matte' entries to match others and do before/after sanity check

unique(laptop_data_train_df$screen_surface)
laptop_data_train_df$screen_surface <- gsub("matte", "Matte", laptop_data_train_df$screen_surface)
laptop_data_train_df$screen_surface <- gsub("glossy", "Glossy", laptop_data_train_df$screen_surface)

laptop_data_train_df$screen_surface <- factor(laptop_data_train_df$screen_surface, 
                                              levels = unique(laptop_data_train_df$screen_surface))
unique(laptop_data_train_df$screen_surface)


#### FEATURE ENGINEERING PRIOR TO MODELLING ------

### Define HD resolution categories

laptop_data_train_df <- laptop_data_train_df %>%
  mutate(resolution_string = paste0(laptop_data_train_df$pixels_x, 'x', laptop_data_train_df$pixels_y)) %>%
  mutate(hd_resolution_category = ifelse(resolution_string %in% c("2304x1440", "2560x1600", "2880x1800"), "Retina", 
                                         ifelse(pixels_x >= 1200 & pixels_x <= 1600, "HD", 
                                                ifelse(pixels_x == 1920, "FullHD",
                                                       ifelse(pixels_x > 1920 & pixels_x < 3840, "QHD/UHD",
                                                              ifelse(pixels_x == 3840, "4K", "SD"))))))

## Test this feature, construct a simple EDA of how many of each category exist
## They should have a fair number of datapoints in each

laptop_data_train_df %>%
  group_by(resolution_string, hd_resolution_category) %>%
  summarize(count = n()) 

laptop_data_train_df %>%
  group_by(hd_resolution_category) %>%
  summarize(count = n()) 

### Define SSD size categories
laptop_data_train_df <- laptop_data_train_df %>%
  mutate(ssd_category = ifelse(ssd == 0,"None", 
                               ifelse(ssd < 64, "Small", 
                                      ifelse(ssd <= 256, "Medium", "Large"))))

## Test this feature, construct a simple EDA of how many of each category exist

laptop_data_train_df %>%
  group_by(ssd_category) %>%
  summarize(count = n()) 

### Define CPU frequency in GHz

## Use a regex to get the GHz out from cpu_details

laptop_data_train_df$cpu_frequency <- str_extract(laptop_data_train_df$cpu_details,"\\s+[0-9].[0-9][0-9]\\s+[gG][hH][zZ]|\\s+[0-9].[0-9]\\s+[gG][hH][zZ]")

## Test this feature, construct a simple EDA of how many of each category exist

testing <- laptop_data_train_df %>%
  group_by(cpu_frequency) %>%
  summarize(count = n()) 

## Continue to drill down, remove GHz string and group by percentiles 

laptop_data_train_df$cpu_frequency <- as.numeric(gsub(" GHz", "", laptop_data_train_df$cpu_frequency))

#laptop_data_train_df <- laptop_data_train_df %>%
#  mutate(cpu_frequency_categories)

laptop_data_train_df <- laptop_data_train_df %>%
  mutate(storage_category = ifelse(storage == 0,"None", 
                               ifelse(storage <= 256, "Small", 
                                      ifelse(storage <= 1028, "Medium", 
                                          ifelse(storage <= 2056, "Large", "Very Large")))))

modelling_dataset <- laptop_data_train_df %>%
  dplyr::select(-contains("i_id")) %>%
  dplyr::select(-contains("name")) %>%
  dplyr::select(-contains("pixels")) %>%
  dplyr::select(-contains("cpu_details")) %>%
  dplyr::select(-contains("os_details")) %>%
  dplyr::select(-contains("resolution_string")) %>%
  dplyr::select(-contains("price_diff"))
  
modelling_dataset_min_price <- modelling_dataset %>%
  dplyr::select(-contains("max_price")) 
  
simple_lm <- lm(min_price ~ ., data = modelling_dataset_min_price)
summary(simple_lm)

