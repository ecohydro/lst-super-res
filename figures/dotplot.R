# makes figure from TWEEDS extended abstract 2023
# Anna Boser
library(ggplot2)
library(lfe)
library(dplyr)
library(tidyverse)

#############################################################################
# read in and format the RF data
#############################################################################

RF <- read.csv("~/Downloads/results_4.csv")

#remove square brackets around landcover
RF$landcover <- gsub("\\[|\\]|'", "", RF$landcover) 

# Remove brackets from the file column
RF$file <- gsub("\\[|\\]", "", RF$file)

# Create a new "run" column by extracting the desired string
# This is in order to calcualte clustered standard errors. 
RF <- RF %>%
  mutate(run = sub("^(.*?)_.*", "\\1", file),
         run = sub("[0-9]+$", "", run))
# RF$run <- RF$file # if you don't want clustered standard errors

#############################################################################
# read in and format the CNN data. This also has coarse metrics
#############################################################################

CNN <- read.csv("~/Downloads/val/prediction_metrics.csv")

# Create a new "run" column by extracting the desired string
# This is in order to calcualte clustered standard errors. 
CNN <- CNN %>%
  mutate(run = sub("^(.*?)_.*", "\\1", file),
         run = sub("[0-9]+$", "", run))
# CNN$run <- CNN$file if you don't want clustered standard errors

# mse to rmse
CNN$rmse_pred <- sqrt(CNN$mse_pred)
CNN$rmse_coarse <- sqrt(CNN$mse_coarse)

#############################################################################
# helper function gets the mean as well as min and max CI values from an felm model
#############################################################################

df_from_felm <- function(input_lm, model, metric, name_length = 10, var="landcover"){
  df <- as.data.frame(cbind(input_lm$coefficients, input_lm$cse))
  colnames(df) <- c("value", "se")
  df$min <- df$value-(1.96*df$se)
  df$max <- df$value+(1.96*df$se)
  df[,var] <- substring(rownames(df), name_length)
  df$model <- model
  df$metric <- metric
  return(df)
}

#############################################################################
# summarize function takes the columns and metrics you want to summarize and 
# calcualtes the mean and CI values for each
#############################################################################

summarize_metrics <- function(df, columns, model, metrics){
  
  summary_df <- data.frame()
  
  for(i in 1:length(columns)){
    df$metric <- df[columns[i]]
    lc <- df_from_felm(felm(unlist(metric)~landcover-1 | 0 | 0 | run, data = df), model = model, metric = metrics[i])
    all <- df_from_felm(felm(unlist(metric)~1 | 0 | 0 | run, data = df), model = model, metric = metrics[i])
    all$landcover <- "All"
    lc_all <- rbind(lc, all)
    summary_df <- rbind(summary_df, lc_all)
  }
  
  return(summary_df)
}

#############################################################################
# get summaries for each model and metric you want
#############################################################################

RF_summary <- summarize_metrics(df = RF, 
                                columns = c("r2_pred", "rmse_pred"), 
                                model = "RF", 
                                metrics = c("r2", "rmse"))

CNN_summary <- summarize_metrics(df = CNN, 
                                columns = c("r2_pred", "rmse_pred"), 
                                model = "CNN", 
                                metrics = c("r2", "rmse"))

Coarse_summary <- summarize_metrics(df = CNN, 
                                columns = c("r2_coarse", "rmse_coarse"), 
                                model = "Coarse", 
                                metrics = c("r2", "rmse"))

# merge summaries together into a single data frame
plot_data <- rbind(RF_summary, CNN_summary, Coarse_summary)

#############################################################################
# plots
#############################################################################

# so that they're presented in the right order in the figure, 
# make the model a factor in order coarse, RF, CNN
plot_data$model <- as.factor(plot_data$model)
plot_data$model <- relevel(plot_data$model, 'RF')
plot_data$model <- relevel(plot_data$model, 'Coarse')

# so that they're presented in the right order in the figure, 
# make the lct a factor with "all" first
plot_data$landcover <- as.factor(plot_data$landcover)
plot_data$landcover <- relevel(plot_data$landcover, 'All')

# specify colors you want
landcover_colors <- c("Natural" = "palegreen3", "Urban" = "indianred1", "Agricultural" = "royalblue1", "All" = "black")

# plot all metrics (not yet beautified)
ggplot(plot_data) +
  geom_point(aes(x = model, y = value, color = landcover), position=position_dodge(width=.5)) +
  geom_linerange(aes(x = model, ymin = min, ymax = max, color = landcover), position=position_dodge(width=.5)) + 
  facet_grid(rows = vars(metric), scales = "free") + 
  scale_color_manual(values = landcover_colors) +  # Apply custom colors
  theme_bw()

# only plot the R2 -- beautified
ggplot(filter(plot_data, metric == "r2")) +
  geom_point(aes(x = model, y = value, color = landcover), position=position_dodge(width=.5)) +
  geom_linerange(aes(x = model, ymin = min, ymax = max, color = landcover), position=position_dodge(width=.5)) + 
  xlab("") + 
  ylab("R2") + 
  labs(x = "",
       y = expression(R^2),
       color = "Landcover") + 
  scale_color_manual(values = landcover_colors) +  # Apply custom colors
  theme_bw()
