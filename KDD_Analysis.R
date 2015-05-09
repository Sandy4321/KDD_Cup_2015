# ==============================================================================
# LIBRARIES
# ==============================================================================

library(dplyr)
library(readr)
library(lubridate)
library(magrittr)
library(ggplot2)
library(ROCR)


# ==============================================================================
# LOADING DATA
# ==============================================================================

enroll_df <- read_csv("../enrollment_train.csv")
log_df <- read_csv("../log_train.csv", col_types = list(time = col_character()))
object_df <- read_csv("../object.csv")
label_df <- read_csv("../truth_train.csv",
                     col_names = c("enrollment_id", "dropout"))

# Format time column to make a POSIXct object
log_df %<>% mutate(time = ymd_hms(gsub("T", " ", time)),
                   event_date = as.Date(time))

# Don't have label for enroll id 139669. Remove it from data
log_df %<>% filter(enrollment_id != 139669)
enroll_df %<>% filter(enrollment_id != 139669)

# Add labels to enroll_df
enroll_df <- inner_join(enroll_df, label_df)

# Change spelling error nagivate to navigate!
log_df[log_df$event == "nagivate",]$event = "navigate"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Get the AUC
calcAUC <- function(predcol, outcol) {
    perf <- performance(prediction(predcol, outcol == 1), "auc")
    as.numeric(perf@y.values)
}

# Create some summary features
create_summary <- function(df) {
    summary_df <- 
        df %>%
        group_by(enrollment_id, username, course_id) %>%
        summarise(num_videos = sum(ifelse(event == "video", 1, 0)),
                  num_navigate = sum(ifelse(event == "navigate", 1, 0)),
                  num_access = sum(ifelse(event == "access", 1, 0)),
                  num_problem = sum(ifelse(event == "problem", 1, 0)),
                  num_page_close = sum(ifelse(event == "page_close", 1, 0)),
                  num_discussion = sum(ifelse(event == "discussion", 1, 0)),
                  num_wiki = sum(ifelse(event == "wiki", 1, 0)),
                  num_events = n(),
                  num_events_lst_wk = sum(
                      ifelse(time >= (max(time) - as.difftime(1,
                                                              units = "weeks")), 
                             1, 0)
                  ),
                  num_access_lst_wk = sum(
                      ifelse(
                          time >= (max(time) - as.difftime(1,
                                                           units = "weeks")) &
                                 event == "access", 1, 0
                          )
                  ),
                  num_access_lst2_wk = sum(
                      ifelse(
                          time >= (max(time) - as.difftime(2,
                                                           units = "weeks")) &
                                 event == "access", 1, 0
                          )
                  ),
                  days_spent = as.numeric(difftime(max(time), min(time),
                                                   units = "days")),
                  unique_days_accessed = n_distinct(event_date)
        ) %>%
        ungroup
    summary_df
}


# ==============================================================================
# TRAIN TEST SPLIT
# ==============================================================================

# Split into test and train
set.seed(729375)
summary_df$rgroup <- runif(nrow(summary_df))
train <- summary_df %>% filter(rgroup <= 0.8)
test <- summary_df %>% filter(rgroup > 0.8)
train$rgroup <- NULL
test$rgroup <- NULL


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# Let's see the trend of dropouts with unique number of days MOOC was accessed
ggplot(train, aes(x = unique_days_accessed, y = as.numeric(dropout))) +
    geom_point(position = position_jitter(w = 0.05, h = 0.05)) +
    geom_smooth() +
    xlab("Number of unique days MOOC was accessed") +
    ylab("Dropout") +
    ggtitle("Trend of dropouts with unique number of days accessed")


# ==============================================================================
# MODELS
# ==============================================================================

# Let's fit a decision tree
library(rpart)
library(rpart.plot)

tree.model <- rpart(as.factor(dropout) ~ ., data = train,
                    control = rpart.control(maxdepth = 5))
prp(tree.model)

pred <- predict(tree.model, newdata = test)
pred <- ifelse(pred[, 2] >= 0.5, 1, 0)
table(pred, test$dropout)
calcAUC(pred, test$dropout)

# Logistic regression
logit.model <- glm(as.factor(dropout) ~ ., data = train[,-8],
                   family = "binomial")
summary(logit.model)

pred.logit <- predict(logit.model, newdata = test, type = "response")
pred.logit <- ifelse(pred.logit >= 0.5, 1, 0)
table(pred.logit, test$dropout)
calcAUC(pred.logit, test$dropout)


# ==============================================================================
# SUBMISSIONS
# ==============================================================================

# Load actual test data
enroll_test_df <- read_csv("../test/enrollment_test.csv")
log_test_df <- read_csv("../test/log_test.csv",
                        col_types = list(time = col_character()))
# Format time column to make a POSIXct object
log_test_df %<>% mutate(time = ymd_hms(gsub("T", " ", time)),
                   event_date = as.Date(time))
summary_test_df <- create_summary(log_test_df)
summary_test_df %<>% left_join(enroll_test_df, .)
model_test_df  <- summary_test_df %>% select(num_videos:unique_days_accessed)

pred <- predict(tree.model, newdata = model_test_df)
pred <- ifelse(pred[, 2] >= 0.5, 1, 0)
submit_df <- data.frame(enroll_id = enroll_test_df$enrollment_id, 
                        prediction = pred)

pred.logit <- predict(logit.model, newdata = model_test_df, type = "response")
pred.logit <- ifelse(pred.logit >= 0.5, 1, 0)
pred.logit <- ifelse(is.na(pred.logit), 1, 0)
submit_df <- data.frame(enroll_id = enroll_test_df$enrollment_id, 
                        prediction = pred.logit)

write_csv(submit_df, "../Submissions/submission1_logit_May9.csv", col_names = F)
